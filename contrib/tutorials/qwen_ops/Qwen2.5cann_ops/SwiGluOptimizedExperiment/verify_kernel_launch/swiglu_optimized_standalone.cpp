#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_swiglu_optimized_kernel.h"
#include "swiglu_tiling.h"

#define CHECK_ACL(call)                                                   \
    do {                                                                  \
        const aclError ret = (call);                                      \
        if (ret != ACL_SUCCESS) {                                        \
            std::fprintf(stderr, "[ERROR] ACL fail %s:%d ret=%d\n",      \
                __FILE__, __LINE__, static_cast<int>(ret));              \
            return -1;                                                    \
        }                                                                 \
    } while (0)

struct Options {
    uint32_t rows = 128;
    uint32_t hidden = 1024;
    uint32_t blockDim = 8;
    uint32_t tileLength = 1024;
    uint32_t warmup = 10;
    uint32_t repeat = 50;
    uint32_t rounds = 5;
};

static bool ParseUint(const char *text, uint32_t *value)
{
    if (text == nullptr || value == nullptr) return false;
    char *end = nullptr;
    unsigned long v = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || v == 0 || v > UINT32_MAX) return false;
    *value = static_cast<uint32_t>(v);
    return true;
}

static bool ParseOptions(int argc, char **argv, Options *opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") return false;
        if (i + 1 >= argc) return false;
        const char *value = argv[++i];
        bool ok = false;
        if (key == "--rows") ok = ParseUint(value, &opt->rows);
        else if (key == "--hidden") ok = ParseUint(value, &opt->hidden);
        else if (key == "--block-dim") ok = ParseUint(value, &opt->blockDim);
        else if (key == "--tile-length") ok = ParseUint(value, &opt->tileLength);
        else if (key == "--warmup") ok = ParseUint(value, &opt->warmup);
        else if (key == "--repeat") ok = ParseUint(value, &opt->repeat);
        else if (key == "--rounds") ok = ParseUint(value, &opt->rounds);
        else return false;
        if (!ok) return false;
    }
    return true;
}

static double Mean(const std::vector<double> &values)
{
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

static double Median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return (n % 2 == 1) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static float SwigluRef(float gate, float up)
{
    return gate * (1.0f / (1.0f + std::exp(-gate))) * up;
}

int main(int argc, char **argv)
{
    Options opt;
    if (!ParseOptions(argc, argv, &opt)) {
        std::printf("Usage: %s --rows N --hidden N --block-dim N --tile-length N --warmup N --repeat N --rounds N\n", argv[0]);
        return argc > 1 ? -1 : 0;
    }

    const uint64_t total = static_cast<uint64_t>(opt.rows) * opt.hidden;
    if (total == 0 || total > UINT32_MAX) {
        std::fprintf(stderr, "[ERROR] total elements must be in (0, UINT32_MAX]\n");
        return -1;
    }
    const uint32_t totalSize = static_cast<uint32_t>(total);
    const uint32_t blockDim = std::max(1u, std::min(opt.blockDim, totalSize));
    const size_t bytes = static_cast<size_t>(totalSize) * sizeof(float);

    SwiGluTilingData tiling;
    tiling.totalSize = totalSize;
    tiling.coreNum = blockDim;
    tiling.elementsPerCore = (totalSize + blockDim - 1u) / blockDim;
    tiling.tileLength = opt.tileLength;

    std::printf("[INFO] SwiGLU optimized rows=%u hidden=%u total=%u blockDim=%u elementsPerCore=%u tileLength=%u\n",
        opt.rows, opt.hidden, totalSize, blockDim, tiling.elementsPerCore, tiling.tileLength);

    std::vector<float> gate(totalSize), up(totalSize), golden(totalSize), output(totalSize);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t i = 0; i < totalSize; ++i) {
        gate[i] = dist(rng);
        up[i] = dist(rng);
        golden[i] = SwigluRef(gate[i], up[i]);
    }

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *gateD = nullptr, *upD = nullptr, *outD = nullptr, *workspaceD = nullptr, *tilingD = nullptr;
    CHECK_ACL(aclrtMalloc(&gateD, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&upD, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&outD, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&workspaceD, 32, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&tilingD, sizeof(SwiGluTilingData), ACL_MEM_MALLOC_NORMAL_ONLY));

    CHECK_ACL(aclrtMemcpy(gateD, bytes, gate.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(upD, bytes, up.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingD, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));

    for (uint32_t i = 0; i < opt.warmup; ++i) {
        ACLRT_LAUNCH_KERNEL(swiglu_optimized_kernel)(blockDim, stream, gateD, upD, outD, workspaceD, tilingD);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<double> deviceUs;
    deviceUs.reserve(opt.rounds);
    for (uint32_t r = 0; r < opt.rounds; ++r) {
        aclrtEvent start = nullptr;
        aclrtEvent stop = nullptr;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&stop));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        for (uint32_t i = 0; i < opt.repeat; ++i) {
            ACLRT_LAUNCH_KERNEL(swiglu_optimized_kernel)(blockDim, stream, gateD, upD, outD, workspaceD, tilingD);
        }
        CHECK_ACL(aclrtRecordEvent(stop, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        float elapsedMs = 0.0f;
        CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, start, stop));
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(stop));
        const double us = static_cast<double>(elapsedMs) * 1000.0 / opt.repeat;
        deviceUs.push_back(us);
        std::printf("[BENCH] round=%u device=%.3f us\n", r + 1, us);
    }

    CHECK_ACL(aclrtMemcpy(output.data(), bytes, outD, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    double maxAbs = 0.0;
    double meanAbs = 0.0;
    for (uint32_t i = 0; i < totalSize; ++i) {
        const double diff = std::abs(static_cast<double>(output[i]) - golden[i]);
        maxAbs = std::max(maxAbs, diff);
        meanAbs += diff;
    }
    meanAbs /= totalSize;

    auto mm = std::minmax_element(deviceUs.begin(), deviceUs.end());
    std::printf("[RESULT] mean=%.3f median=%.3f min=%.3f max=%.3f us\n",
        Mean(deviceUs), Median(deviceUs), *mm.first, *mm.second);
    std::printf("[CHECK] max_abs_diff=%.8e mean_abs_diff=%.8e\n", maxAbs, meanAbs);
    std::printf("[%s] correctness threshold atol=1e-3\n", maxAbs <= 1e-3 ? "PASS" : "FAIL");

    CHECK_ACL(aclrtFree(gateD));
    CHECK_ACL(aclrtFree(upD));
    CHECK_ACL(aclrtFree(outD));
    CHECK_ACL(aclrtFree(workspaceD));
    CHECK_ACL(aclrtFree(tilingD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return maxAbs <= 1e-3 ? 0 : 1;
}
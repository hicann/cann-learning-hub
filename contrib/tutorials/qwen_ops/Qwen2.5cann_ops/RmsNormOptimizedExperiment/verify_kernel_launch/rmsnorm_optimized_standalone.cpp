#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_rmsnorm_optimized_kernel.h"

#pragma pack(push, 1)
struct RmsNormOptimizedTiling {
    uint32_t rows = 0;
    uint32_t hidden = 0;
    uint32_t coreNum = 1;
    uint32_t rowsPerCore = 0;
    float eps = 1e-6f;
    float invHidden = 1.0f;
};
#pragma pack(pop)

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
    uint32_t warmup = 10;
    uint32_t repeat = 50;
    uint32_t rounds = 5;
    float eps = 1e-6f;
};

static bool ParseUint(const char *text, uint32_t *value)
{
    if (text == nullptr || value == nullptr) {
        return false;
    }
    char *end = nullptr;
    unsigned long v = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || v == 0 || v > UINT32_MAX) {
        return false;
    }
    *value = static_cast<uint32_t>(v);
    return true;
}

static bool ParseFloat(const char *text, float *value)
{
    if (text == nullptr || value == nullptr) {
        return false;
    }
    char *end = nullptr;
    float v = std::strtof(text, &end);
    if (end == text || *end != '\0' || !(v > 0.0f)) {
        return false;
    }
    *value = v;
    return true;
}

static bool ParseOptions(int argc, char **argv, Options *opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            return false;
        }
        if (i + 1 >= argc) {
            return false;
        }
        const char *value = argv[++i];
        bool ok = false;
        if (key == "--rows") ok = ParseUint(value, &opt->rows);
        else if (key == "--hidden") ok = ParseUint(value, &opt->hidden);
        else if (key == "--block-dim") ok = ParseUint(value, &opt->blockDim);
        else if (key == "--warmup") ok = ParseUint(value, &opt->warmup);
        else if (key == "--repeat") ok = ParseUint(value, &opt->repeat);
        else if (key == "--rounds") ok = ParseUint(value, &opt->rounds);
        else if (key == "--eps") ok = ParseFloat(value, &opt->eps);
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

static void RmsNormRef(
    const std::vector<float> &input,
    const std::vector<float> &weight,
    std::vector<float> *golden,
    uint32_t rows,
    uint32_t hidden,
    float eps)
{
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t base = row * hidden;
        float squareSum = 0.0f;
        for (uint32_t col = 0; col < hidden; ++col) {
            const float x = input[base + col];
            squareSum += x * x;
        }
        const float scale = 1.0f / std::sqrt(squareSum / static_cast<float>(hidden) + eps);
        for (uint32_t col = 0; col < hidden; ++col) {
            (*golden)[base + col] = input[base + col] * scale * weight[col];
        }
    }
}

int main(int argc, char **argv)
{
    Options opt;
    if (!ParseOptions(argc, argv, &opt)) {
        std::printf("Usage: %s --rows N --hidden N --block-dim N --warmup N --repeat N --rounds N --eps E\n", argv[0]);
        return argc > 1 ? -1 : 0;
    }

    const uint64_t total = static_cast<uint64_t>(opt.rows) * opt.hidden;
    if (total == 0 || total > UINT32_MAX) {
        std::fprintf(stderr, "[ERROR] total elements must be in (0, UINT32_MAX]\n");
        return -1;
    }

    const uint32_t totalSize = static_cast<uint32_t>(total);
    const uint32_t blockDim = std::max(1u, std::min(opt.blockDim, opt.rows));
    const uint32_t rowsPerCore = (opt.rows + blockDim - 1u) / blockDim;
    const size_t inputBytes = static_cast<size_t>(totalSize) * sizeof(float);
    const size_t weightBytes = static_cast<size_t>(opt.hidden) * sizeof(float);

    if ((opt.hidden % 8u) != 0u) {
        std::fprintf(stderr, "[ERROR] optimized path requires hidden to be a multiple of 8 floats for aligned GM<->UB copy\n");
        return -1;
    }

    RmsNormOptimizedTiling tiling;
    tiling.rows = opt.rows;
    tiling.hidden = opt.hidden;
    tiling.coreNum = blockDim;
    tiling.rowsPerCore = rowsPerCore;
    tiling.eps = opt.eps;
    tiling.invHidden = 1.0f / static_cast<float>(opt.hidden);

    std::printf("[INFO] RMSNorm optimized rows=%u hidden=%u total=%u blockDim=%u rowsPerCore=%u "
                "eps=%.8g invHidden=%.8g\n",
        opt.rows, opt.hidden, totalSize, blockDim, rowsPerCore, opt.eps, tiling.invHidden);

    std::vector<float> input(totalSize), weight(opt.hidden), golden(totalSize), output(totalSize);
    std::mt19937 rng(42);
    std::normal_distribution<float> inputDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> weightDist(0.8f, 1.2f);
    for (uint32_t i = 0; i < totalSize; ++i) {
        input[i] = inputDist(rng);
    }
    for (uint32_t i = 0; i < opt.hidden; ++i) {
        weight[i] = weightDist(rng);
    }
    RmsNormRef(input, weight, &golden, opt.rows, opt.hidden, opt.eps);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *inputD = nullptr, *weightD = nullptr, *outD = nullptr, *workspaceD = nullptr, *tilingD = nullptr;
    CHECK_ACL(aclrtMalloc(&inputD, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&weightD, weightBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&outD, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&workspaceD, 32, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&tilingD, sizeof(RmsNormOptimizedTiling), ACL_MEM_MALLOC_NORMAL_ONLY));

    CHECK_ACL(aclrtMemcpy(inputD, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(weightD, weightBytes, weight.data(), weightBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingD, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));

    for (uint32_t i = 0; i < opt.warmup; ++i) {
        ACLRT_LAUNCH_KERNEL(rmsnorm_optimized_kernel)(blockDim, stream, inputD, weightD, outD, workspaceD, tilingD);
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
            ACLRT_LAUNCH_KERNEL(rmsnorm_optimized_kernel)(blockDim, stream, inputD, weightD, outD, workspaceD, tilingD);
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

    CHECK_ACL(aclrtMemcpy(output.data(), inputBytes, outD, inputBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    double maxAbs = 0.0;
    double meanAbs = 0.0;
    double maxOutputAbs = 0.0;
    for (uint32_t i = 0; i < totalSize; ++i) {
        const double diff = std::abs(static_cast<double>(output[i]) - golden[i]);
        maxAbs = std::max(maxAbs, diff);
        meanAbs += diff;
        maxOutputAbs = std::max(maxOutputAbs, std::abs(static_cast<double>(output[i])));
    }
    meanAbs /= totalSize;

    auto mm = std::minmax_element(deviceUs.begin(), deviceUs.end());
    std::printf("[RESULT] mean=%.3f median=%.3f min=%.3f max=%.3f us\n",
        Mean(deviceUs), Median(deviceUs), *mm.first, *mm.second);
    std::printf("[CHECK] max_abs_diff=%.8e mean_abs_diff=%.8e max_output_abs=%.8e\n",
        maxAbs, meanAbs, maxOutputAbs);
    const uint32_t sampleCount = std::min<uint32_t>(8u, totalSize);
    for (uint32_t i = 0; i < sampleCount; ++i) {
        std::printf("[SAMPLE] i=%u input=% .8e weight=% .8e output=% .8e golden=% .8e\n",
            i, input[i], weight[i % opt.hidden], output[i], golden[i]);
    }
    std::printf("[%s] correctness threshold atol=2e-3\n", maxAbs <= 2e-3 ? "PASS" : "FAIL");

    CHECK_ACL(aclrtFree(inputD));
    CHECK_ACL(aclrtFree(weightD));
    CHECK_ACL(aclrtFree(outD));
    CHECK_ACL(aclrtFree(workspaceD));
    CHECK_ACL(aclrtFree(tilingD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return maxAbs <= 2e-3 ? 0 : 1;
}

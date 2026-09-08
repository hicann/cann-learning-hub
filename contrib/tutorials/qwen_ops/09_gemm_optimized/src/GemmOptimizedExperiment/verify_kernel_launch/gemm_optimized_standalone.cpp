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
#include "aclrtlaunch_gemm_optimized_kernel.h"
#include "gemm_tiling.h"

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
    uint32_t m = 128;
    uint32_t k = 1024;
    uint32_t n = 512;
    uint32_t blockDim = 16;
    uint32_t warmup = 5;
    uint32_t repeat = 20;
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
        if (key == "--m") ok = ParseUint(value, &opt->m);
        else if (key == "--k") ok = ParseUint(value, &opt->k);
        else if (key == "--n") ok = ParseUint(value, &opt->n);
        else if (key == "--block-dim") ok = ParseUint(value, &opt->blockDim);
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

static void GemmRef(const std::vector<float> &a, const std::vector<float> &b,
    std::vector<float> *c, uint32_t m, uint32_t k, uint32_t n)
{
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < k; ++kk) {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            (*c)[row * n + col] = acc;
        }
    }
}

static void TransposeB(const std::vector<float> &b, std::vector<float> *bTrans, uint32_t k, uint32_t n)
{
    for (uint32_t kk = 0; kk < k; ++kk) {
        for (uint32_t col = 0; col < n; ++col) {
            (*bTrans)[col * k + kk] = b[kk * n + col];
        }
    }
}

static GemmOptimizedTiling BuildTiling(const Options &opt)
{
    GemmOptimizedTiling t;
    t.m = opt.m;
    t.n = opt.n;
    t.k = opt.k;
    t.coreNum = std::max(1u, std::min(opt.blockDim, 32u));
    t.tileM = 8;
    t.tileN = 8;
    t.tileK = 128;
    const uint32_t mTiles = (t.m + t.tileM - 1) / t.tileM;
    const uint32_t nTiles = (t.n + t.tileN - 1) / t.tileN;
    t.totalTiles = mTiles * nTiles;
    return t;
}

int main(int argc, char **argv)
{
    Options opt;
    if (!ParseOptions(argc, argv, &opt)) {
        std::printf("Usage: %s --m M --k K --n N --block-dim N --warmup N --repeat N --rounds N\n", argv[0]);
        return argc > 1 ? -1 : 0;
    }

    const uint64_t aElems = static_cast<uint64_t>(opt.m) * opt.k;
    const uint64_t bElems = static_cast<uint64_t>(opt.k) * opt.n;
    const uint64_t cElems = static_cast<uint64_t>(opt.m) * opt.n;
    if (aElems > UINT32_MAX || bElems > UINT32_MAX || cElems > UINT32_MAX) {
        std::fprintf(stderr, "[ERROR] matrix is too large for this optimized demo\n");
        return -1;
    }

    GemmOptimizedTiling tiling = BuildTiling(opt);
    const uint32_t blockDim = std::min(tiling.coreNum, tiling.totalTiles);
    const size_t aBytes = static_cast<size_t>(aElems) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(bElems) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(cElems) * sizeof(float);

    std::printf("[INFO] GEMM optimized M=%u K=%u N=%u blockDim=%u tileM=%u tileN=%u tileK=%u totalTiles=%u\n",
        opt.m, opt.k, opt.n, blockDim, tiling.tileM, tiling.tileN, tiling.tileK, tiling.totalTiles);

    std::vector<float> a(aElems), b(bElems), bTrans(bElems), golden(cElems), output(cElems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (float &v : a) v = dist(rng);
    for (float &v : b) v = dist(rng);
    GemmRef(a, b, &golden, opt.m, opt.k, opt.n);
    TransposeB(b, &bTrans, opt.k, opt.n);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *aD = nullptr, *bD = nullptr, *cD = nullptr, *workspaceD = nullptr, *tilingD = nullptr;
    CHECK_ACL(aclrtMalloc(&aD, aBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&bD, bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&cD, cBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&workspaceD, 32, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&tilingD, sizeof(GemmOptimizedTiling), ACL_MEM_MALLOC_NORMAL_ONLY));

    CHECK_ACL(aclrtMemcpy(aD, aBytes, a.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bD, bBytes, bTrans.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingD, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));

    for (uint32_t i = 0; i < opt.warmup; ++i) {
        ACLRT_LAUNCH_KERNEL(gemm_optimized_kernel)(blockDim, stream, aD, bD, cD, workspaceD, tilingD);
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
            ACLRT_LAUNCH_KERNEL(gemm_optimized_kernel)(blockDim, stream, aD, bD, cD, workspaceD, tilingD);
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

    CHECK_ACL(aclrtMemcpy(output.data(), cBytes, cD, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    double maxAbs = 0.0;
    double meanAbs = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        const double diff = std::abs(static_cast<double>(output[i]) - golden[i]);
        maxAbs = std::max(maxAbs, diff);
        meanAbs += diff;
    }
    meanAbs /= output.size();

    auto mm = std::minmax_element(deviceUs.begin(), deviceUs.end());
    std::printf("[RESULT] mean=%.3f median=%.3f min=%.3f max=%.3f us\n",
        Mean(deviceUs), Median(deviceUs), *mm.first, *mm.second);
    std::printf("[CHECK] max_abs_diff=%.8e mean_abs_diff=%.8e\n", maxAbs, meanAbs);
    std::printf("[%s] correctness threshold atol=1e-3\n", maxAbs <= 1e-3 ? "PASS" : "FAIL");

    CHECK_ACL(aclrtFree(aD));
    CHECK_ACL(aclrtFree(bD));
    CHECK_ACL(aclrtFree(cD));
    CHECK_ACL(aclrtFree(workspaceD));
    CHECK_ACL(aclrtFree(tilingD));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return maxAbs <= 1e-3 ? 0 : 1;
}
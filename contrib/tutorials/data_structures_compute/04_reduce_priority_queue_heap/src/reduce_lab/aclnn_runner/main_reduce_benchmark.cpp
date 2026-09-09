#include "common/acl_utils.h"
#include "common/bin_utils.h"

#include <aclnn_reduce_sum_lite.h>
#include <aclnn_reduce_max_lite.h>
#include <aclnn_top_k_reduce_lite.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr size_t FP16_BYTES = 2;
#ifndef REDUCE_LAB_BLOCK_DIM
#define REDUCE_LAB_BLOCK_DIM 20
#endif
static constexpr uint32_t BLOCK_DIM = REDUCE_LAB_BLOCK_DIM;
static constexpr size_t CACHE_LINE_BYTES = 64;
static constexpr size_t PARTIAL_STRIDE = CACHE_LINE_BYTES / FP16_BYTES;
static constexpr size_t TOPK_VALUE_STRIDE = CACHE_LINE_BYTES / FP16_BYTES;
static constexpr size_t TOPK_INDEX_STRIDE = CACHE_LINE_BYTES / sizeof(int32_t);
static constexpr int64_t LITE_FIXED_TOP_K = 4;
static constexpr int WARMUP_ITERS = 3;
static constexpr int TIMED_ITERS = 10;

static inline uint16_t float_to_fp16(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint16_t sign = (bits >> 16) & 0x8000;
    int16_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint16_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | mant;
}
static inline float fp16_to_float(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    return (sign ? -1 : 1) * (exp ? powf(2, exp - 15) * (1 + mant / 1024.0f) : powf(2, -14) * (mant / 1024.0f));
}

template <typename Fn>
static double RunOpTimed(aclrtStream stream, int warmup, int iters, Fn &&fn) {
    aclrtEvent startEvt, endEvt;
    CHECK_ACL(aclrtCreateEvent(&startEvt));
    CHECK_ACL(aclrtCreateEvent(&endEvt));
    for (int i = 0; i < warmup; ++i) fn();
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtRecordEvent(startEvt, stream));
    for (int i = 0; i < iters; ++i) fn();
    CHECK_ACL(aclrtRecordEvent(endEvt, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    float elapsedMs = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, startEvt, endEvt));
    aclrtDestroyEvent(startEvt);
    aclrtDestroyEvent(endEvt);
    return static_cast<double>(elapsedMs) / iters;
}

static std::pair<double, float> RunReduceSum(void *xDev, void *yDev, int64_t N, aclrtStream stream) {
    aclTensor *xT = CreateTensor({N}, ACL_FLOAT16, ACL_FORMAT_ND, xDev);
    aclTensor *yT = CreateTensor({(int64_t)(BLOCK_DIM * PARTIAL_STRIDE)}, ACL_FLOAT16, ACL_FORMAT_ND, yDev);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnReduceSumLiteGetWorkspaceSize(xT, yT, &ws, &ex));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        CHECK_ACL(aclnnReduceSumLite(workspace, ws, ex, stream));
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // Each core's partial sum is stored at core * PARTIAL_STRIDE.
    float result = 0.0f;
    std::vector<uint16_t> partial(BLOCK_DIM * PARTIAL_STRIDE);
    CHECK_ACL(aclrtMemcpy(partial.data(), partial.size() * sizeof(uint16_t),
                          yDev, partial.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t core = 0; core < BLOCK_DIM; ++core) {
        result += fp16_to_float(partial[core * PARTIAL_STRIDE]);
    }
    CHECK_ACL(aclDestroyAclOpExecutor(ex));
    aclDestroyTensor(xT); aclDestroyTensor(yT);
    if (workspace) aclrtFree(workspace);
    return {ms, result};
}

static std::pair<double, float> RunReduceMax(void *xDev, void *yDev, int64_t N, aclrtStream stream) {
    aclTensor *xT = CreateTensor({N}, ACL_FLOAT16, ACL_FORMAT_ND, xDev);
    aclTensor *yT = CreateTensor({(int64_t)(BLOCK_DIM * PARTIAL_STRIDE)}, ACL_FLOAT16, ACL_FORMAT_ND, yDev);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnReduceMaxLiteGetWorkspaceSize(xT, yT, &ws, &ex));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        CHECK_ACL(aclnnReduceMaxLite(workspace, ws, ex, stream));
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // Each core's partial maximum is stored at core * PARTIAL_STRIDE.
    float result = -1e30f;
    std::vector<uint16_t> partial(BLOCK_DIM * PARTIAL_STRIDE);
    CHECK_ACL(aclrtMemcpy(partial.data(), partial.size() * sizeof(uint16_t),
                          yDev, partial.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t core = 0; core < BLOCK_DIM; ++core) {
        float p = fp16_to_float(partial[core * PARTIAL_STRIDE]);
        if (p > result) result = p;
    }
    CHECK_ACL(aclDestroyAclOpExecutor(ex));
    aclDestroyTensor(xT); aclDestroyTensor(yT);
    if (workspace) aclrtFree(workspace);
    return {ms, result};
}

struct TopKResult {
    double ms;
    std::vector<float> values;
    std::vector<int32_t> indices;
};

static TopKResult RunTopK(void *xDev, void *valDev, void *idxDev,
                          int64_t N, int64_t K, aclrtStream stream) {
    const int64_t valueElements = BLOCK_DIM * TOPK_VALUE_STRIDE;
    const int64_t indexElements = BLOCK_DIM * TOPK_INDEX_STRIDE;
    aclTensor *xT = CreateTensor({N}, ACL_FLOAT16, ACL_FORMAT_ND, xDev);
    aclTensor *valT = CreateTensor({valueElements}, ACL_FLOAT16, ACL_FORMAT_ND, valDev);
    aclTensor *idxT = CreateTensor({indexElements}, ACL_INT32, ACL_FORMAT_ND, idxDev);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnTopKReduceLiteGetWorkspaceSize(xT, valT, idxT, &ws, &ex));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        CHECK_ACL(aclnnTopKReduceLite(workspace, ws, ex, stream));
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Each core owns one cache-line-sized values slot and one index slot.
    uint32_t numCandidates = BLOCK_DIM * K;
    TopKResult res;
    res.ms = ms;
    std::vector<uint16_t> candVal(valueElements);
    std::vector<int32_t> candIdx(indexElements);
    CHECK_ACL(aclrtMemcpy(candVal.data(), candVal.size() * sizeof(uint16_t),
                          valDev, candVal.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(candIdx.data(), candIdx.size() * sizeof(int32_t),
                          idxDev, candIdx.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

    // Host-side merge: find global TopK from all cores' candidates
    std::vector<std::pair<float, int32_t>> all;
    all.reserve(numCandidates);
    for (uint32_t core = 0; core < BLOCK_DIM; ++core) {
        for (uint32_t k = 0; k < static_cast<uint32_t>(K); ++k) {
            float v = fp16_to_float(candVal[core * TOPK_VALUE_STRIDE + k]);
            int32_t index = candIdx[core * TOPK_INDEX_STRIDE + k];
            if (index >= 0) {
                all.push_back({v, index});
            }
        }
    }
    std::sort(all.begin(), all.end(), [](auto &a, auto &b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    uint32_t outK = std::min((uint32_t)K, (uint32_t)all.size());
    res.values.resize(outK);
    res.indices.resize(outK);
    for (uint32_t i = 0; i < outK; ++i) {
        res.values[i] = all[i].first;
        res.indices[i] = all[i].second;
    }
    CHECK_ACL(aclDestroyAclOpExecutor(ex));
    aclDestroyTensor(xT); aclDestroyTensor(valT); aclDestroyTensor(idxT);
    if (workspace) aclrtFree(workspace);
    return res;
}

int main(int argc, char **argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: main_reduce_benchmark <data_dir> <num_tokens> [top_k]\n";
            return 1;
        }
        std::string dataDir = argv[1];
        int64_t N = std::stoll(argv[2]);
        int64_t K = 4; // default
        if (argc >= 4) K = std::stoll(argv[3]);
        if (K != LITE_FIXED_TOP_K) {
            throw std::runtime_error("TopKReduceLite supports top_k=4 only. Regenerate data with --top_k 4.");
        }

        auto xHost = ReadBinary(dataDir + "/input/x.bin");
        AclRuntimeGuard guard(0);
        aclrtStream stream = guard.stream();

        void *xDev = MallocDevice(xHost.size());
        void *yDev = MallocDevice(BLOCK_DIM * PARTIAL_STRIDE * FP16_BYTES);
        CHECK_ACL(aclrtMemcpy(xDev, xHost.size(), xHost.data(), xHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));

        std::cout << "=== Reduce Operator Lab Benchmark ===\n";
        std::cout << "N=" << N << " K=" << K << " BLOCK_DIM=" << BLOCK_DIM << "\n";
        std::cout << "Timing: aclrtEvent, " << WARMUP_ITERS << " warmup + " << TIMED_ITERS << " timed iterations\n\n";

        // ReduceSum
        auto [sumMs, sumResult] = RunReduceSum(xDev, yDev, N, stream);
        auto refSumBin = ReadBinary(dataDir + "/input/ref_sum.bin");
        uint16_t refSumU16; memcpy(&refSumU16, refSumBin.data(), FP16_BYTES);
        float refSum = fp16_to_float(refSumU16);
        float sumErr = std::fabs(sumResult - refSum);
        std::cout << "[ReduceSum]  time=" << std::fixed << std::setprecision(4) << sumMs
                  << " ms  result=" << sumResult << "  ref=" << refSum
                  << "  error=" << sumErr << (sumErr < 1.0f ? "  PASS" : "  FAIL") << "\n";

        // ReduceMax
        auto [maxMs, maxResult] = RunReduceMax(xDev, yDev, N, stream);
        auto refMaxBin = ReadBinary(dataDir + "/input/ref_max.bin");
        uint16_t refMaxU16; memcpy(&refMaxU16, refMaxBin.data(), FP16_BYTES);
        float refMax = fp16_to_float(refMaxU16);
        float maxErr = std::fabs(maxResult - refMax);
        std::cout << "[ReduceMax]  time=" << std::fixed << std::setprecision(4) << maxMs
                  << " ms  result=" << maxResult << "  ref=" << refMax
                  << "  error=" << maxErr << (maxErr < 0.01f ? "  PASS" : "  FAIL") << "\n";

        // TopK
        void *valDev = MallocDevice(BLOCK_DIM * TOPK_VALUE_STRIDE * FP16_BYTES);
        void *idxDev = MallocDevice(BLOCK_DIM * TOPK_INDEX_STRIDE * sizeof(int32_t));
        auto topkRes = RunTopK(xDev, valDev, idxDev, N, K, stream);

        // Read reference TopK
        auto refTopkValBin = ReadBinary(dataDir + "/input/ref_topk_val.bin");
        auto refTopkIdxBin = ReadBinary(dataDir + "/input/ref_topk_idx.bin");

        std::cout << "[TopK]       time=" << std::fixed << std::setprecision(4) << topkRes.ms << " ms\n";

        // Compare TopK values
        bool topkPass = true;
        uint32_t refK = refTopkValBin.size() / FP16_BYTES;
        if (topkRes.values.size() >= refK) {
            std::cout << "  result values: ";
            for (size_t i = 0; i < topkRes.values.size(); ++i) {
                std::cout << std::setprecision(4) << topkRes.values[i] << " ";
            }
            std::cout << "\n  result indices:";
            for (size_t i = 0; i < topkRes.indices.size(); ++i) {
                std::cout << " " << topkRes.indices[i];
            }
            std::cout << "\n  ref values:    ";
            for (uint32_t i = 0; i < refK; ++i) {
                uint16_t rv; memcpy(&rv, refTopkValBin.data() + i * FP16_BYTES, FP16_BYTES);
                std::cout << std::setprecision(4) << fp16_to_float(rv) << " ";
            }
            std::cout << "\n  ref indices:   ";
            for (uint32_t i = 0; i < refK; ++i) {
                int32_t ri; memcpy(&ri, refTopkIdxBin.data() + i * sizeof(int32_t), sizeof(int32_t));
                std::cout << ri << " ";
            }
            std::cout << "\n";

            // Check match (values close enough, indices match)
            for (uint32_t i = 0; i < refK; ++i) {
                uint16_t rv; memcpy(&rv, refTopkValBin.data() + i * FP16_BYTES, FP16_BYTES);
                float refV = fp16_to_float(rv);
                int32_t refI; memcpy(&refI, refTopkIdxBin.data() + i * sizeof(int32_t), sizeof(int32_t));
                float vErr = std::fabs(topkRes.values[i] - refV);
                if (vErr > 0.1f || topkRes.indices[i] != refI) {
                    topkPass = false;
                    break;
                }
            }
        } else {
            topkPass = false;
        }
        std::cout << "  " << (topkPass ? "PASS" : "FAIL") << "\n";

        std::cout << "\n=== Done ===\n";
        aclrtFree(xDev); aclrtFree(yDev);
        aclrtFree(valDev); aclrtFree(idxDev);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 99;
    }
}

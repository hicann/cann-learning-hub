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
#include <string>
#include <vector>

static constexpr size_t FP16_BYTES = 2;
static constexpr uint32_t BLOCK_DIM = 20;
static constexpr size_t HALF_BYTES = 2;
static constexpr size_t CACHE_LINE_BYTES = 128;
static constexpr size_t CACHE_LINE_HALF = 64;  // 128/2
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

static double RunOpTimed(aclrtStream stream, int warmup, int iters, auto &&fn) {
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
    aclTensor *yT = CreateTensor({(int64_t)BLOCK_DIM}, ACL_FLOAT16, ACL_FORMAT_ND, yDev);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnReduceSumLiteGetWorkspaceSize(xT, yT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        aclnnReduceSumLite(workspace, ws, ex, stream);
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // Read partial sums from output tensor y[0..7]
    float result = 0.0f;
    std::vector<uint16_t> partial(BLOCK_DIM);
    CHECK_ACL(aclrtMemcpy(partial.data(), BLOCK_DIM * sizeof(uint16_t),
                          yDev, BLOCK_DIM * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint16_t h : partial) result += fp16_to_float(h);
    aclDestroyTensor(xT); aclDestroyTensor(yT);
    if (workspace) aclrtFree(workspace);
    return {ms, result};
}

static std::pair<double, float> RunReduceMax(void *xDev, void *yDev, int64_t N, aclrtStream stream) {
    aclTensor *xT = CreateTensor({N}, ACL_FLOAT16, ACL_FORMAT_ND, xDev);
    aclTensor *yT = CreateTensor({(int64_t)BLOCK_DIM}, ACL_FLOAT16, ACL_FORMAT_ND, yDev);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnReduceMaxLiteGetWorkspaceSize(xT, yT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        aclnnReduceMaxLite(workspace, ws, ex, stream);
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // Read partial maxes from output tensor y[0..7]
    float result = -1e30f;
    std::vector<uint16_t> partial(BLOCK_DIM);
    CHECK_ACL(aclrtMemcpy(partial.data(), BLOCK_DIM * sizeof(uint16_t),
                          yDev, BLOCK_DIM * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint16_t h : partial) { float p = fp16_to_float(h); if (p > result) result = p; }
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
    int64_t totalK = BLOCK_DIM * K;  // 8 cores * K
    aclTensor *xT = CreateTensor({N}, ACL_FLOAT16, ACL_FORMAT_ND, xDev);
    aclTensor *valT = CreateTensor({totalK}, ACL_FLOAT16, ACL_FORMAT_ND, valDev);
    aclTensor *idxT = CreateTensor({totalK}, ACL_INT32, ACL_FORMAT_ND, idxDev);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnTopKReduceLiteGetWorkspaceSize(xT, valT, idxT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        aclnnTopKReduceLite(workspace, ws, ex, stream);
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Read partial TopK from output tensors values[0..totalK-1] and indices[0..totalK-1]
    uint32_t numCandidates = totalK;
    TopKResult res;
    res.ms = ms;
    std::vector<uint16_t> candVal(numCandidates);
    std::vector<int32_t> candIdx(numCandidates);
    CHECK_ACL(aclrtMemcpy(candVal.data(), numCandidates * sizeof(uint16_t),
                          valDev, numCandidates * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(candIdx.data(), numCandidates * sizeof(int32_t),
                          idxDev, numCandidates * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

    // Host-side merge: find global TopK from all cores' candidates
    std::vector<std::pair<float, int32_t>> all;
    for (uint32_t i = 0; i < numCandidates; ++i) {
        float v = fp16_to_float(candVal[i]);
        if (candIdx[i] >= 0) {
            all.push_back({v, candIdx[i]});
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

        auto xHost = ReadBinary(dataDir + "/input/x.bin");
        AclRuntimeGuard guard(0);
        aclrtStream stream = guard.stream();

        void *xDev = MallocDevice(xHost.size());
        void *yDev = MallocDevice(BLOCK_DIM * FP16_BYTES);  // 8 partial results
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
        void *valDev = MallocDevice(BLOCK_DIM * K * FP16_BYTES);   // 8 cores * K half values
        void *idxDev = MallocDevice(BLOCK_DIM * K * sizeof(int32_t)); // 8 cores * K int32 indices
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
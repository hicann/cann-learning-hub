#include "common/acl_utils.h"
#include "common/bin_utils.h"

#include <aclnn_attention_custom.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// 运行参数：main_attention_benchmark <data_dir> <seq_len> [dim]
// 数据文件：<data_dir>/<seq_len>/q.bin k.bin v.bin ref.bin（均为 float16 原始数据）
static constexpr int WARMUP_ITERS = 3;
static constexpr int TIMED_ITERS = 10;

static void PrintUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " <data_dir> <seq_len> [dim] [warmup] [iters]\n";
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

static inline float HalfToFloat(uint16_t h) {
    // 二进制转 float（不依赖 fp16 硬件指令）
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x3FFu;
    uint32_t fbits;
    if (exp == 0) {
        if (frac == 0) { fbits = sign << 31; }
        else {
            exp = 127 - 15 + 1;
            while ((frac & 0x400u) == 0) { frac <<= 1; exp--; }
            frac &= 0x3FFu;
            fbits = (sign << 31) | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        fbits = (sign << 31) | 0x7F800000u | (frac << 13);
    } else {
        fbits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &fbits, 4);
    return f;
}

static inline uint16_t FloatToHalf(float f) {
    uint32_t fbits;
    memcpy(&fbits, &f, 4);
    uint32_t sign = (fbits >> 31) & 1u;
    int32_t exp = (fbits >> 23) & 0xFF;
    uint32_t frac = fbits & 0x7FFFFFu;
    uint16_t h;
    if (exp == 0xFF) {  // inf / nan
        h = (uint16_t)((sign << 15) | 0x7C00u | ((frac >> 13) & 0x3FFu));
    } else if (exp == 0) {  // 零/次正规 float -> 零
        h = (uint16_t)(sign << 15);
    } else {
        int32_t nexp = exp - 127 + 15;
        if (nexp >= 31) { h = (uint16_t)((sign << 15) | 0x7C00u); }
        else if (nexp <= 0) {
            if (nexp < -10) { h = (uint16_t)(sign << 15); }
            else {
                uint32_t nfrac = frac | 0x800000u;
                uint32_t shift = (uint32_t)(14 - nexp);
                uint32_t rounded = (nfrac >> shift) + (((nfrac >> (shift - 1)) & 1u) != 0u ? 1u : 0u);
                h = (uint16_t)((sign << 15) | rounded);
            }
        } else {
            uint32_t nfrac = (frac >> 13) + (((frac >> 12) & 1u) != 0u ? 1u : 0u);
            if (nfrac & 0x400u) { nfrac = 0; nexp++; }
            if (nexp >= 31) { h = (uint16_t)((sign << 15) | 0x7C00u); }
            else { h = (uint16_t)((sign << 15) | ((uint32_t)nexp << 10) | nfrac); }
        }
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc < 3) { PrintUsage(argv[0]); return 1; }
    std::string dataDir = argv[1];
    int64_t seqLen = std::stoll(argv[2]);
    int64_t dim = (argc > 3) ? std::stoll(argv[3]) : 64;
    int warmup = (argc > 4) ? std::stoi(argv[4]) : WARMUP_ITERS;
    int iters = (argc > 5) ? std::stoi(argv[5]) : TIMED_ITERS;

    std::string dir = dataDir + "/" + std::to_string(seqLen);
    auto qBin = ReadBinary(dir + "/q.bin");
    auto kBin = ReadBinary(dir + "/kt.bin");
    auto vBin = ReadBinary(dir + "/v.bin");
    auto refBin = ReadBinary(dir + "/ref.bin");

    int64_t qkElements = seqLen * dim;
    int64_t oElements = seqLen * dim;
    if (qBin.size() != static_cast<size_t>(qkElements * 2) ||
        refBin.size() != static_cast<size_t>(oElements * 2)) {
        std::cerr << "数据文件大小与 shape 不匹配\n";
        return 1;
    }

    AclRuntimeGuard guard(0);
    aclrtStream stream = guard.stream();

    void *qDev = MallocDevice(qBin.size());
    void *kDev = MallocDevice(kBin.size());
    void *vDev = MallocDevice(vBin.size());
    void *oDev = MallocDevice(oElements * 2);
    CHECK_ACL(aclrtMemcpy(qDev, qBin.size(), qBin.data(), qBin.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(kDev, kBin.size(), kBin.data(), kBin.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(vDev, vBin.size(), vBin.data(), vBin.size(), ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> qkShape = {seqLen, dim};
    aclTensor *qT = CreateTensor(qkShape, ACL_FLOAT16, ACL_FORMAT_ND, qDev);
    aclTensor *kT = CreateTensor(qkShape, ACL_FLOAT16, ACL_FORMAT_ND, kDev);
    aclTensor *vT = CreateTensor(qkShape, ACL_FLOAT16, ACL_FORMAT_ND, vDev);
    aclTensor *oT = CreateTensor(qkShape, ACL_FLOAT16, ACL_FORMAT_ND, oDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnAttentionCustomGetWorkspaceSize(qT, kT, vT, oT, &workspaceSize, &executor));
    void *workspace = workspaceSize > 0 ? MallocDevice(workspaceSize) : nullptr;

    double ms = RunOpTimed(stream, warmup, iters, [&]() {
        CHECK_ACL(aclnnAttentionCustom(workspace, workspaceSize, executor, stream));
    });

    std::vector<uint16_t> output(oElements);
    CHECK_ACL(aclrtMemcpy(output.data(), oElements * 2, oDev, oElements * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    // 与 torch 参考（ref.bin）比对
    const uint16_t *ref = reinterpret_cast<const uint16_t *>(refBin.data());
    double maxAbs = 0.0, maxRel = 0.0;
    bool pass = true;
    for (int64_t i = 0; i < oElements; ++i) {
        double a = HalfToFloat(output[i]);
        double b = HalfToFloat(ref[i]);
        double diff = std::fabs(a - b);
        double rel = diff / std::max(1e-7, std::fabs(b));
        maxAbs = std::max(maxAbs, diff);
        maxRel = std::max(maxRel, rel);
        if (diff > 1e-2 && rel > 1e-2) pass = false;
    }

    double gflops = 2.0 * static_cast<double>(seqLen) * seqLen * dim / (ms * 1e-3) / 1e9;
    std::cout << "[Attention] seq_len=" << seqLen << " dim=" << dim
              << " time=" << std::fixed << std::setprecision(4) << ms << " ms"
              << " (" << std::setprecision(1) << gflops << " GFLOPS)\n";
    std::cout << "  maxAbsErr=" << std::setprecision(6) << maxAbs
              << " maxRelErr=" << maxRel << "\n";
    std::cout << "  result: " << (pass ? "PASS" : "FAIL") << "\n";

    // 采样输出前/中/后各 8 个元素
    auto printSample = [&](const char* tag, int64_t start) {
        std::cout << "  " << tag << ": ";
        for (int64_t i = start; i < std::min<int64_t>(start + 8, oElements); ++i) {
            std::cout << std::setprecision(4) << HalfToFloat(output[i]) << " ";
        }
        std::cout << "\n";
    };
    printSample("o[0..7]     ", 0);
    printSample("o[64..71]   ", 64);
    printSample("o[128..135] ", 128);
    printSample("o[192..199] ", 192);
    printSample("o[256..263] ", 256);
    printSample("o[320..327] ", 320);
    printSample("o[384..391] ", 384);
    printSample("o[448..455] ", 448);
    printSample("o[last..last+7]", oElements - 8);

    aclDestroyTensor(qT); aclDestroyTensor(kT); aclDestroyTensor(vT); aclDestroyTensor(oT);
    aclrtFree(qDev); aclrtFree(kDev); aclrtFree(vDev); aclrtFree(oDev);
    if (workspace) aclrtFree(workspace);

    return pass ? 0 : 1;
}

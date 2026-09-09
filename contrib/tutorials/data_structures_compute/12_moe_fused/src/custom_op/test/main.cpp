/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * MoeRouterFused 正确性测试程序（aclnn C++ 调用）
 *
 * 用法: execute_moe_router_fused <case_dir> [--device <id>]
 *   case_dir 指向 data/case_<N>_<D>_<E>_<K>/ 目录，含:
 *     meta.json            N/D/E/K 参数
 *     x_fp16.bin           x [N,D] FP16 原始字节
 *     wgate_fp16.bin       w_gate [D,E] FP16 原始字节
 *     ref_topk_idx.npy     topk_idx [N,K] int32 参考
 *     ref_topk_weights.npy topk_weights [N,K] float32 参考
 *
 * 对比口径：
 *   - topk_idx：逐元素精确匹配；允许“近似并列”差异（|w_kernel - w_ref| < 1e-4 视为并列翻转）
 *   - topk_weights：rtol=atol=1e-2（FP16 量化误差）
 */

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn/acl_meta.h>

#include "aclnn_moe_router_fused.h"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// ---------------- 文件 / 解析工具 ----------------

std::vector<uint8_t> ReadFile(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[FATAL] cannot open file: " << path << std::endl;
        exit(2);
    }
    f.seekg(0, std::ios::end);
    std::streampos endPos = f.tellg();
    if (endPos == std::streampos(-1)) {
        std::cerr << "[FATAL] cannot determine size of: " << path << std::endl;
        exit(2);
    }
    size_t n = static_cast<size_t>(endPos);
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(n);
    if (n > 0) {
        f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(n));
        if (!f) {
            std::cerr << "[FATAL] incomplete read of: " << path << std::endl;
            exit(2);
        }
    }
    return buf;
}

int64_t JsonInt(const std::string &text, const std::string &key)
{
    std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return -1;
    }
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return -1;
    }
    pos = text.find_first_of("0123456789-", pos + 1);
    if (pos == std::string::npos) {
        return -1;
    }
    return std::atoll(text.c_str() + pos);
}

// 极简 .npy v1.0/v2.0 解析（仅支持 descr '<f4' / '<i4'，fortran_order=False）
struct NpyArray {
    char elemType = 'f';          // 'f' float32, 'i' int32
    std::vector<int64_t> shape;
    const uint8_t *data = nullptr; // 指向调用方 buffer 内的数据区
    size_t dataLen = 0;
};

bool ParseNpy(const std::vector<uint8_t> &buf, NpyArray &out)
{
    if (buf.size() < 12 || std::memcmp(buf.data(), "\x93NUMPY", 6) != 0) {
        return false;
    }
    uint8_t major = buf[6];
    size_t headerLen = 0;
    size_t headerStart = 0;
    if (major == 1) {
        headerLen = static_cast<size_t>(buf[8]) | (static_cast<size_t>(buf[9]) << 8);
        headerStart = 10;
    } else if (major == 2) {
        headerLen = static_cast<size_t>(buf[8]) | (static_cast<size_t>(buf[9]) << 8) |
                    (static_cast<size_t>(buf[10]) << 16) | (static_cast<size_t>(buf[11]) << 24);
        headerStart = 12;
    } else {
        return false;
    }
    if (headerStart + headerLen > buf.size()) {
        return false;
    }
    std::string header(reinterpret_cast<const char *>(buf.data() + headerStart), headerLen);

    // descr
    size_t dpos = header.find("'descr'");
    if (dpos != std::string::npos) {
        dpos = header.find("'<", dpos);
        if (dpos != std::string::npos) {
            out.elemType = (header[dpos + 2] == 'i') ? 'i' : 'f';
        }
    }
    // shape
    size_t spos = header.find("'shape'");
    if (spos != std::string::npos) {
        size_t lp = header.find('(', spos);
        size_t rp = header.find(')', lp);
        if (lp != std::string::npos && rp != std::string::npos) {
            std::string shapeStr = header.substr(lp + 1, rp - lp - 1);
            size_t i = 0;
            while (i < shapeStr.size()) {
                while (i < shapeStr.size() && (shapeStr[i] == ' ' || shapeStr[i] == ',')) {
                    ++i;
                }
                size_t j = i;
                while (j < shapeStr.size() && shapeStr[j] != ' ' && shapeStr[j] != ',') {
                    ++j;
                }
                if (j > i) {
                    out.shape.push_back(std::atoll(shapeStr.substr(i, j - i).c_str()));
                }
                i = j;
            }
        }
    }
    out.data = buf.data() + headerStart + headerLen;
    out.dataLen = buf.size() - (headerStart + headerLen);
    return true;
}

// IEEE 754 half (1-5-10) -> float32
inline float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) {
            u = sign;  // ±0
        } else {
            // 非规格化数：左移规格化
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                --exp;
            }
            man &= 0x3FFu;
            u = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (man << 13);  // inf / nan
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------------- ACL 工具 ----------------

#define CHECK_ACL(call)                                                        \
    do {                                                                       \
        aclError _e = (call);                                                  \
        if (_e != ACL_SUCCESS) {                                               \
            std::cerr << "[ACL ERROR] " << #call << " -> " << _e               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(3);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_ACLNN(call)                                                      \
    do {                                                                       \
        aclnnStatus _s = (call);                                               \
        if (_s != 0) {                                                         \
            std::cerr << "[ACLNN ERROR] " << #call << " -> " << _s             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(4);                                                           \
        }                                                                      \
    } while (0)

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "usage: execute_moe_router_fused <case_dir> [--device <id>] [--dump <prefix>] "
                     "[--bench <iters>]" << std::endl;
        return 1;
    }
    std::string caseDir = argv[1];
    int deviceId = 0;
    std::string dumpPrefix;
    int benchIters = 0;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--device") {
            deviceId = std::atoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--dump") {
            dumpPrefix = argv[i + 1];
        } else if (std::string(argv[i]) == "--bench") {
            benchIters = std::atoi(argv[i + 1]);
        }
    }

    // ---- 读取 meta ----
    std::vector<uint8_t> metaBuf = ReadFile(caseDir + "/meta.json");
    std::string metaText(reinterpret_cast<char *>(metaBuf.data()), metaBuf.size());
    int64_t N = JsonInt(metaText, "N");
    int64_t D = JsonInt(metaText, "D");
    int64_t E = JsonInt(metaText, "E");          // 真实专家数（"e" 属性）
    int64_t E_pad = JsonInt(metaText, "E_pad");  // padding 后专家数（w_gate 第二维）
    int64_t K = JsonInt(metaText, "K");
    if (E_pad <= 0) {
        E_pad = E;
    }
    if (N <= 0 || D <= 0 || E <= 0 || E_pad < E || K <= 0) {
        std::cerr << "[FATAL] bad meta.json" << std::endl;
        return 2;
    }

    // ---- 读取数据 ----
    std::vector<uint8_t> xBuf = ReadFile(caseDir + "/x_fp16.bin");
    std::vector<uint8_t> wBuf = ReadFile(caseDir + "/wgate_fp16.bin");
    std::vector<uint8_t> idxRefBuf = ReadFile(caseDir + "/ref_topk_idx.npy");
    std::vector<uint8_t> wtRefBuf = ReadFile(caseDir + "/ref_topk_weights.npy");
    if (xBuf.size() != static_cast<size_t>(N * D * 2) ||
        wBuf.size() != static_cast<size_t>(D * E_pad * 2)) {
        std::cerr << "[FATAL] input bin size mismatch" << std::endl;
        return 2;
    }
    NpyArray idxRef;
    NpyArray wtRef;
    if (!ParseNpy(idxRefBuf, idxRef) || idxRef.elemType != 'i') {
        std::cerr << "[FATAL] bad ref_topk_idx.npy" << std::endl;
        return 2;
    }
    if (!ParseNpy(wtRefBuf, wtRef) || wtRef.elemType != 'f') {
        std::cerr << "[FATAL] bad ref_topk_weights.npy" << std::endl;
        return 2;
    }
    const int32_t *refIdx = reinterpret_cast<const int32_t *>(idxRef.data);
    const float *refWt = reinterpret_cast<const float *>(wtRef.data);
    size_t outElems = static_cast<size_t>(N * K);
    if (idxRef.dataLen < outElems * sizeof(int32_t) || wtRef.dataLen < outElems * sizeof(float)) {
        std::cerr << "[FATAL] reference npy size mismatch" << std::endl;
        return 2;
    }

    // ---- 初始化 ACL ----
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ---- 分配设备内存 ----
    void *xDev = nullptr;
    void *wDev = nullptr;
    void *idxDev = nullptr;
    void *wtDev = nullptr;
    CHECK_ACL(aclrtMalloc(&xDev, xBuf.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&wDev, wBuf.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&idxDev, outElems * sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&wtDev, outElems * 2, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMemcpy(xDev, xBuf.size(), xBuf.data(), xBuf.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(wDev, wBuf.size(), wBuf.data(), wBuf.size(), ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 创建 tensor ----
    int64_t shapeX[2] = {N, D};
    int64_t shapeW[2] = {D, E_pad};
    int64_t shapeOut[2] = {N, K};
    aclTensor *xT = aclCreateTensor(shapeX, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shapeX, 2, xDev);
    aclTensor *wT = aclCreateTensor(shapeW, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shapeW, 2, wDev);
    aclTensor *idxT = aclCreateTensor(shapeOut, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, shapeOut, 2, idxDev);
    aclTensor *wtT = aclCreateTensor(shapeOut, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, shapeOut, 2, wtDev);
    if (!xT || !wT || !idxT || !wtT) {
        std::cerr << "[FATAL] aclCreateTensor failed" << std::endl;
        return 4;
    }

    // ---- 执行算子 ----
    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACLNN(aclnnMoeRouterFusedGetWorkspaceSize(xT, wT, static_cast<int64_t>(K), static_cast<int64_t>(E),
        idxT, wtT, &wsSize, &executor));
    void *workspace = nullptr;
    if (wsSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    CHECK_ACLNN(aclnnMoeRouterFused(workspace, wsSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // ---- 可选：kernel 耗时测量（M5 预采集；事件计时，含 3 次预热）----
    if (benchIters > 0) {
        for (int i = 0; i < 3; ++i) {
            CHECK_ACLNN(aclnnMoeRouterFused(workspace, wsSize, executor, stream));
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        aclrtEvent evStart = nullptr;
        aclrtEvent evStop = nullptr;
        CHECK_ACL(aclrtCreateEvent(&evStart));
        CHECK_ACL(aclrtCreateEvent(&evStop));
        CHECK_ACL(aclrtRecordEvent(evStart, stream));
        for (int i = 0; i < benchIters; ++i) {
            CHECK_ACLNN(aclnnMoeRouterFused(workspace, wsSize, executor, stream));
        }
        CHECK_ACL(aclrtRecordEvent(evStop, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        float elapsedMs = 0.0f;
        CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, evStart, evStop));
        std::printf("[bench] iters=%d mean=%.4f ms\n", benchIters, elapsedMs / benchIters);
        aclrtDestroyEvent(evStart);
        aclrtDestroyEvent(evStop);
    }

    // ---- 取回结果 ----
    std::vector<int32_t> idxOut(outElems);
    std::vector<uint16_t> wtOut(outElems);
    CHECK_ACL(aclrtMemcpy(idxOut.data(), idxOut.size() * sizeof(int32_t), idxDev,
        idxOut.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(wtOut.data(), wtOut.size() * 2, wtDev, wtOut.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- 可选：导出内核输出供离线逐行分析 ----
    if (!dumpPrefix.empty()) {
        std::ofstream fIdx(dumpPrefix + "_idx.bin", std::ios::binary);
        std::ofstream fWt(dumpPrefix + "_wt.bin", std::ios::binary);
        fIdx.write(reinterpret_cast<const char *>(idxOut.data()), idxOut.size() * sizeof(int32_t));
        fWt.write(reinterpret_cast<const char *>(wtOut.data()), wtOut.size() * sizeof(uint16_t));
        std::printf("[dump] %s_idx.bin / %s_wt.bin written\n", dumpPrefix.c_str(), dumpPrefix.c_str());
    }

    // ---- 对比 ----
    int64_t idxMatch = 0;
    int64_t idxTieDiff = 0;
    int64_t idxRealDiff = 0;
    double wtMaxRel = 0.0;
    double wtMaxAbs = 0.0;
    int64_t wtFail = 0;
    for (size_t i = 0; i < outElems; ++i) {
        float wk = Fp16ToFloat(wtOut[i]);
        if (idxOut[i] == refIdx[i]) {
            ++idxMatch;
        } else {
            if (std::fabs(wk - refWt[i]) < 1e-4) {
                ++idxTieDiff;  // 近似并列导致 top-k 顺序翻转
            } else {
                ++idxRealDiff;
            }
        }
        double absErr = std::fabs(static_cast<double>(wk) - refWt[i]);
        double relErr = absErr / std::max(1e-6, std::fabs(static_cast<double>(refWt[i])));
        wtMaxAbs = std::max(wtMaxAbs, absErr);
        wtMaxRel = std::max(wtMaxRel, relErr);
        if (absErr > 1e-2 + 1e-2 * std::fabs(static_cast<double>(refWt[i]))) {
            ++wtFail;
        }
    }

    std::printf("[case] N=%lld D=%lld E=%lld K=%lld\n", (long long)N, (long long)D, (long long)E, (long long)K);
    std::printf("[idx ] match=%lld/%zu tie_diff=%lld real_diff=%lld\n",
        (long long)idxMatch, outElems, (long long)idxTieDiff, (long long)idxRealDiff);
    std::printf("[wt  ] max_abs=%.6e max_rel=%.6e fail=%lld/%zu\n",
        wtMaxAbs, wtMaxRel, (long long)wtFail, outElems);

    bool idxOk = (idxRealDiff == 0);
    bool wtOk = (wtFail == 0);
    if (idxOk && wtOk) {
        std::printf("PASS\n");
    } else {
        std::printf("FAIL\n");
    }

    // ---- 清理 ----
    aclDestroyTensor(xT);
    aclDestroyTensor(wT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(wtT);
    if (workspace) aclrtFree(workspace);
    aclDestroyAclOpExecutor(executor);
    aclrtFree(xDev);
    aclrtFree(wDev);
    aclrtFree(idxDev);
    aclrtFree(wtDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (idxOk && wtOk) ? 0 : 1;
}
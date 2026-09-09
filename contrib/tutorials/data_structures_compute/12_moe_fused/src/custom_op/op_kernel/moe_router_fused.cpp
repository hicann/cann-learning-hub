/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"); you may not use this file except in compliance
 * with the License.
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT
 * LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the program repository for details regarding the License.
 */

/**
 * moe_router_fused: MoE Router 4 合 1 融合算子（单 kernel，纯标量实现）
 *
 *   scores = x @ w_gate   [N,D] x [D,E] -> [N,E]   逐行标量点积（FP16 输入 / FP32 累加）
 *   p      = softmax(scores)                       逐行标量（max/FastExp/sum）
 *   (v,i)  = topk(p, K)                            K 轮 max（选择问题，K<=E<=32）
 *   w      = v / sum(v)                            标量 renorm 后转 FP16 写回
 *
 * 设计与同级章节 08_engineering_deployment_and_perf_analysis 的 attention_custom 完全同构：
 * 只用「标量 GM 访问 + UB 普通数组」这一最朴素、最稳定的 API 子集——
 *   - GlobalTensor GetValue/SetValue 直接标量读写 GM；不用 DataCopy/MTE、不用
 *     TPipe/TQue、不用任何向量指令、不申请 workspace；
 *   - 多核按 token 行「连续块」切分（core c 负责 [c*rowsPerCore, (c+1)*rowsPerCore)，
 *     尾核收尾），行间无共享状态。注意：不能用跨步行进切分（row += blockDim）——
 *     标量 GM 写经 L2（line=64B），多核并发写同一 line 会丢失部分写（910B 实测），
 *     连续块 + 64B 对齐边界可保证每条 line 只被一个核写入（详见 Process() 注释）；
 *   - 中间结果（scores/softmax/topk 暂存）全部留在 UB 普通数组（类成员），不落 GM。
 *
 * 为什么选纯标量（本环境 910B + CANN 9.0.0 实测）：
 *   1. 向量 Cast 不支持 BF16<->FP32，接口采用 FP16；
 *   2. SyncAll() 被归为 AIC 通道指令，触发 MIX 编译模式，向量指令在 AIC 侧为
 *      空操作且双任务竞写输出；
 *   3. reduce 族指令（vcadd/vpadd/vcgadd）不产出结果，高阶库 ReduceSum 挂起；
 *   4. 「循环内标量读取 + 向量指令混用」存在编译器缺陷（08 章纯标量实现即为
 *      规避该缺陷而作）：向量写入的数据经标量 GetValue 回读会命中竞态/读零。
 *   纯标量路线（0 向量指令、0 队列协议）整体绕开上述失效面，同构实现已在
 *   08 章 attention_custom 上验证可用。
 *
 * 性能形态说明：逐行 O(D*E) 次标量 GM 读 + 标量 FMA，计算吞吐受标量流水限制；
 * 融合收益体现在中间张量（scores/gate/topk_scores）零 GM 往返与 4 次发射合 1 次。
 * 定量实测与归因见 docs/design.md 及实验报告。
 */

#include <cstdint>
#include "kernel_operator.h"
#include "moe_router_fused_tiling.h"

using namespace AscendC;

namespace {
// 快速指数近似（与 08 章 attention_custom 同款，softmax 输入范围约 [-20, 0]）
__aicore__ inline float FastExp(float x)
{
    if (x < -20.0f) x = -20.0f;
    if (x > 0.0f) x = 0.0f;
    float y = x * 1.4426950408889634f;  // log2(e)
    int yi = static_cast<int>(y);
    float yf = y - static_cast<float>(yi);
    // 2^yf 多项式（yf ∈ [0,1)）
    float p = 1.0f + yf * (0.69314718056f + yf * (0.240226507f + yf * (0.05550410866f + yf * 0.009618129107f)));
    // 2^yi 位操作
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = static_cast<uint32_t>(yi + 127) << 23;
    return p * bits.f;
}

// scores/topk 标量数组上限（host 侧约束 E <= 32）
constexpr uint32_t kMaxE = 32;
}  // namespace

class KernelMoeRouterFused {
public:
    __aicore__ inline KernelMoeRouterFused() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wGate, GM_ADDR topkIdx, GM_ADDR topkWeights,
                                const MoeRouterFusedTilingData &tilingData);
    __aicore__ inline void Process();

    GlobalTensor<half> xGlobal;           // x [N, D]
    GlobalTensor<half> wGlobal;           // w_gate [D, E]（E 为含 padding 的第二维）
    GlobalTensor<int32_t> idxGlobal;      // topk_idx [N, K]
    GlobalTensor<half> wtGlobal;          // topk_weights [N, K]

    uint32_t N;
    uint32_t D;
    uint32_t E;       // w_gate 第二维（含 padding 列，决定 GM 行距）
    uint32_t realE;   // 真实专家数（<= E，仅 [0, realE) 参与 softmax/topk）
    uint32_t K;
    uint32_t blockDim;
    uint32_t rowsPerCore;

private:
    // UB 普通数组（类成员，与 08 章 scoresRow 同用法；合计 < 0.5KB，栈帧无压力）
    float scoresRow[kMaxE];
    float topkVal[kMaxE];
    int32_t topkIdxArr[kMaxE];
};

__aicore__ inline void KernelMoeRouterFused::Init(GM_ADDR x, GM_ADDR wGate, GM_ADDR topkIdx, GM_ADDR topkWeights,
                                                  const MoeRouterFusedTilingData &tilingData)
{
    N = tilingData.N;
    D = tilingData.D;
    E = tilingData.E;
    realE = tilingData.realE;
    K = tilingData.K;
    blockDim = tilingData.blockDim;
    rowsPerCore = tilingData.rowsPerCore;

    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), N * D);
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wGate), D * E);
    idxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(topkIdx), N * K);
    wtGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(topkWeights), N * K);
}

__aicore__ inline void KernelMoeRouterFused::Process()
{
    // 连续块切分（非跨步行进）：每核负责连续行区间 [n0, n1)。
    // 原因（910B 实测）：标量 GM 写经 L2 缓存，cache line 64B；多核并发写
    // 同一 line 会非确定性地丢失部分写（核间无写一致性）。跨步行进切分下
    // 各核输出行按 K*(4+2)B 粒度交错，必然共享 line；改为连续块切分并使
    // 块边界落在 64B 对齐处（host 侧按 R_align = 64/gcd(2K,64) 行对齐），
    // 保证每条 cache line 只被一个核写入。
    const uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t n0 = coreId * rowsPerCore;
    const uint32_t n1 = (coreId == blockDim - 1) ? N : (n0 + rowsPerCore);
    for (uint32_t n = n0; n < n1; ++n) {
        // ---- 1) scores[n,e] = Σ_d x[n,d] * w_gate[d,e]（标量点积，顺带求行内 max）----
        float rowMax = -1.0e30f;
        for (uint32_t e = 0; e < realE; ++e) {
            float acc = 0.0f;
            for (uint32_t d = 0; d < D; ++d) {
                acc += static_cast<float>(xGlobal.GetValue(n * D + d)) *
                       static_cast<float>(wGlobal.GetValue(d * E + e));
            }
            scoresRow[e] = acc;
            if (acc > rowMax) rowMax = acc;
        }

        // ---- 2) 逐行 softmax（先减行内 max 防溢出）----
        float rowSum = 0.0f;
        for (uint32_t e = 0; e < realE; ++e) {
            float v = FastExp(scoresRow[e] - rowMax);
            scoresRow[e] = v;
            rowSum += v;
        }
        float invSum = 1.0f / rowSum;
        for (uint32_t e = 0; e < realE; ++e) {
            scoresRow[e] *= invSum;
        }

        // ---- 3) TopK：K 轮 max（严格大于保留最小索引，与参考实现的并列处理一致）----
        for (uint32_t k = 0; k < K; ++k) {
            float best = -1.0e30f;
            int32_t bestIdx = 0;
            for (uint32_t e = 0; e < realE; ++e) {
                if (scoresRow[e] > best) {
                    best = scoresRow[e];
                    bestIdx = static_cast<int32_t>(e);
                }
            }
            topkVal[k] = best;
            topkIdxArr[k] = bestIdx;
            scoresRow[bestIdx] = -1.0e30f;  // 屏蔽已选专家
        }

        // ---- 4) renorm 并写回 GM（仅 N*K 个标量，融合后无中间张量落 GM）----
        float sumK = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            sumK += topkVal[k];
        }
        float invK = 1.0f / sumK;
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t off = n * K + k;
            idxGlobal.SetValue(off, topkIdxArr[k]);
            wtGlobal.SetValue(off, static_cast<half>(topkVal[k] * invK));
        }
    }
}

extern "C" __global__ __aicore__ void moe_router_fused(GM_ADDR x, GM_ADDR w_gate, GM_ADDR topk_idx,
                                                       GM_ADDR topk_weights, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MoeRouterFusedTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelMoeRouterFused op;
    op.Init(x, w_gate, topk_idx, topk_weights, tilingData);
    op.Process();
}

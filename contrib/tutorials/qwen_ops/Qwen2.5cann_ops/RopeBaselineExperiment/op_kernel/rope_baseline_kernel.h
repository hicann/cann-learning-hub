#ifndef ROPE_BASELINE_KERNEL_H
#define ROPE_BASELINE_KERNEL_H

#include "kernel_operator.h"

using namespace AscendC;

// ═══════════════════════════════════════════════════════════════
//  RoPE 动态 tiling 数据结构 (通用版)
//
//  对标 PyTorch apply_rotary_pos_emb:
//    y = x * cos + rotate_half(x) * sin
//    rotate_half(x): 前半后半交换取反 → [-x[half:], x[:half]]
//
//  cos/sin 由 Qwen2RotaryEmbedding 预计算 (含 rope_theta + scaling)
// ═══════════════════════════════════════════════════════════════

#pragma pack(push, 1)
struct RoPeTiling {
    uint32_t totalTokens = 0;
    uint32_t headDim     = 0;
    uint32_t coreNum     = 1;
    uint32_t rowsPerCore = 0;
};
#pragma pack(pop)

static_assert(sizeof(RoPeTiling) == 16, "Unexpected RoPe tiling data size");


// ═══════════════════════════════════════════════════════════════
//  RoPE 通用 Kernel — 外部 cos/sin, half-split 版本
//
//  ★ 刻意不做任何优化，作为 profiling 基线
// ═══════════════════════════════════════════════════════════════

class KernelRoPeBaseline {
public:
    __aicore__ inline KernelRoPeBaseline() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR cos_in,
        GM_ADDR sin_in,
        GM_ADDR output,
        uint32_t totalTokens,
        uint32_t headDim,
        uint32_t coreNum,
        uint32_t rowsPerCore)
    {
        totalTokens_ = totalTokens;
        headDim_     = headDim;
        coreNum_     = coreNum;
        rowsPerCore_ = rowsPerCore;
        totalElements_ = totalTokens * headDim;

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(input), totalElements_);
        cosGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(cos_in), totalElements_);
        sinGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(sin_in), totalElements_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(output), totalElements_);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        if (coreId >= coreNum_) return;

        uint32_t halfHead = headDim_ / 2;
        uint32_t startRow = coreId * rowsPerCore_;
        uint32_t endRow   = (startRow + rowsPerCore_ < totalTokens_)
                          ? (startRow + rowsPerCore_) : totalTokens_;

        for (uint32_t row = startRow; row < endRow; row++) {
            uint32_t rowOffset = row * headDim_;

            // y = x * cos + rotate_half(x) * sin
            // rotate_half: [-x[half:], x[:half]]
            // 对于 i ∈ [0, half):
            //   y[i]       = x[i]*cos[i] + (-x[i+half])*sin[i]
            //   y[i+half]  = x[i+half]*cos[i+half] + x[i]*sin[i+half]
            for (uint32_t i = 0; i < halfHead; i++) {
                uint32_t idx0 = rowOffset + i;
                uint32_t idx1 = rowOffset + i + halfHead;

                float x0 = inputGm_.GetValue(idx0);
                float x1 = inputGm_.GetValue(idx1);
                float c0 = cosGm_.GetValue(idx0);
                float s0 = sinGm_.GetValue(idx0);
                float c1 = cosGm_.GetValue(idx1);
                float s1 = sinGm_.GetValue(idx1);

                outputGm_.SetValue(idx0, x0 * c0 - x1 * s0);
                outputGm_.SetValue(idx1, x1 * c1 + x0 * s1);
            }
        }
    }

protected:
    GlobalTensor<float> inputGm_;
    GlobalTensor<float> cosGm_;
    GlobalTensor<float> sinGm_;
    GlobalTensor<float> outputGm_;
    uint32_t totalTokens_, headDim_, coreNum_, rowsPerCore_, totalElements_;
};

#endif // ROPE_BASELINE_KERNEL_H

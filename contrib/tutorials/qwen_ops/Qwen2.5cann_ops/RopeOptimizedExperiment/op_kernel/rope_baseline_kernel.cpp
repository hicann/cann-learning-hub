// RoPE 通用版 kernel 编译入口 (外部 cos/sin 版本)
// 职责: 包含 rope_baseline_kernel.h, 提供 extern "C" kernel 入口
#include "rope_baseline_kernel.h"

extern "C" __global__ __aicore__ void rope_baseline_kernel(
    GM_ADDR input, GM_ADDR cos_in, GM_ADDR sin_in, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // GM → 栈 拷贝 tiling
    RoPeTiling t;
    const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    uint32_t *dst = reinterpret_cast<uint32_t *>(&t);
    for (uint32_t i = 0; i < sizeof(RoPeTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }

    KernelRoPeBaseline op;
    op.Init(input, cos_in, sin_in, output,
        t.totalTokens, t.headDim, t.coreNum, t.rowsPerCore,
        t.seqLen, t.numHeads, t.trigTokens, t.compactTrig, t.tileSize);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// extern "C" — 供 Python ctypes 直调
extern "C" void rope_baseline_kernel_do(
    uint32_t blockDim, void *stream,
    uint8_t *input, uint8_t *cos_in, uint8_t *sin_in, uint8_t *output,
    uint8_t *workspace, uint8_t *tiling)
{
    rope_baseline_kernel<<<blockDim, nullptr, stream>>>(
        input, cos_in, sin_in, output, workspace, tiling);
}
#endif

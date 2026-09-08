#include "swiglu_baseline_kernel.h"

extern "C" __global__ __aicore__ void swiglu_baseline_kernel(
    GM_ADDR gate, GM_ADDR up, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    SwiGluTiling t;
    const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    uint32_t *dst = reinterpret_cast<uint32_t *>(&t);
    for (uint32_t i = 0; i < sizeof(SwiGluTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }

    KernelSwiGluBaseline op;
    op.Init(gate, up, output, t.totalSize, t.coreNum, t.elementsPerCore);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void swiglu_baseline_kernel_do(
    uint32_t blockDim, void *stream,
    uint8_t *gate, uint8_t *up, uint8_t *output,
    uint8_t *workspace, uint8_t *tiling)
{
    swiglu_baseline_kernel<<<blockDim, nullptr, stream>>>(
        gate, up, output, workspace, tiling);
}
#endif
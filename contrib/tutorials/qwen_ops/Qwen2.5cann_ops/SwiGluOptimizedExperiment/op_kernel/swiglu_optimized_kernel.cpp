#include "swiglu_optimized_kernel.h"

extern "C" __global__ __aicore__ void swiglu_optimized_kernel(
    GM_ADDR gate, GM_ADDR up, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    SwiGluTilingData t;
    const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    uint32_t *dst = reinterpret_cast<uint32_t *>(&t);

    for (uint32_t i = 0; i < sizeof(SwiGluTilingData) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }

    SwiGluOptimizedKernel op;
    op.Init(gate, up, output, t);
    op.Process();
}
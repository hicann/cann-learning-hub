#include "gemm_optimized_kernel.h"

extern "C" __global__ __aicore__ void gemm_optimized_kernel(
    GM_ADDR a, GM_ADDR bTrans, GM_ADDR c,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    GemmOptimizedTiling t;
    const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    uint32_t *dst = reinterpret_cast<uint32_t *>(&t);

    for (uint32_t i = 0; i < sizeof(GemmOptimizedTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }

    KernelGemmOptimized op;
    op.Init(a, bTrans, c, t);
    op.Process();
}
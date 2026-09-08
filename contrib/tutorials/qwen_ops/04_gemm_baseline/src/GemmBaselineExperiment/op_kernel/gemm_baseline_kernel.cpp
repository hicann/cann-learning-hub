#include "gemm_baseline_kernel.h"

extern "C" __global__ __aicore__ void gemm_baseline_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    GemmBaselineTiling t;
    const __gm__ uint32_t *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    uint32_t *dst = reinterpret_cast<uint32_t *>(&t);
    for (uint32_t i = 0; i < sizeof(GemmBaselineTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }

    KernelGemmBaseline op;
    op.Init(a, b, c, t.m, t.n, t.k, t.coreNum, t.rowsPerCore);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void gemm_baseline_kernel_do(
    uint32_t blockDim, void *stream,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *workspace, uint8_t *tiling)
{
    gemm_baseline_kernel<<<blockDim, nullptr, stream>>>(
        a, b, c, workspace, tiling);
}
#endif
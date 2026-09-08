#include "gqa_attention_baseline_kernel.h"

extern "C" __global__ __aicore__ void gqa_attention_baseline_kernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    const __gm__ uint32_t *u = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    const __gm__ float *f = reinterpret_cast<const __gm__ float *>(tiling);
    KernelGqaAttentionBaseline op;
    op.Init(query, key, value, output, u[0], u[1], u[2], u[3], u[4], u[5], u[7], u[8], u[9], f[10]);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void gqa_attention_baseline_kernel_do(uint32_t blockDim, void *stream,
    uint8_t *query, uint8_t *key, uint8_t *value, uint8_t *output, uint8_t *workspace, uint8_t *tiling)
{
    gqa_attention_baseline_kernel<<<blockDim, nullptr, stream>>>(query, key, value, output, workspace, tiling);
}
#endif

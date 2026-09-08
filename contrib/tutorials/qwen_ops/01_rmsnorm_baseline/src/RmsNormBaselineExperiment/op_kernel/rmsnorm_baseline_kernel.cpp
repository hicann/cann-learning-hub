#include "rmsnorm_baseline_kernel.h"

extern "C" __global__ __aicore__ void rmsnorm_baseline_kernel(
    GM_ADDR input, GM_ADDR weight, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    // Read each tiling field with its declared type. Copying float bit patterns
    // through a local uint32_t pointer is not a valid float load on AICore.
    const __gm__ uint32_t *uintTiling = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    const __gm__ float *floatTiling = reinterpret_cast<const __gm__ float *>(tiling);
    const uint32_t rows = uintTiling[0];
    const uint32_t hidden = uintTiling[1];
    const uint32_t coreNum = uintTiling[2];
    const uint32_t rowsPerCore = uintTiling[3];
    const float eps = floatTiling[4];
    const float invHidden = floatTiling[5];

    KernelRmsNormBaseline op;
    op.Init(input, weight, output, rows, hidden, coreNum, rowsPerCore, eps, invHidden);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void rmsnorm_baseline_kernel_do(
    uint32_t blockDim, void *stream,
    uint8_t *input, uint8_t *weight, uint8_t *output,
    uint8_t *workspace, uint8_t *tiling)
{
    rmsnorm_baseline_kernel<<<blockDim, nullptr, stream>>>(
        input, weight, output, workspace, tiling);
}
#endif

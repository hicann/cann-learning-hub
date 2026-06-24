#define IS_BLAZE true

#include "matmul_basic_tutorial_blaze.h"
#include "matmul_basic_tiling_data.h"

using namespace AscendC;

using layoutA = AscendC::Te::NDExtLayoutPtn;
using layoutB = AscendC::Te::NDExtLayoutPtn;
using layoutC = AscendC::Te::NDExtLayoutPtn;

extern "C" __global__ __aicore__ void matmul_basic_tutorial(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    GET_TILING_DATA_WITH_STRUCT(MatmulBasicTutorialTilingData, tilingData, tilingGM);
    MatmulBasicTutorial::MatMulBasicKernel<
        bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t,
        layoutA, layoutB, layoutC>(
        aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_basic_tutorial_do(uint32_t blockDim, void* stream,
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    matmul_basic_tutorial<<<blockDim, nullptr, stream>>>(aGM, bGM, biasGM, cGM, workspaceGM, tilingGM);
}
#endif

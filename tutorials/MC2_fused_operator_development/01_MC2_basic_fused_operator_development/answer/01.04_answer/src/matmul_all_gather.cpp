#include "kernel_operator.h"
#include "kernel/matmul_all_gather.h"

using namespace AscendC;
using namespace MatmulAllGatherImpl;

extern "C" __global__ __aicore__ void MatmulAllGather_Generic(
    GM_ADDR a, GM_ADDR b, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    MatmulAllGatherOp<bfloat16_t> op;
    op.Init(a, b, y, mc2Context, tilingGM, &pipe);
    op.Process();
}

extern "C" void matmul_all_gather_demo(uint32_t blockDim, void* stream,
    uint8_t* a, uint8_t* b, uint8_t* y, uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM)
{
    MatmulAllGather_Generic<<<blockDim, nullptr, stream>>>(
        a, b, y, workspaceGM, mc2Context, tilingGM);
}
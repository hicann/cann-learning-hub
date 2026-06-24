#ifndef MATMUL_BASIC_TUTORIAL_BLAZE_H
#define MATMUL_BASIC_TUTORIAL_BLAZE_H

#include "blaze/gemm/kernel/kernel_matmul_basic.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "matmul_basic_tiling_data.h"

namespace MatmulBasicTutorial {

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
    uint64_t FULL_LOAD_MODE = 0, uint64_t FUSED_OP_TYPE = 0,
    class KERNEL_SCHEDULE = Blaze::Gemm::KernelMmadMultiBlockBasic>
__aicore__ inline void MatMulBasicKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    const MatmulBasicTutorialTilingData& tilingData, int64_t batch = 0)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using LayoutBias = LayoutC;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE>;

    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<FULL_LOAD_MODE, FUSED_OP_TYPE, KERNEL_SCHEDULE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, batch},
        {aGM, bGM, cGM, biasGM},
        {},
        {tilingData.mL1,
         tilingData.nL1,
         tilingData.kL1,
         tilingData.baseM,
         tilingData.baseN,
         tilingData.baseK,
         tilingData.mTailCnt,
         tilingData.nTailCnt,
         tilingData.mBaseTailSplitCnt,
         tilingData.nBaseTailSplitCnt,
         tilingData.mTailMain,
         tilingData.nTailMain,
         tilingData.isHf32,
         tilingData.l1BufferNum,
         tilingData.l0cDB,
         tilingData.ubDB,
         tilingData.l2CacheDisable,
         tilingData.sliceM,
         tilingData.srcNdStride,
         tilingData.innerBatch}};
    MatmulKernel mm;
    mm(params);
}

} // namespace MatmulBasicTutorial
#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_mx_kernel_swat.h
 * \brief Kernel-side SWAT MX implementation for the non-full-load path.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"

#include "block/quant_matmul_mx_block_mmad_swat.h"
#include "block/quant_matmul_mx_block_scheduler_swat.h"
#include "utils/constant.h"

namespace Kernel {
// Keep the class template parameter list in one place so the declaration and
// out-of-line member definitions stay perfectly aligned.
#define QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
class QuantMatmulMxKernelSwat {
public:
    __aicore__ inline QuantMatmulMxKernelSwat()
    {}
    __aicore__ inline ~QuantMatmulMxKernelSwat()
    {}

    static constexpr bool weightNz = BlockMmad::weightNz;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleAType = typename BlockMmad::ScaleAType;
    using ScaleBType = typename BlockMmad::ScaleBType;
    using CType = typename BlockMmad::CType;

    // The kernel layer does not own a scheduling policy itself; it selects the
    // concrete scheduler from the block pipeline traits.
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, transA, transB, AType>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA = typename BlockMmad::LayoutA;
    using MakeLayoutB = typename BlockMmad::LayoutB;
    using MakeLayoutScaleA = typename BlockMmad::LayoutScaleA;
    using MakeLayoutScaleB = typename BlockMmad::LayoutScaleB;

    struct QBMMTiling {
        // Base tile shape and L0C buffering mode selected by host tiling.
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint8_t dbL0C;
    };

    struct Params {
        // Translate the serialized tiling payload once at the launcher
        // boundary, then keep the kernel, scheduler, and block MMAD layers on
        // their own typed parameter bundles.
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    // Launch one SWAT kernel instance on the current AIC block.
    __aicore__ inline void operator()(const Params& params);

private:
    // Bind raw GM addresses to typed kernel pointers once per launch.
    __aicore__ inline void ResetGmAddr(const Params& params);

    // Wrap the full GM tensors and repeatedly slice them into per-block views
    // according to the scheduler output.
    __aicore__ inline void Process(const Params& params, BlockSchedulerOp& bs);

    // `ProblemShape` stays generic at the template boundary, but the block
    // MMAD operator consumes a fixed 3D tuple internally.
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ ScaleAType* scaleAGmAddr_;
    __gm__ ScaleBType* scaleBGmAddr_;
};

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::operator()(
    const Params& params)
{
    // The quantized matmul compute path runs on AIC only. AIV blocks exit
    // immediately so mixed-core launches remain valid.
    if ASCEND_IS_AIV {
        return;
    }

    ResetGmAddr(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    TupleShape problemShape_ = ToShapeTuple(params.problemShape);
    // The kernel layer only builds the scheduling and tensor wrappers; the
    // heavy data movement and MMAD sequence lives in `BlockMmad`.
    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0cPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, enableL0cPingPong);
    Process(params, bs);
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::ResetGmAddr(
    const Params& params)
{
    // The sample launcher passes raw GM addresses through `Params`; convert
    // them here once so the hot loop works with typed pointers only.
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ ScaleAType*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ ScaleBType*>(params.mmadParams.scaleBGmAddr);
}

QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
__aicore__ inline void QuantMatmulMxKernelSwat<QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS>::Process(
    const Params& params, BlockSchedulerOp& bs)
{
    // Build full-GM tensor views once, then slice them per scheduled block.
    int64_t kScaleSize =
        ::CeilDiv(static_cast<int64_t>(params.problemShape.k), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
    auto layoutScaleA = MakeLayoutScaleA{}(params.problemShape.m, kScaleSize);
    auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
    auto layoutScaleB = MakeLayoutScaleB{}(kScaleSize, params.problemShape.n);
    auto layoutC =
        AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(params.problemShape.m, params.problemShape.n);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
    auto gmScaleA =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    constexpr int64_t kPos = 0L;
    while (bs.template GetTileIdx<weightNz>(blockIdx)) {
        // The scheduler packs GM origin into M/N and retains logical tile
        // indices in K/B so shape reconstruction still works.
        int64_t mPos = AscendC::Te::Get<MNK_M>(blockIdx);
        int64_t nPos = AscendC::Te::Get<MNK_N>(blockIdx);
        BlockShape singleShape = bs.template GetBlockShape<weightNz>(blockIdx);
        if (AscendC::Te::Get<MNK_M>(singleShape) <= 0 || AscendC::Te::Get<MNK_N>(singleShape) <= 0) {
            // Tail splitting can create empty logical slices; ignore them and
            // stop the current block once no useful work remains.
            return;
        }

        // `blockIdx` now carries both GM origin and logical tile metadata.
        auto gmBlockA =
            gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleShape), params.problemShape.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleShape), kScaleSize));
        auto gmBlockB =
            gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(params.problemShape.k, AscendC::Te::Get<MNK_N>(singleShape)));
        auto gmBlockScaleB =
            gmScaleB.Slice(AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(kScaleSize, AscendC::Te::Get<MNK_N>(singleShape)));
        auto gmBlockC =
            gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(AscendC::Te::Get<MNK_M>(singleShape), AscendC::Te::Get<MNK_N>(singleShape)));

        // The block MMAD layer owns all data movement below GM granularity and
        // performs the actual accumulation for this scheduled tile.
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);
    }
}

} // namespace Kernel

#undef QBMM_MX_KERNEL_NO_FULL_LOAD_CLASS_TEM_PARAMS
#undef QBMM_MX_KERNEL_NO_FULL_LOAD_FUN_TEM_PARAMS

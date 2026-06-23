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
 * \file moe_distribute_dispatch_quant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_QUANT_H
#define MOE_DISTRIBUTE_DISPATCH_QUANT_H

#include "moe_distribute_comm.h"

namespace Quant {

constexpr int DIGIT_TWO = 2;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;

using namespace AscendC;

__aicore__ inline constexpr uint32_t GetUbBlockSizeDispatch()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSizeDispatch()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template<typename T>
__aicore__ inline void ComputeMaxExp(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vdExp0;
        MicroAPI::RegTensor<T> vdExp1;
        MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        MicroAPI::RegTensor<uint16_t> vdExpExtract1;

        MicroAPI::RegTensor<uint16_t> expMaskBF16;
        MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        MicroAPI::RegTensor<uint16_t> vdMaxExp;
        MicroAPI::MaskReg scaleMask1;
        MicroAPI::MaskReg scaleMask2;
        MicroAPI::UnalignReg u1;
        static constexpr MicroAPI::CastTrait castTraitHalf2Bf16 = {
            MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = MicroAPI::UpdateMask<T>(totalCountInUB);
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE,
            MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            if constexpr (Std::IsSame<T, half>::value) {
                MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                MicroAPI::And(vdExpExtract0, (MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16,
                    scaleMask1);
                MicroAPI::And(vdExpExtract1, (MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16,
                    scaleMask1);
            } else {
                MicroAPI::And(vdExpExtract0, (MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16,
                    scaleMask1);
                MicroAPI::And(vdExpExtract1, (MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16,
                    scaleMask1);
            }

            MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            MicroAPI::DataCopyUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxExpAddr,
                vdMaxExp, u1, elementAfterReduce);
        }
        MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template<typename T>
__aicore__ inline void ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
    __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(uint16_t);
    uint16_t f8Emax = std::is_same<T, fp8_e4m3fn_t>::value ? FP8_E4M3_MAX_EXP : FP8_E5M2_MAX_EXP;
    uint16_t loopNumScale = Ceil(totalScaleInUB, vlForHalfNumber);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> expMask;
        MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        MicroAPI::RegTensor<uint16_t> vdMaxExp;

        MicroAPI::MaskReg cmpResult;
        MicroAPI::MaskReg zeroMask;
        MicroAPI::MaskReg preMaskScale;
        MicroAPI::RegTensor<uint16_t> maxExpValue;
        MicroAPI::Duplicate(maxExpValue, f8Emax);
        MicroAPI::RegTensor<uint16_t> sharedExp;
        MicroAPI::RegTensor<uint16_t> scaleValue;
        MicroAPI::RegTensor<uint16_t> scaleBias;
        MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        MicroAPI::RegTensor<uint16_t> halfScale;
        MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        MicroAPI::Duplicate(zeroRegTensor, 0);
        MicroAPI::RegTensor<uint16_t> nanRegTensor;
        MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        MicroAPI::MaskReg invalidDataMask;
        MicroAPI::MaskReg specialDataMask;
        MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vdMaxExp,
                maxExpAddr, vlForHalfNumber);
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                preMaskScale);

            MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

            MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue, vlForHalfNumber / DIGIT_TWO,
                preMaskScale);

            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                preMaskScale);
            MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr,
                halfScale, vlForHalfNumber, preMaskScale);
        }
    }
}

template <typename T, typename U, RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void ComputeData(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint32_t totalCountInUB2 = totalCountInUB * DIGIT_TWO;
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg dataMask1;
        MicroAPI::MaskReg dataMask2;
        MicroAPI::MaskReg dataMask3;
        MicroAPI::MaskReg dataMask4;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint16_t,
            MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        MicroAPI::RegTensor<float> floatScaleForMul;
        MicroAPI::RegTensor<T> vdExp0;
        MicroAPI::RegTensor<T> vdExp1;

        MicroAPI::RegTensor<float> vdExp0FP32Zero;
        MicroAPI::RegTensor<float> vdExp0FP32One;
        MicroAPI::RegTensor<float> vdExp1FP32Zero;
        MicroAPI::RegTensor<float> vdExp1FP32One;
        MicroAPI::RegTensor<U> vdExp0FP8Zero;
        MicroAPI::RegTensor<U> vdExp0FP8One;
        MicroAPI::RegTensor<U> vdExp1FP8Zero;
        MicroAPI::RegTensor<U> vdExp1FP8One;

        static constexpr MicroAPI::CastTrait castTraitZero = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait castTraitOne = {
            MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait castTrait32to8 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask2 = MicroAPI::UpdateMask<T>(totalCountInUB);
            dataMask3 = MicroAPI::UpdateMask<T>(totalCountInUB2);
            dataMask4 = MicroAPI::UpdateMask<T>(totalCountInUB2);
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            if constexpr (Std::IsSame<T, half>::value) {
                MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                MicroAPI::Cast<float, bfloat16_t, castTraitZero>(floatScaleForMul,
                    (MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                MicroAPI::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                MicroAPI::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);
                MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                MicroAPI::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                MicroAPI::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);
                MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                MicroAPI::Interleave(vdExp0FP32Zero, vdExp1FP32Zero, vdExp0FP32Zero, vdExp1FP32Zero);
                MicroAPI::Interleave(vdExp0FP32One, vdExp1FP32One, vdExp0FP32One, vdExp1FP32One);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp1FP32Zero, dataMask3);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp0FP32One, dataMask4);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            } else {
                MicroAPI::Mul(vdExp0, vdExp0, (MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
                MicroAPI::Mul(vdExp1, vdExp1, (MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
                MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
                MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            }
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (MicroAPI::RegTensor<int8_t>&)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
}
}

namespace Mc2QuantKernel {
using namespace AscendC;
using namespace Mc2Kernel;

class MoeDistributeDispatchQuant{
public:
    using XType = float16_t;
    using ExpandXOutType = fp8_e5m2_t;

    uint32_t axisH_{0};
    uint32_t hOutSizeAlign_{0};

    LocalTensor<float> floatLocalTemp_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;

    __aicore__ inline MoeDistributeDispatchQuant() = default;

    __aicore__ inline void SetQuantInitParams(
        LocalTensor<float> floatLocalTemp, GlobalTensor<uint8_t> dynamicScalesOutGMTensor) 
    {
        floatLocalTemp_ = floatLocalTemp;
        dynamicScalesOutGMTensor_ = dynamicScalesOutGMTensor;
    }

    __aicore__ inline void QuantInit(uint32_t &hAlignSize_, uint32_t &hOutSize_,
                                     int32_t &tokenQuantAlign_, uint32_t &hScaleIdxSize_, uint32_t &scaleOutBytes, uint32_t axisH)
    {
        axisH_ = axisH;
        hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN; // scale起始放置偏移
        hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; //用于搬入token数据xInQueue_大小申请
        hOutSizeAlign_ = Align256(axisH_) * sizeof(ExpandXOutType);
        hAlignSize_ = Align128(axisH_) * sizeof(XType); // MX量化计算scale时每次搬入128个数据
        hOutSizeAlign_ += Align2(Ceil32(axisH_)); 
        scaleOutBytes = Align2(Ceil32(axisH_)) * sizeof(fp8_e8m0_t); // MX量化每32个值生成一个scale，且scale数量需为偶数
        uint32_t hScaleSizeAlign = Ceil(hOutSizeAlign_, UB_ALIGN) * UB_ALIGN; //保证后面填充三元组的起始地址对齐32
        tokenQuantAlign_ = hScaleSizeAlign / sizeof(int32_t);
        // 实际搬运大小，搬运Align32(token_align + scaleOutBytes) + 3*4B(三元组)
        hScaleIdxSize_ = hScaleSizeAlign + EXPAND_IDX_INFO * sizeof(int32_t);
    }

    __aicore__ inline void QuantProcess(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal)
    {
        QuantDynamicMxFp8(outLocal, inLocal);
    }

    __aicore__ inline void QuantDynamicMxFp8(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal)
    {
        uint32_t mxScaleNum = Align2(Ceil32(axisH_));
        __ubuf__ XType* srcAddr = (__ubuf__ XType*)inLocal.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)floatLocalTemp_.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)floatLocalTemp_[Align32(mxScaleNum)].GetPhyAddr();
        __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)outLocal[Align256<uint32_t>(axisH_)].GetPhyAddr();

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_); // 计算最大Exp
        Quant::ComputeScale<ExpandXOutType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum); // 计算scales并填充
        Quant::ComputeData<XType, ExpandXOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_); // 计算量化后的expandx并填充
    }

    __aicore__ inline void CopyScalesToOut(uint32_t currentTokenIndex, uint32_t scaleOutBytes,
                                           LocalTensor<ExpandXOutType> &quantTok, DataCopyExtParams &scaleOutParams)
    {
        auto scaleLT = quantTok[(Ceil(axisH_, UB_ALIGN) * UB_ALIGN)].template ReinterpretCast<uint8_t>();
        scaleLT = quantTok[Align256<uint32_t>(axisH_)].template ReinterpretCast<uint8_t>();
        DataCopyPad(dynamicScalesOutGMTensor_[currentTokenIndex * scaleOutBytes], scaleLT, scaleOutParams);
    }
};
}
#endif // MOE_DISTRIBUTE_DISPATCH_QUANT_H
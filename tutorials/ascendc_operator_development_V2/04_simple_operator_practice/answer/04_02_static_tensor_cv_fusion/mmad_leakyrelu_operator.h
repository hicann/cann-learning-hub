/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MMAD_LEAKYRELU_OPERATOR_H
#define MMAD_LEAKYRELU_OPERATOR_H

// half type, cube block: [16, 16]
constexpr uint32_t CUBE_BLOCK = 16;

template <
    uint32_t M, uint32_t K, uint32_t N, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN,
    uint32_t baseM, uint32_t baseK, uint32_t baseN>
class KernelMmad {
public:
    __aicore__ inline KernelMmad() {}

    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias, __gm__ uint8_t* c)
    {
        aGM.SetGlobalBuffer((__gm__ half*)a);
        bGM.SetGlobalBuffer((__gm__ half*)b);
        biasGM.SetGlobalBuffer((__gm__ half*)bias);
        cGM.SetGlobalBuffer((__gm__ half*)c);
        aGM = aGM[AscendC::GetBlockIdx() * singleCoreM * K];
        cGM = cGM[AscendC::GetBlockIdx() * singleCoreM * N];
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalMemAllocator<AscendC::Hardware::L1> l1Allocator;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0A> l0aAllocator;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0B> l0bAllocator;
        AscendC::LocalMemAllocator<AscendC::Hardware::L0C> l0cAllocator;
        AscendC::LocalMemAllocator<AscendC::Hardware::BIAS> biasAllocator;

        AscendC::LocalTensor<half> a1Local = l1Allocator.Alloc<AscendC::TPosition::A1, half>(baseM * baseK);
        AscendC::LocalTensor<half> b1Local = l1Allocator.Alloc<AscendC::TPosition::B1, half>(baseK * baseN);
        AscendC::LocalTensor<half> bias1Local = l1Allocator.Alloc<AscendC::TPosition::C1, half>(baseN);
        AscendC::LocalTensor<half> a2Local = l0aAllocator.Alloc<AscendC::TPosition::A2, half>(baseM * baseK);
        AscendC::LocalTensor<half> b2Local = l0bAllocator.Alloc<AscendC::TPosition::B2, half>(baseK * baseN);
        AscendC::LocalTensor<float> bias2Local = biasAllocator.Alloc<AscendC::TPosition::C2, float>(baseN);
        AscendC::LocalTensor<float> cLocal = l0cAllocator.Alloc<AscendC::TPosition::CO1, float>(baseM * baseN);

        AscendC::DataCopy(a1Local, aGM, AscendC::Nd2NzParams{1, baseM, baseK, 0, K, baseM, 1, 0});
        AscendC::DataCopy(b1Local, bGM, AscendC::Nd2NzParams{1, baseK, baseN, 0, N, baseK, 1, 0});
        AscendC::DataCopy(bias1Local, biasGM, baseN);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        AscendC::DataCopy(bias2Local, bias1Local, {1, static_cast<uint16_t>(baseN * sizeof(float) / 64), 0, 0});
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        for (int i = 0; i < baseM / CUBE_BLOCK; ++i) {
            AscendC::LoadData(a2Local[i * baseK * CUBE_BLOCK], a1Local[i * 512 / sizeof(half)],
                AscendC::LoadData2DParams{0, baseK / CUBE_BLOCK, baseM / CUBE_BLOCK, 0, 0, false, 0});
        }
        for (int i = 0; i < baseK / CUBE_BLOCK; ++i) {
            AscendC::LoadData(b2Local[i * baseN * CUBE_BLOCK], b1Local[i * 512 / sizeof(half)],
                AscendC::LoadData2DParams{0, baseN / CUBE_BLOCK, baseK / CUBE_BLOCK, 0, 0, true, 0});
        }
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        AscendC::LoadData(a2Local, a1Local,
            AscendC::LoadData2DParamsV2{0, 0, baseM / CUBE_BLOCK, baseK / CUBE_BLOCK,
                baseM / CUBE_BLOCK, baseM / CUBE_BLOCK, false, 0});
        AscendC::LoadData(b2Local, b1Local,
            AscendC::LoadData2DParamsV2{0, 0, baseK / CUBE_BLOCK, baseN / CUBE_BLOCK,
                baseK / CUBE_BLOCK, baseN / CUBE_BLOCK, true, 0});
#endif
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

        AscendC::MmadParams mmADParams;
        mmADParams.m = baseM;
        mmADParams.n = baseN;
        mmADParams.k = baseK;
        mmADParams.cmatrixInitVal = false;
        AscendC::Mmad(cLocal, a2Local, b2Local, bias2Local, mmADParams);

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::Fixpipe(cGM, cLocal,
            AscendC::FixpipeParamsV220{baseN, baseM, baseM, N, false, QuantMode_t::F322F16, 0, 1, 0, 0, 0});
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
    }

private:
    AscendC::GlobalTensor<half> aGM;
    AscendC::GlobalTensor<half> bGM;
    AscendC::GlobalTensor<half> biasGM;
    AscendC::GlobalTensor<half> cGM;
};

template <
    uint32_t M, uint32_t K, uint32_t N, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN,
    uint32_t baseM, uint32_t baseK, uint32_t baseN>
class KernelResidualAddLeakyRelu {
public:
    __aicore__ inline KernelResidualAddLeakyRelu() {}

    __aicore__ inline void Init(__gm__ uint8_t* residual, __gm__ uint8_t* c)
    {
        residualGM.SetGlobalBuffer((__gm__ half*)residual);
        cGM.SetGlobalBuffer((__gm__ half*)c);
        uint32_t cubeBlockOffset = AscendC::GetBlockIdx() / 2 * singleCoreM * N;
        uint32_t vectorBlockOffset = AscendC::GetBlockIdx() % 2 * (baseM / 2 * N);
        residualGM = residualGM[cubeBlockOffset + vectorBlockOffset];
        cGM = cGM[cubeBlockOffset + vectorBlockOffset];
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        AscendC::LocalTensor<half> vecLocal =
            ubAllocator.Alloc<AscendC::TPosition::VECCALC, half>(baseM / 2 * baseN);
        AscendC::LocalTensor<half> residualLocal =
            ubAllocator.Alloc<AscendC::TPosition::VECCALC, half>(baseM / 2 * baseN);

        AscendC::CrossCoreWaitFlag(0);

        AscendC::DataCopy(vecLocal, cGM, baseM / 2 * baseN);
        AscendC::DataCopy(residualLocal, residualGM, baseM / 2 * baseN);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Add(vecLocal, vecLocal, residualLocal, baseM / 2 * baseN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LeakyRelu(vecLocal, vecLocal, (half)0.001, baseM / 2 * baseN);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(cGM, vecLocal, baseM / 2 * baseN);
    }

private:
    AscendC::GlobalTensor<half> residualGM;
    AscendC::GlobalTensor<half> cGM;
};

template <
    uint32_t M, uint32_t K, uint32_t N, uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN,
    uint32_t baseM, uint32_t baseK, uint32_t baseN>
__global__ __mix__(1, 2) void mmad_radd_leakyrelu_custom(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias, __gm__ uint8_t* residual, __gm__ uint8_t* c)
{
    AscendC::InitSocState();
    if ASCEND_IS_AIC {
        KernelMmad<M, K, N, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN> op;
        op.Init(a, b, bias, c);
        op.Process();
    }
    if ASCEND_IS_AIV {
        KernelResidualAddLeakyRelu<M, K, N, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN> op;
        op.Init(residual, c);
        op.Process();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

#endif // MMAD_LEAKYRELU_OPERATOR_H

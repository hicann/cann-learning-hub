/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mmad_gelu_adv_operator.h
 * \brief 基于静态 Tensor 的 MMAD-GELU 高性能 CV 融合样例。
 */

#ifndef MMAD_GELU_ADV_OPERATOR_H
#define MMAD_GELU_ADV_OPERATOR_H

#include "kernel_operator.h"

__aicore__ __inline__ constexpr uint32_t DivCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// half type, cube block: [16, 16]
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;
constexpr uint32_t L1_PINGPONG_BYTES = 256 * 1024;
constexpr bool IS_B_TRANSPOSE = true;
constexpr float GELU_COEFF_A = 0.044715f;
constexpr float GELU_COEFF_B = -1.595769f;

#include "gelu_unroll_practice.h"

// Fixpipe配置
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

template <
    uint32_t M, uint32_t K, uint32_t N, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t singleCoreM,
    uint32_t singleCoreK, uint32_t singleCoreN, uint32_t stepKa, uint32_t stepKb>
class KernelMmadGelu {
public:
    __aicore__ inline KernelMmadGelu() {}

    __aicore__ inline void Init(
        __gm__ uint8_t* xMatrix, __gm__ uint8_t* yMatrix, __gm__ uint8_t* bias, __gm__ uint8_t* zMatrix)
    {
        aGMOri.SetGlobalBuffer((__gm__ half*)xMatrix);
        bGMOri.SetGlobalBuffer((__gm__ half*)yMatrix);
        biasGMOri.SetGlobalBuffer((__gm__ half*)bias);
        cGMOri.SetGlobalBuffer((__gm__ float*)zMatrix);

        // xUB 接收双目的 Fixpipe 输出；geluOutUB 保存 GELU 结果。
        xUB = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, 0, baseM / 2 * baseN);
        geluOutUB = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, geluOutUBAddr, baseM / 2 * baseN);
    }

    __aicore__ inline void Process()
    {
        InitComputeParams();
        static_assert(
            2 * baseM * baseK * stepKa * sizeof(half) + baseN * sizeof(half) <= L1_PINGPONG_BYTES,
            "A1 Ping/Pong and Bias exceed the first 256 KiB of L1");
        static_assert(
            2 * baseK * baseN * stepKb * sizeof(half) <= L1_PINGPONG_BYTES,
            "B1 Ping/Pong exceed the second 256 KiB of L1");
        static_assert(
            baseM * baseK * sizeof(half) <= L0_PINGPONG_BYTES,
            "A2 Ping/Pong buffer exceeds 32 KiB");
        static_assert(
            baseK * baseN * sizeof(half) <= L0_PINGPONG_BYTES,
            "B2 Ping/Pong buffer exceeds 32 KiB");
        static_assert(baseM * baseN * sizeof(float) <= 256 * 1024, "L0C tile exceeds 256 KiB");
        static_assert(
            baseM * baseN * sizeof(float) <= 248 * 1024,
            "Two Vector Core UB buffers together exceed the 3510 usable UB capacity");
        static_assert(
            singleCoreK % (baseK * stepKa) == 0 && singleCoreK % (baseK * stepKb) == 0,
            "singleCoreK must contain complete A/B L1 copy-in batches");
        static_assert(
            M % 2 == 0 && singleCoreM % 2 == 0 && baseM % 2 == 0,
            "M partitions must be even for dual-Vector-Core row splitting");

        // ============================================================
        // Cube Core 侧：Buffer 分配 + 计算循环 + Fixpipe 搬出
        // ============================================================
        uint32_t a1PingpongSize = baseM * baseK * stepKa;
        uint32_t b1PingpongSize = baseK * baseN * stepKb;
        uint32_t a2PingpongSize = baseM * baseK;
        uint32_t b2PingpongSize = baseK * baseN;

        AscendC::LocalTensor<half> a1LocalPing(AscendC::TPosition::A1, 0, a1PingpongSize);
        AscendC::LocalTensor<half> a1LocalPong(AscendC::TPosition::A1, a1PingpongSize * sizeof(half), a1PingpongSize);
        AscendC::LocalTensor<half> a2LocalPing(AscendC::TPosition::A2, 0, a2PingpongSize);
        AscendC::LocalTensor<half> a2LocalPong(AscendC::TPosition::A2, L0_PINGPONG_BYTES, a2PingpongSize);

        AscendC::LocalTensor<half> b1LocalPing(AscendC::TPosition::B1, L1_PINGPONG_BYTES, b1PingpongSize);
        AscendC::LocalTensor<half> b1LocalPong(
            AscendC::TPosition::B1, L1_PINGPONG_BYTES + b1PingpongSize * sizeof(half), b1PingpongSize);
        AscendC::LocalTensor<half> b2LocalPing(AscendC::TPosition::B2, 0, b2PingpongSize);
        AscendC::LocalTensor<half> b2LocalPong(AscendC::TPosition::B2, L0_PINGPONG_BYTES, b2PingpongSize);
        AscendC::LocalTensor<half> bias1Local(
            AscendC::TPosition::C1, 2 * a1PingpongSize * sizeof(half), baseN);
        AscendC::LocalTensor<float> bias2Local(AscendC::TPosition::C2, 0, baseN);
        AscendC::LocalTensor<float> cLocal(AscendC::TPosition::CO1, 0, baseM * baseN);

        if ASCEND_IS_AIC {
            InitAicSyncFlags();
        }

        for (uint32_t nBlockIdx = 0; nBlockIdx < nLoopCount; nBlockIdx++) {
            for (uint32_t mBlockIdx = 0; mBlockIdx < mLoopCount; mBlockIdx++) {
                if ASCEND_IS_AIC {
                    ProcessLoopAic(
                        a1LocalPing, a1LocalPong, a2LocalPing, a2LocalPong, b1LocalPing, b1LocalPong, b2LocalPing,
                        b2LocalPong, bias1Local, bias2Local, cLocal, mBlockIdx, nBlockIdx);
                }
                if ASCEND_IS_AIV {
                    ProcessLoopAiv(mBlockIdx, nBlockIdx);
                }
            }
        }

        if ASCEND_IS_AIC {
            WaitAicSyncFlags();
        }
    }

private:
    __aicore__ inline void InitAicSyncFlags()
    {
        // ============================================================
        // 同步标志初始化：预置反向事件，建立首次 WaitFlag 所需的初始可写状态
        // ============================================================
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    }

    __aicore__ inline void WaitAicSyncFlags()
    {
        // ============================================================
        // 等待所有同步完成
        // ============================================================
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    }

    // Cube Core 侧单个 M/N 分块计算：DataCopyIn + DataLoad + Compute + Fixpipe 搬出
    __aicore__ inline void ProcessLoopAic(
        AscendC::LocalTensor<half>& a1LocalPing, AscendC::LocalTensor<half>& a1LocalPong,
        AscendC::LocalTensor<half>& a2LocalPing, AscendC::LocalTensor<half>& a2LocalPong,
        AscendC::LocalTensor<half>& b1LocalPing, AscendC::LocalTensor<half>& b1LocalPong,
        AscendC::LocalTensor<half>& b2LocalPing, AscendC::LocalTensor<half>& b2LocalPong,
        AscendC::LocalTensor<half>& bias1Local, AscendC::LocalTensor<float>& bias2Local,
        AscendC::LocalTensor<float>& cLocal, uint32_t mBlockIdx, uint32_t nBlockIdx)
    {
        // ============================================================
        // DataCopyIn 进度跟踪变量
        // ============================================================
        uint32_t a1NextKChunkIdx = 0;
        uint32_t b1NextKChunkIdx = 0;
        uint8_t a1CopyInIdx = 0;
        uint8_t b1CopyInIdx = 0;

        // Bias 地址由 N 分块确定。每个输出 tile 准备一次，K 循环复用 C2。
        PrepareBias(bias1Local, bias2Local, nBlockIdx);

        // ---- 搬入 A1 Ping 和 B1 Ping 的首个批量块 ----
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        DataCopyInA(a1LocalPing, a1NextKChunkIdx, mBlockIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        a1NextKChunkIdx += stepKa;
        a1CopyInIdx ^= 1;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        DataCopyInB(b1LocalPing, b1NextKChunkIdx, nBlockIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        b1NextKChunkIdx += stepKb;
        b1CopyInIdx ^= 1;

        // ---- K 方向主循环 ----
        for (uint32_t kBlockIdx = 0; kBlockIdx < kLoopCount; kBlockIdx++) {
            uint32_t a1ReadIdx = (kBlockIdx / stepKa) % 2;
            uint32_t b1ReadIdx = (kBlockIdx / stepKb) % 2;
            uint32_t kOffsetInChunkA = kBlockIdx % stepKa;
            uint32_t kOffsetInChunkB = kBlockIdx % stepKb;

            AscendC::LocalTensor<half> a1ReadBuf = (a1ReadIdx == 0) ? a1LocalPing : a1LocalPong;
            AscendC::LocalTensor<half> b1ReadBuf = (b1ReadIdx == 0) ? b1LocalPing : b1LocalPong;

            AscendC::LocalTensor<half> a2Local = (mte1DBFlag == 0) ? a2LocalPing : a2LocalPong;
            AscendC::LocalTensor<half> b2Local = (mte1DBFlag == 0) ? b2LocalPing : b2LocalPong;

            // ---- 反向同步：等待上一轮 Compute 释放 L0 缓冲区 ----
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1DBFlag);

            // ---- 正向同步：等待 L1 批量块数据就绪 ----
            if (kOffsetInChunkA == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((a1ReadIdx == 0) ? EVENT_ID0 : EVENT_ID1);
            }
            if (kOffsetInChunkB == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((b1ReadIdx == 0) ? EVENT_ID2 : EVENT_ID3);
            }

            // ---- DataLoad: L1 → L0 ----
            DataLoadA(a1ReadBuf, a2Local, mBlockIdx, kOffsetInChunkA);
            DataLoadB(b1ReadBuf, b2Local, nBlockIdx, kOffsetInChunkB);

            // ---- 反向同步：当前 L1 批量块消费完成，允许 DataCopyIn 覆盖对应缓冲区 ----
            if ((kOffsetInChunkA + 1) == stepKa) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((a1ReadIdx == 0) ? EVENT_ID0 : EVENT_ID1);
            }
            if ((kOffsetInChunkB + 1) == stepKb) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((b1ReadIdx == 0) ? EVENT_ID2 : EVENT_ID3);
            }

            // ---- Compute: Mmad 矩阵乘累加 ----
            Compute(cLocal, a2Local, b2Local, bias2Local, kBlockIdx, mBlockIdx, nBlockIdx);

            // ---- 搬入下一个 L1 批量块，使 DataCopyIn 与 Compute 流水重叠 ----
            if (((kBlockIdx == 0) || ((kOffsetInChunkB + 1) == stepKb)) && b1NextKChunkIdx < kLoopCount) {
                AscendC::LocalTensor<half> b1WriteBuf = (b1CopyInIdx == 0) ? b1LocalPing : b1LocalPong;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((b1CopyInIdx == 0) ? EVENT_ID2 : EVENT_ID3);
                DataCopyInB(b1WriteBuf, b1NextKChunkIdx, nBlockIdx);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((b1CopyInIdx == 0) ? EVENT_ID2 : EVENT_ID3);
                b1NextKChunkIdx += stepKb;
                b1CopyInIdx ^= 1;
            }
            if (((kBlockIdx == 0) || ((kOffsetInChunkA + 1) == stepKa)) && a1NextKChunkIdx < kLoopCount) {
                AscendC::LocalTensor<half> a1WriteBuf = (a1CopyInIdx == 0) ? a1LocalPing : a1LocalPong;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((a1CopyInIdx == 0) ? EVENT_ID0 : EVENT_ID1);
                DataCopyInA(a1WriteBuf, a1NextKChunkIdx, mBlockIdx);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((a1CopyInIdx == 0) ? EVENT_ID0 : EVENT_ID1);
                a1NextKChunkIdx += stepKa;
                a1CopyInIdx ^= 1;
            }
        }

        // ---- CopyOut Cube Core 侧：Fixpipe L0C → 双 Vector Core UB ----
        CopyOutAic(cLocal, mBlockIdx, nBlockIdx);
    }

    // Vector Core 侧单个 M/N 分块计算
    __aicore__ inline void ProcessLoopAiv(uint32_t mBlockIdx, uint32_t nBlockIdx)
    {
        GeluFromUB(mBlockIdx, nBlockIdx);
    }

    __aicore__ inline void GeluRegBaseCompute(
        const AscendC::LocalTensor<float>& xLocal, const AscendC::LocalTensor<float>& yLocal, uint32_t n)
    {
        __ubuf__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();
        __ubuf__ float* yAddr = (__ubuf__ float*)yLocal.GetPhyAddr();
        GeluVf(xAddr, yAddr, n);
    }

    __aicore__ inline void InitComputeParams()
    {
        constexpr uint32_t mIter = DivCeil(M, singleCoreM);
        // __mix__(1,2) 模式下，每个 AI Core 由 1 个 Cube Core 和 2 个 Vector Core 组成。
        // GetBlockIdx() 在 Cube Core 和 Vector Core 分支分别连续编号。
        uint32_t aiCoreId;
        if ASCEND_IS_AIC {
            aiCoreId = AscendC::GetBlockIdx();
        } else {
            aiCoreId = AscendC::GetBlockIdx() / 2;
        }
        uint32_t mIterIdx = aiCoreId % mIter;
        uint32_t nIterIdx = aiCoreId / mIter;

        uint64_t gmOffsetA = mIterIdx * singleCoreM * K;
        uint64_t gmOffsetB = IS_B_TRANSPOSE ? nIterIdx * K * singleCoreN : nIterIdx * singleCoreN;
        uint64_t gmOffsetC = mIterIdx * singleCoreM * N + nIterIdx * singleCoreN;
        aGM = aGMOri[gmOffsetA];
        bGM = bGMOri[gmOffsetB];
        biasGM = biasGMOri[nIterIdx * singleCoreN];
        cGM = cGMOri[gmOffsetC];

        actualSingleCoreM = M - mIterIdx * singleCoreM;
        actualSingleCoreM = actualSingleCoreM < singleCoreM ? actualSingleCoreM : singleCoreM;
        actualSingleCoreN = N - nIterIdx * singleCoreN;
        actualSingleCoreN = actualSingleCoreN < singleCoreN ? actualSingleCoreN : singleCoreN;

        kLoopCount = DivCeil(singleCoreK, baseK);
        mLoopCount = DivCeil(actualSingleCoreM, baseM);
        nLoopCount = DivCeil(actualSingleCoreN, baseN);

        baseNCount = actualSingleCoreN / baseN;
        tailN = actualSingleCoreN % baseN;
        tailNAlign = DivCeil(tailN, CUBE_BLOCK) * CUBE_BLOCK;

        baseMCount = actualSingleCoreM / baseM;
        tailM = actualSingleCoreM % baseM;
        tailMAlign = DivCeil(tailM, CUBE_BLOCK) * CUBE_BLOCK;
    }

    // GM → A1: 将 A 矩阵的 stepKa 个 baseM * baseK 子块批量搬入 L1
    __aicore__ inline void DataCopyInA(AscendC::LocalTensor<half> a1Local, uint32_t kChunkIdx, uint32_t mBlockIdx)
    {
        uint32_t curM = (mBlockIdx != baseMCount) ? baseM : tailM;
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = curM;
        nd2nzParams.dValue = baseK * stepKa;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = K;
        nd2nzParams.dstNzC0Stride = baseM;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM[kChunkIdx * baseK + mBlockIdx * K * baseM], nd2nzParams);
    }

    // GM → B1: 将 B 矩阵的 stepKb 个 baseK * baseN 子块批量搬入 L1
    __aicore__ inline void DataCopyInB(AscendC::LocalTensor<half> b1Local, uint32_t kChunkIdx, uint32_t nBlockIdx)
    {
        uint32_t curN = (nBlockIdx != baseNCount) ? baseN : tailN;
        AscendC::Nd2NzParams nd2nzParams;
        if constexpr (!IS_B_TRANSPOSE) {
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = baseK * stepKb;
            nd2nzParams.dValue = curN;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = N;
            nd2nzParams.dstNzC0Stride = baseK * stepKb;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(b1Local, bGM[kChunkIdx * baseK * N + nBlockIdx * baseN], nd2nzParams);
        } else {
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = curN;
            nd2nzParams.dValue = baseK * stepKb;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = K;
            nd2nzParams.dstNzC0Stride = baseN;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(b1Local, bGM[kChunkIdx * baseK + nBlockIdx * baseN * K], nd2nzParams);
        }
    }

    // A1 → A2: 将 L1 中的一个 baseM * baseK 搬入 L0
    __aicore__ inline void DataLoadA(
        AscendC::LocalTensor<half> a1Local, AscendC::LocalTensor<half> a2Local, uint32_t mBlockIdx,
        uint32_t kOffsetInChunkA)
    {
        uint32_t srcAddr = kOffsetInChunkA * baseK * baseM;
        uint32_t curMAlign = (mBlockIdx != baseMCount) ? baseM : tailMAlign;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        AscendC::LoadData3DParamsV2<half> loadDataParams;
        loadDataParams.l1H = 1;
        loadDataParams.l1W = baseM;
        loadDataParams.channelSize = baseK;
        loadDataParams.kExtension = baseK;
        loadDataParams.mExtension = curMAlign;
        loadDataParams.mStartPt = 0;
        loadDataParams.kStartPt = 0;
        AscendC::LoadData(a2Local, a1Local[srcAddr], loadDataParams);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeil(curMAlign, CUBE_BLOCK);
        loadDataParams.kStep = DivCeil(baseK, CUBE_BLOCK);
        loadDataParams.srcStride = DivCeil(baseM, CUBE_BLOCK);
        loadDataParams.dstStride = DivCeil(curMAlign, CUBE_BLOCK);
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local[srcAddr], loadDataParams);
#endif
    }

    // B1 → B2: 将 L1 中的一个 baseK * baseN 搬入 L0
    __aicore__ inline void DataLoadB(
        AscendC::LocalTensor<half> b1Local, AscendC::LocalTensor<half> b2Local, uint32_t nBlockIdx,
        uint32_t kOffsetInChunkB)
    {
        uint32_t srcAddr = kOffsetInChunkB * baseK * (IS_B_TRANSPOSE ? baseN : CUBE_BLOCK);
        uint32_t curNAlign = (nBlockIdx != baseNCount) ? baseN : tailNAlign;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
        if constexpr (!IS_B_TRANSPOSE) {
            // B 非转置: 使用 LoadData（卷积数据搬运）v2 完成 [K, N] → [N, K] 转置搬运
            AscendC::LoadData3DParamsV2<half> loadDataParams;
            loadDataParams.l1H = 1;
            loadDataParams.l1W = baseK * stepKb;
            loadDataParams.channelSize = baseN;
            loadDataParams.kExtension = curNAlign;
            loadDataParams.mExtension = baseK;
            loadDataParams.mStartPt = kOffsetInChunkB * baseK;
            loadDataParams.kStartPt = 0;
            loadDataParams.strideW = 1;
            loadDataParams.strideH = 1;
            loadDataParams.filterW = 1;
            loadDataParams.filterH = 1;
            loadDataParams.dilationFilterW = 1;
            loadDataParams.dilationFilterH = 1;
            loadDataParams.filterSizeW = false;
            loadDataParams.filterSizeH = false;
            loadDataParams.enTranspose = true;
            loadDataParams.fMatrixCtrl = false;
            AscendC::LoadData(b2Local, b1Local, loadDataParams);
        } else {
            // B 转置: 按 CUBE_BLOCK 粒度逐块搬入 L0，无需转置
            AscendC::LoadData2DParams loadDataParams;
            uint32_t dstOffset = curNAlign * CUBE_BLOCK;
            uint32_t srcOffset = baseN * CUBE_BLOCK;
            loadDataParams.repeatTimes = DivCeil(curNAlign, CUBE_BLOCK);
            loadDataParams.srcStride = 1;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            for (int i = 0; i < DivCeil(baseK, CUBE_BLOCK); ++i) {
                AscendC::LoadData(b2Local[i * dstOffset], b1Local[srcAddr + i * srcOffset], loadDataParams);
            }
        }
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if constexpr (!IS_B_TRANSPOSE) {
            // B 非转置: 使用 LoadData2D V2 完成 [K, N] → [N, K] 转置搬运
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = DivCeil(baseK, CUBE_BLOCK);
            loadDataParams.kStep = DivCeil(curNAlign * sizeof(half), 32);
            loadDataParams.srcStride = DivCeil(baseK * stepKb, CUBE_BLOCK);
            loadDataParams.dstStride = DivCeil(curNAlign, CUBE_BLOCK);
            loadDataParams.ifTranspose = true;
            AscendC::LoadData(b2Local, b1Local[srcAddr], loadDataParams);
        } else {
            // B 转置: 无需转置，直接按块搬运
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = DivCeil(curNAlign, CUBE_BLOCK);
            loadDataParams.kStep = DivCeil(baseK * sizeof(half), 32);
            loadDataParams.srcStride = DivCeil(baseN, CUBE_BLOCK);
            loadDataParams.dstStride = DivCeil(curNAlign, CUBE_BLOCK);
            loadDataParams.ifTranspose = false;
            AscendC::LoadData(b2Local, b1Local[srcAddr], loadDataParams);
        }
#endif
    }

    __aicore__ inline void PrepareBias(
        AscendC::LocalTensor<half>& bias1Local, AscendC::LocalTensor<float>& bias2Local, uint32_t nBlockIdx)
    {
        AscendC::DataCopy(bias1Local, biasGM[nBlockIdx * baseN], baseN);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::DataCopy(
            bias2Local, bias1Local,
            {1, static_cast<uint16_t>(baseN * sizeof(float) / 64), 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID2);
    }

    // 执行 MMAD 计算并累加到 CO1。
    __aicore__ inline void Compute(
        AscendC::LocalTensor<float> cLocal, AscendC::LocalTensor<half> a2Local, AscendC::LocalTensor<half> b2Local,
        AscendC::LocalTensor<float> bias2Local, uint32_t kBlockIdx, uint32_t mBlockIdx, uint32_t nBlockIdx)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1DBFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1DBFlag);
        uint32_t curM = (mBlockIdx != baseMCount) ? baseM : tailM;
        uint32_t curN = (nBlockIdx != baseNCount) ? baseN : tailN;
        AscendC::MmadParams mmadParams;
        mmadParams.m = curM;
        mmadParams.n = curN;
        mmadParams.k = baseK;
        mmadParams.cmatrixInitVal = (kBlockIdx == 0);
        mmadParams.isBias = (kBlockIdx == 0);
        mmadParams.unitFlag = (kBlockIdx != kLoopCount - 1) ? 2 : 3;
        if (kBlockIdx == 0) {
            AscendC::Mmad(cLocal, a2Local, b2Local, bias2Local, mmadParams);
        } else {
            AscendC::Mmad(cLocal, a2Local, b2Local, mmadParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1DBFlag);
        mte1DBFlag ^= 1;
    }

    // Cube Core 侧：Fixpipe 将 L0C 按 M 维均分到两个 Vector Core 的 UB。
    __aicore__ inline void CopyOutAic(AscendC::LocalTensor<float> cLocal, uint32_t mBlockIdx, uint32_t nBlockIdx)
    {
        uint32_t curMAlign = (mBlockIdx != baseMCount) ? baseM : tailMAlign;
        uint32_t curM = (mBlockIdx != baseMCount) ? baseM : tailM;
        uint32_t curN = (nBlockIdx != baseNCount) ? baseN : tailN;

        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = DivCeil(curM, 2) * 2;
        fixpipeParams.nSize = curN;
        fixpipeParams.srcStride = curMAlign;
        fixpipeParams.dstStride = curN;
        fixpipeParams.dualDstCtl = 0b01;
        fixpipeParams.unitFlag = 3;
        AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_UB>(xUB, cLocal, fixpipeParams);

        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
    }

    // Vector Core 侧：L0C-UB 直通、RegBase GELU、UB-GM 搬出。
    __aicore__ inline void GeluFromUB(uint32_t mBlockIdx, uint32_t nBlockIdx)
    {
        AscendC::CrossCoreWaitFlag(0x8);

        uint32_t curM = (mBlockIdx != baseMCount) ? baseM : tailM;
        uint32_t curN = (nBlockIdx != baseNCount) ? baseN : tailN;

        // dualDstCtl=0b01 按 M 维拆分，每个 Vector Core 持有 M/2 行。
        uint32_t halfM = curM / 2;
        uint32_t computeLen = halfM * curN;

        GeluRegBaseCompute(xUB, geluOutUB, computeLen);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        uint32_t localSubIdx = AscendC::GetSubBlockIdx() % 2;
        uint32_t offset = halfM * N;
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = halfM;
        copyParams.blockLen = curN * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = (N - curN) * sizeof(float);
        AscendC::DataCopyPad<float>(
            cGM[mBlockIdx * baseM * N + nBlockIdx * baseN + localSubIdx * offset], geluOutUB, copyParams);
    }

private:
    AscendC::GlobalTensor<half> aGM;
    AscendC::GlobalTensor<half> bGM;
    AscendC::GlobalTensor<half> biasGM;
    AscendC::GlobalTensor<float> cGM;
    AscendC::GlobalTensor<half> aGMOri;
    AscendC::GlobalTensor<half> bGMOri;
    AscendC::GlobalTensor<half> biasGMOri;
    AscendC::GlobalTensor<float> cGMOri;

    AscendC::LocalTensor<float> xUB;
    AscendC::LocalTensor<float> geluOutUB;

    uint32_t actualSingleCoreM, actualSingleCoreN;
    uint32_t mLoopCount, nLoopCount, kLoopCount;
    uint32_t baseMCount, baseNCount;
    uint32_t tailM, tailN;
    uint32_t tailMAlign, tailNAlign;
    uint8_t mte1DBFlag = 0;

    static constexpr uint32_t geluOutUBAddr = baseM / 2 * baseN * sizeof(float);
};

template <
    uint32_t M, uint32_t K, uint32_t N, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t singleCoreM,
    uint32_t singleCoreK, uint32_t singleCoreN, uint32_t stepKa, uint32_t stepKb>
__global__ __mix__(1, 2) void mmad_gelu_adv(
    __gm__ uint8_t* xMatrix, __gm__ uint8_t* yMatrix, __gm__ uint8_t* bias, __gm__ uint8_t* zMatrix)
{
    AscendC::InitSocState();
    KernelMmadGelu<M, K, N, baseM, baseK, baseN, singleCoreM, singleCoreK, singleCoreN, stepKa, stepKb> op;
    op.Init(xMatrix, yMatrix, bias, zMatrix);
    op.Process();
    AscendC::PipeBarrier<PIPE_ALL>();
}

#endif  // MMAD_GELU_ADV_OPERATOR_H

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
 * \file moe_distribute_dispatch.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_H
#define MOE_DISTRIBUTE_DISPATCH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/math/cumsum.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "shmem.h"
#include "moe_distribute_comm.h"
#include "moe_distribute_dispatch_quant.h"

struct alignas(8) DispatchTilingData {
    uint32_t epWorldSize;                // epWorldSize
    uint32_t epRankId;                   // epRankId
    uint32_t moeExpertNum;               // moe expert number
    uint32_t bs;                         // bs
    uint32_t k;                          // k
    uint32_t h;                          // h
    uint32_t expertTokenNumsType;        // expert token nums type, support 0: cumsum mode, 1: count mode
    uint32_t aivNum;                     // aivNum
    uint64_t symMemSize;                 // size of symmetric memory in bytes
};

namespace DispatchImpl {
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL; // 64 + 320
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL; // 384 * 2，0/1标识区偏移
constexpr uint64_t CUMSUM_CAL_OFFSET = 868UL * 1024UL; // 768 + 100
constexpr uint64_t CUMSUM_FLAG_OFFSET = 876UL * 1024UL; // 868 + 8
constexpr int32_t  MAX_UB_SIZE = 240 * 1024;
constexpr uint32_t COMPARE_COUNT_PER_BLOCK = 256 / sizeof(int32_t);
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint8_t QUANT_PADDING_VALUE = 0;
constexpr int8_t FLOAT_OVERFLOW_MODE_CTRL = 60;
constexpr AscendC::CumSumConfig cumSumConfig{true, true, false};

using namespace Mc2Kernel;
using namespace AscendC;
using namespace Mc2QuantKernel;

class MoeDistributeDispatch {
public:
    using XType = float16_t;
    using ExpandXOutType = fp8_e5m2_t;

    __aicore__ inline MoeDistributeDispatch() {};
    __aicore__ inline void Init(__gm__ void* shmemSpace, GM_ADDR x, GM_ADDR expertIds,
                            GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                            GM_ADDR sendCountsOut,
                            GM_ADDR workspaceGM, TPipe *pipe, const DispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ExpIdsCopy();
    __aicore__ inline void SetDataStatus();
    __aicore__ inline void SetTilingData(const DispatchTilingData *tilingData);
    __aicore__ inline void MaxSizeCal();
    __aicore__ inline void SetTilingDataAndCal(const DispatchTilingData *tilingData);
    __aicore__ inline void SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> expertMaskBuf, TBuf<> outBuf);
    __aicore__ inline void CalExpertSendNum(TBuf<> outBuf, TBuf<> expertMaskBuf);
    __aicore__ inline void AllToAllDispatch();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void CalAndSendCnt();
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor,
         LocalTensor<uint32_t> &gatherTmpTensor, LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlag(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyInAndOut(LocalTensor<int32_t> xOutInt32Tensor,
                                        GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum);
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void SplitExpertNumToCore(uint32_t &delCurExpertGroupNum, uint32_t &groupIdx);
    __aicore__ inline void FillTriple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void TokenToExpertInQuant(int32_t toRankId, GlobalTensor<ExpandXOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                                uint32_t srcTokenIndex, uint32_t toExpertId, uint32_t toExpertIndex);

    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)aclshmem_ptr(shmemContextGM_, rankId) + winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)aclshmem_ptr(shmemContextGM_, rankId) + STATUS_REGION_OFFSET + dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetSymMemDataRegionAddr()
    {
        return (GM_ADDR)shmemContextGM_ + winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetSymMemStatusRegionAddr()
    {
        return (GM_ADDR)shmemContextGM_ + STATUS_REGION_OFFSET + dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t InitWinState(GlobalTensor<uint32_t> selfDataStatusGMTensor, TBuf<> dataStateBuf)
    {
        LocalTensor<uint32_t> dataStateLocalTensor = dataStateBuf.Get<uint32_t>();
        DataCopy(dataStateLocalTensor, selfDataStatusGMTensor, UB_ALIGN / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        uint32_t dataState = dataStateLocalTensor.GetValue(ZERONE_STATE_POS);
        dataStateLocalTensor.SetValue(ZERONE_STATE_POS, dataState == 0 ? 1 : 0);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
    
        DataCopy(selfDataStatusGMTensor, dataStateLocalTensor, UB_ALIGN / sizeof(uint32_t));

        return dataState;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIdsTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<ExpandXOutType> outTensor_;
    LocalTensor<ExpandXOutType> tempTensor_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<uint8_t> flagCompResultU8_;
    LocalTensor<uint64_t> flagCompResultLtU64_;
    LocalTensor<uint32_t> flagRecvGatherMask_;
    LocalTensor<ExpandXOutType> xTmpTensor_;

    LocalTensor<float> flagGatherOutTensor_;
    LocalTensor<float> flagRecvTensor_;

    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_;
    GM_ADDR expandXOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR statusSpaceGM_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGM_;
    
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    int32_t epRankId_{0};
    uint32_t aivId_{0};           // aiv id
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t hOutSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hAlignSize_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t startId_;
    uint32_t endId_;
    uint32_t sendNum_;
    uint32_t statusCntAlign_;
    uint32_t dataState_{0};
    uint32_t tBufRealSize_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t flagPadOffset_{0};
    uint64_t symMemSize_{0};
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t axisHCommu_{0};
    uint32_t hCommuSize_{0};
    uint32_t maskSizePerExpert_{0};
    uint32_t expertIdsBufSize_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t delLastExpertId_{0};
    uint32_t remainderExpertNum_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedAllToAll_{0};
    uint32_t maxSize_{0};
    uint32_t expertIdsSize_{0};
    __gm__ void* shmemContextGM_;

    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams hCopyParams_;
    DataCopyParams dataStateParams_{1U, sizeof(uint32_t), 0U, 0U};

    MoeDistributeDispatchQuant quantInst_;
};

__aicore__ inline void MoeDistributeDispatch::SetTilingData(
    const DispatchTilingData *tilingData)
{
    epWorldSize_ = tilingData->epWorldSize;
    epRankId_ = tilingData->epRankId;
    moeExpertNum_ = tilingData->moeExpertNum;
    axisBS_ = tilingData->bs;
    axisH_ = tilingData->h;
    axisK_ = tilingData->k;
    aivNum_ = tilingData->aivNum;
    symMemSize_ = tilingData->symMemSize;
    expertTokenNumsType_ = tilingData->expertTokenNumsType;
}

__aicore__ inline void MoeDistributeDispatch::SetTilingDataAndCal(
    const DispatchTilingData *tilingData)
{
    SetTilingData(tilingData);

    // ===== TODO 1：完成量化版本 Buffer 大小计算（开发者补全本注释块内的代码） =====
    // 步骤 1：moeExpertNumPerRank_ = ?；expertIdsCnt_ = ?；hOutSize_ = axisH_ * sizeof(ExpandXOutType)。
    // 步骤 2：调用 quantInst_.QuantInit(...)，让量化模板算出
    //         hAlignSize_、hOutSize_（注意此处会被改写为量化后的对齐尺寸）、
    //         tokenQuantAlign_、hOutSizeAlign_、scaleOutBytes_。
    //         签名：QuantInit(uint32_t &hAlignSize_, uint32_t &hOutSize_,
    //                         int32_t &tokenQuantAlign_, uint32_t &hOutSizeAlign_,
    //                         uint32_t &scaleOutBytes_, uint32_t axisH);
    // 步骤 3：在 hOutSizeAlign_ 末尾追加一个 UB_ALIGN 字节，承载三元组。
    //         hOutSizeAlign_ = tokenQuantAlign_ * sizeof(int32_t) + UB_ALIGN;
    // 步骤 4：以 SPLIT_BLOCK_DATA_SIZE = 480B 切块，得到
    //         blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
    //         hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    //         axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    //         expertPerSizeOnWin_ = axisBS_ * hCommuSize_;
    // ===== TODO 1 区结束 =====

    // 以下与非量化版本完全一致，已直接给出，无需修改：
    rscvStatusNum_ = epWorldSize_ * moeExpertNumPerRank_;
    statusCntAlign_ = Ceil(moeExpertNum_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;
    aivUsedCumSum_ = moeExpertNum_ / 32; // 单核处理32个专家cnt发送
    aivUsedCumSum_ = (aivUsedCumSum_ == 0) ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= (aivNum_ / 2)) ? (aivNum_ / 2) : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= CUMSUM_MAX_CORE_NUM) ? CUMSUM_MAX_CORE_NUM : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= rscvStatusNum_) ? rscvStatusNum_ : aivUsedCumSum_; // 确保每个核至少处理一个状态
    aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;
}

__aicore__ inline void MoeDistributeDispatch::SetDataStatus()
{
    statusDataSpaceGM_ = GetWindStateAddrByRankId(epRankId_);

    selfDataStatusGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)(statusDataSpaceGM_ + FLAG_FIELD_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, dataStateBuf);
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffset_ = dataState_ * (symMemSize_ / BUFFER_NUM)
                         + axisBS_ * axisK_ * hSizeAlignCombine;
}

__aicore__ inline void MoeDistributeDispatch::MaxSizeCal()
{
    uint32_t hFp32Size = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t bsKAlign256 = Ceil(expertIdsCnt_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    expertIdsSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    maxSize_ = hFp32Size > expertIdsSize_ ? hFp32Size : expertIdsSize_;
    maxSize_ = maxSize_ > xActivateMaskSize ? maxSize_ : xActivateMaskSize;
    maxSize_ = maxSize_ > bsKAlign256 ? maxSize_ : bsKAlign256;
}

__aicore__ inline void MoeDistributeDispatch::Init(__gm__ void* shmemSpace, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
    GM_ADDR workspaceGM, TPipe *pipe, const DispatchTilingData *tilingData)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    shmemContextGM_ = shmemSpace;
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t*)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t*)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(expandIdxOut));
    SetTilingDataAndCal(tilingData);

    SetDataStatus();
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCntWorkspaceGM_ = workspaceGM;
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankId_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGM_));
    selfRankWinInGMTensor_.SetGlobalBuffer((__gm__ float*)(statusDataSpaceGM_));
    windowGM_ = GetWindAddrByRankId(epRankId_);
    hCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    dataStateParams_ = {1U, sizeof(uint32_t), 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(ExpandXOutType)), 0U, 0U, 0U};
    MaxSizeCal();
}

__aicore__ inline void MoeDistributeDispatch::FillTriple(
    LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;        // 0:epRankId index
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;   // 1:token index
    xOutTint32(tokenQuantAlign_ + 2) = k;            // 2:topK value index
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

__aicore__ inline void MoeDistributeDispatch::TokenToExpertInQuant(
    int32_t toRankId, GlobalTensor<ExpandXOutType> dstWinGMTensor,
    TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex, uint32_t fillExpertIdx, uint32_t quantExpertIdx)
{
    // ===== TODO 2：完成量化发送流程（开发者补全本注释块内的代码） =====
    // 步骤 1：构造 DataCopyPadExtParams<XType>{true, 0, 0, 0}（量化要求 isPad = true，
    //         保证尾部不整对齐时填 0）。
    // 步骤 2：从 inQueue 分配 LocalTensor<XType> xInTensor；
    //         以 uint8_t 视角对 xInTensor 调用 Duplicate(QUANT_PADDING_VALUE, ...)
    //         填 0，长度 = Align128(axisH_) * sizeof(XType)（MX 量化一次按 128 数据
    //         计算 scale，尾部不整 128 必须填 0 防脏数据）；插入 SyncFunc<V_MTE2>()。
    // 步骤 3：DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, ...)
    //         随后 inQueue.EnQue / inQueue.DeQue<XType>。
    // 步骤 4：quantInst_.QuantProcess(tempTensor_, xInTensor)
    //         执行 MX 量化（ComputeMaxExp / ComputeScale / ComputeData 已封装）。
    // 步骤 5：FillTriple(tempTensor_, srcTokenIndex, fillExpertIdx) 写三元组；
    //         随后 inQueue.FreeTensor<XType>(xInTensor) + SyncFunc<S_V>()。
    // 步骤 6：把 tempTensor_ 以 int32 视角分两段 Copy 到 outTensor_（int32 视角）：
    //         先插入 PipeBarrier<PIPE_V>() 等待 QuantProcess 内的 Cast 完成。
    //         第一段：起点 outTensorInt32[flagPadOffset_/4]，长度 64 个 int32 (=256B)；
    //         第二段：起点偏移 +64，长度 56 个 int32 (= (480-256)/4)；
    //         迭代次数 = blockCntPerToken_，stride {1, 1, 16, 15}。
    // 步骤 7：SyncFunc<V_MTE3>() 后调用
    //         aclshmemx_mte_put_nbi(dstWinGMTensor,
    //             outTensor_[flagPadOffset_ / sizeof(ExpandXOutType)],
    //             axisHCommu_, toRankId, 0);
    // 步骤 8：flagPadOffset_ = hCommuSize_ - flagPadOffset_（ping-pong 翻转，下一条 token
    //         写入 outTensor_ 的另一半）。
    // ===== TODO 2 区结束 =====
}

__aicore__ inline void MoeDistributeDispatch::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else { // aiv中后面aivUsedCumSum_个核给cusum计算使用
        newAivId = aivId_ - aivUsedAllToAll_;
    }
    startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

__aicore__ inline void MoeDistributeDispatch::CalExpertSendNum(TBuf<> outBuf, TBuf<> expertMaskBuf)
{
    uint64_t maskCnt = 0;
    uint32_t mask = expertIdsCnt_;
    uint32_t compareCount = Ceil(mask * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(int32_t);
    LocalTensor<uint8_t> expertMaskTensorU8 = expertMaskBuf.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = expertMaskBuf.Get<uint32_t>();
    LocalTensor<int32_t> gatherTempTensor = outBuf.Get<int32_t>();
    for (int32_t expertIndex = 0; expertIndex < sendNum_; expertIndex++) {
        int32_t dstExpertId = expertIndex + startId_;
        if ((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0)) {
            dstExpertId = delLastExpertId_;
        }
        CompareScalar(expertMaskTensorU8[maskSizePerExpert_ * expertIndex], validExpertIdsTensor_, dstExpertId, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        GatherMask(gatherTempTensor, validExpertIdsTensor_, expertMaskTensorU32[maskSizePerExpert_ * expertIndex / sizeof(uint32_t)],
            true, mask, {1, 1, 0, 0}, maskCnt); // 是否可以简化计算
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt));
    }
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * BUFFER_NUM / sizeof(float));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void MoeDistributeDispatch::SplitExpertNumToCore(uint32_t &delCurExpertGroupNum, uint32_t &groupIdx)
{
    sendNum_ = moeExpertNum_ / aivUsedAllToAll_;
    remainderExpertNum_ = moeExpertNum_ % aivUsedAllToAll_;
    startId_ = sendNum_ * aivId_;
    if (remainderExpertNum_ != 0) {
        int32_t remainderGroupSize = remainderExpertNum_;
        delLastExpertId_ = aivId_ % remainderExpertNum_;
        delCurExpertGroupNum = aivUsedAllToAll_ / remainderGroupSize;
        if (delLastExpertId_ < aivUsedAllToAll_ % remainderGroupSize) {
            delCurExpertGroupNum++;
        }
        groupIdx = aivId_ / remainderGroupSize;
        delLastExpertId_ += aivUsedAllToAll_ * sendNum_;
        sendNum_ += 1;
    }
}

__aicore__ inline void MoeDistributeDispatch::SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue,
    TBuf<> expertMaskBuf, TBuf<> outBuf)
{
    // 分核
    uint32_t delCurExpertGroupNum, groupIdx, calExpertIdsIdx;
    SplitExpertNumToCore(delCurExpertGroupNum, groupIdx);
    // 计算专家发送数据量 && 发送
    CalExpertSendNum(outBuf, expertMaskBuf);
    uint32_t maskN64Num = Ceil(expertIdsCnt_, 64); // 64：ScalarGetSFFValue按照64长度一次计算
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    LocalTensor<uint64_t> expertMaskTensorU64 = expertMaskBuf.Get<uint64_t>();
    for (int32_t index = 0; index < sendNum_; index++) {
        int32_t dstTokenPreCnt = 0;
        int32_t expertIndex = (index + epRankId_ % sendNum_) % sendNum_;
        int32_t maskExpertU64Cnt = maskSizePerExpert_ * expertIndex / sizeof(uint64_t);
        if (tokenNumToExpertTensor_(expertIndex) == 0) {
            continue;
        }
        int32_t dstExpertId = expertIndex + startId_;
        dstExpertId = ((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0)) ? delLastExpertId_ : dstExpertId;
        for (int32_t maskIndex = 0; maskIndex < maskN64Num; maskIndex++) {
            uint64_t dstExpInfoMask = expertMaskTensorU64(maskIndex + maskExpertU64Cnt);
            int64_t curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            while (curValidIdx >= 0) {
                calExpertIdsIdx = curValidIdx + SFFVALUE_SIZE * maskIndex; // 64：ScalarGetSFFValue按照64长度一次计算
                if (calExpertIdsIdx >= expertIdsCnt_) {
                    break;
                }
                int32_t topKIndex = calExpertIdsIdx % axisK_;
                int32_t srcTokenIndex = calExpertIdsIdx / axisK_;
                int32_t toRankId = dstExpertId / moeExpertNumPerRank_;
                GM_ADDR rankGM = (__gm__ uint8_t*)(GetSymMemDataRegionAddr() +
                                (expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                hCommuSize_ * dstTokenPreCnt); // 计算地址偏移
                dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType*)rankGM);
                if (!((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0) && (dstTokenPreCnt % delCurExpertGroupNum != groupIdx))) {
                    uint32_t quantExpertIdx = dstExpertId;
                    TokenToExpertInQuant(toRankId, dstWinGMTensor, inQueue, srcTokenIndex, topKIndex, quantExpertIdx);
                }
                dstTokenPreCnt++;
                uint64_t cleanMask = ~(uint64_t(1) << curValidIdx);
                dstExpInfoMask = cleanMask & dstExpInfoMask; // 将当前64bit中处理的1置0
                curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            }
        }
    }
}

__aicore__ inline void MoeDistributeDispatch::AllToAllDispatch()
{
    // 使用的全局参数
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<> tempBuf, outBuf, expertIdsBuf, expertMaskBuf;
    TBuf<> smoothScalesBuf, tokenNumToExpertBuf, receiveDataCastFloatBuf;
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(inQueue, BUFFER_NUM, hAlignSize_);
    tpipe_->InitBuffer(outBuf, hCommuSize_ * BUFFER_NUM);
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    outTensor_ = outBuf.Get<ExpandXOutType>();

    hOutAlignUbSize_ = Ceil(hOutSizeAlign_, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tempBuf, hOutAlignUbSize_);
    tpipe_->InitBuffer(receiveDataCastFloatBuf, maxSize_);
    tpipe_->InitBuffer(smoothScalesBuf, maxSize_);
    tempTensor_ = tempBuf.Get<ExpandXOutType>();
    floatLocalTemp_ = receiveDataCastFloatBuf.Get<float>();
    dstExpBuf_ = receiveDataCastFloatBuf; // 内存复用
    subExpBuf_ = smoothScalesBuf;         // 内存复用

    quantInst_.SetQuantInitParams(floatLocalTemp_, dynamicScalesOutGMTensor_);
    ExpIdsCopy();

    maskSizePerExpert_ = Ceil((expertIdsBufSize_ / sizeof(int32_t)) / 8, UB_ALIGN) * UB_ALIGN; // 8 is 1byte->8bit
    uint32_t expertMaskBufSize = maskSizePerExpert_ * Ceil(moeExpertNum_, aivUsedAllToAll_);
    tpipe_->InitBuffer(expertMaskBuf, expertMaskBufSize);
    tpipe_->InitBuffer(tokenNumToExpertBuf, Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    tokenNumToExpertTensor_ = tokenNumToExpertBuf.Get<int32_t>();
    LocalTensor<int32_t> expertMaskTensor = expertMaskBuf.Get<int32_t>();
    Duplicate<int32_t>(expertMaskTensor, 0, int32_t(expertMaskBufSize / sizeof(int32_t)));
    SyncFunc<AscendC::HardEvent::V_S>();
    SendToMoeExpert(inQueue, expertMaskBuf, outBuf);
}

__aicore__ inline void MoeDistributeDispatch::CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

__aicore__ inline void MoeDistributeDispatch::CalAndSendCnt()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    uint32_t maskCnt = expertIdsCnt_;
    SplitToCore(moeExpertNum_, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum, false);
    if (startExpertId >= moeExpertNum_) {return;}
    uint64_t mask[2] = { 0x101010101010101, 0 }; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1, 8); // 0x3F800000为float的1 8为一次操作8个block
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
        int32_t curExpertCnt = 0;
        int32_t cntPosIndex = (curExpertId - startExpertId) * 8 + 1;               // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        int32_t curMoeExpertId = curExpertId;
        CalTokenSendExpertCnt(curMoeExpertId, maskCnt, curExpertCnt);
        statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstRankId = curExpertId;
        uint32_t offset = STATE_OFFSET * epRankId_;
        dstRankId = curExpertId / moeExpertNumPerRank_;
        offset += curExpertId % moeExpertNumPerRank_ * epWorldSize_ * STATE_OFFSET;

        GM_ADDR rankGM = GetSymMemStatusRegionAddr() + offset;
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)rankGM);
        aclshmemx_mte_put_nbi(rankGMTensor, statusTensor_[(curExpertId - startExpertId) * UB_ALIGN_DATA_COUNT],
            UB_ALIGN_DATA_COUNT, dstRankId, 0U);
    }
    // reset操作前需确保前面操作完成
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void MoeDistributeDispatch::BufferInit()
{
    uint32_t waitStatusBufSize = Ceil((recStatusNumPerCore_ * UB_ALIGN), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * moeExpertNumPerRank_ * sizeof(float);
    uint64_t gatherMaskOutSize = (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    uint64_t sumContinueAlignSize = Ceil((aivNum_ * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);           // recStatusNumPerCore_32对齐后大小  * 32B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);                // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);               // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, sumContinueAlignSize);          // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);                       // 96 B
    uint32_t statusBufSize = rscvStatusNum_ * UB_ALIGN;
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t workLocalBufSize = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
}

__aicore__ inline void MoeDistributeDispatch::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次操作256字节，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8); // 8 = 256 / 32
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

__aicore__ inline void MoeDistributeDispatch::GatherSumRecvCnt(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2);  // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;               // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家 recvCnt 收集到gatherMaskOutTensor
    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
        {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) // 对inner要求32对齐
        * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // 把当前核的所有专家recv cnt之和写到状态区
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // 每个核把sumOfRecvCnt重复写 aivUsedCumSum_ 份
    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8); // 8 = 256 / 32
    // 每次处理256字节，8个datablock，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], sumCoreFP32Tensor, sumIntriParams);
}

__aicore__ inline void MoeDistributeDispatch::GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1), 0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, aivUsedCumSum_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();

    // 获取状态区中每个核的recvCnt
    while (true) {
        DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(aivUsedCumSum_), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));
        if (cumSumFlag == aivUsedCumSum_) {
            break;
        }
    }

    // 0核前面所有核recv cnt总和是0
    if (newAivId == 0) {
        outLocal.SetValue(0, 0);
    } else {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }
    // 清除 flag 用于下次aivUsedCumSum_软同步
    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();
    // 一次处理256字节，8个datablock
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);
    // 64 = 256 / sizeof(float) 一次操作字节数，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor, cleanParams);
}

__aicore__ inline void MoeDistributeDispatch::WaitDispatch()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1;
    gatherTmpTensor.SetValue(0, 1);
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    uint64_t performanceTimeStart = static_cast<uint64_t>(GetSystemCycle());
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)], intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    // 清状态
    WaitDispatchClearStatus();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

__aicore__ inline void MoeDistributeDispatch::CalRecvAndSetFlag()
{
    // check flag 用于 aivUsedCumSum_ 软同步并计算 aivUsedCumSum_ 个核各自的recvCount
    LocalTensor<int32_t> outCountLocal;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    GetCumSum(outCountLocal, newAivId);
    // 计算epRecvCnt
    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = statusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + 1);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR wAddr = (__gm__ uint8_t*)(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer((__gm__ int32_t*)wAddr);
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);
    // 复制aivNum_份
    for (uint32_t index = 0; index < aivNum_; index++) {
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
    }
    uint8_t repeatTimes = Ceil(aivNum_, 8);  // 一次处理256字节，8个datablock
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivNum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();
    // 每次处理256字节，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_FLAG_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor, sumIntriParams);  // 软同步
}

__aicore__ inline void MoeDistributeDispatch::SetExpertTokenNums()
{
    uint32_t localExpertNum = moeExpertNumPerRank_;
    DataCopyParams totalStatusCopyParams{static_cast<uint16_t>(localExpertNum * epWorldSize_), 1, 0, 0};
    LocalTensor<float> totalStatusTensorFp32 = statusBuf_.Get<float>();
    DataCopy(totalStatusTensorFp32, windowInstatusFp32Tensor_, totalStatusCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    int64_t expertTokenNumCumsum = 0;
    LocalTensor<int64_t> expertTokenNumsLocalTensor = tokenNumBuf_.Get<int64_t>();
    LocalTensor<float> expertTokenNumTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), 0);
    LocalTensor<float> workLocalTensor = workLocalBuf_.Get<float>();

    for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
        LocalTensor<float> expertStatusTensor = statusBuf_.GetWithOffset<float>(
            epWorldSize_ * UB_ALIGN / static_cast<uint32_t>(sizeof(float)), localExpertIdx * epWorldSize_ * UB_ALIGN);
        uint32_t mask = 2;
        SyncFunc<AscendC::HardEvent::S_V>();
        ReduceSum(expertTokenNumTensor, expertStatusTensor, workLocalTensor, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        int64_t expertTokenNum = static_cast<int64_t>(expertTokenNumTensor.ReinterpretCast<int32_t>().GetValue(0));
        expertTokenNumCumsum += expertTokenNum;
        if (expertTokenNumsType_ == 0) {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNumCumsum);
        } else {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNum);
        }
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)),
                                                0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor, expertTokenNumsCopyParams);
}

__aicore__ inline void MoeDistributeDispatch::CalCumSum()
{
    // 进来的核统一做发送，各专家的token总数发送
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);           // BS * K * 4
    tpipe_->InitBuffer(subExpBuf_, maxSize_);           // BS * K * 4
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);      // BS * K * 4
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    tpipe_->InitBuffer(statusBuf_, statusCntAlign_ * UB_ALIGN);
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    statusTensor_ = statusBuf_.Get<int32_t>();
    ExpIdsCopy();
    CalAndSendCnt();

    SplitToCore(rscvStatusNum_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);
    tpipe_->Reset();
    BufferInit();
    WaitDispatch();
    CalRecvAndSetFlag();
    // 使用newAivId为0的核进行计算
    if (aivId_ == aivUsedAllToAll_) {
        SetExpertTokenNums();
    }
}

__aicore__ inline void MoeDistributeDispatch::WaitCumSumFlag()
{
    // Check cumsum is finished
    int32_t cumSumFlag = 0;
    int32_t targetFlag = aivUsedCumSum_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (CUMSUM_FLAG_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = aivUsedCumSum_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    while (true) {
        DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    // Clean flag for next round
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
}

__aicore__ inline void MoeDistributeDispatch::SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum)
{
    // 获取cumSum
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    for (uint32_t index = startId_; index < endId_; index++) { // 从sendCnt中挑选当前有发送过来的卡的token数量
        expertMapTensor_(validNum) = index;
        if (index == 0) {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index);
        } else {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index) - sendCntTensor_(index - 1);
        }
        if (expertLeftNumTensor_(validNum) != 0) {
            validNum += 1;
        } 
    }
}

__aicore__ inline uint32_t MoeDistributeDispatch::CheckDataArriveWithFlag(uint32_t srcExpDataIdx,
    int32_t beginIdx, int32_t copyCnt)
{
    uint64_t rsvdCnt = 0;
    uint32_t arriveFlagNum = 0;
    uint32_t flagNum = blockCntPerToken_ * uint32_t(copyCnt);
    uint32_t compareCount = Ceil(flagNum, COMPARE_COUNT_PER_BLOCK) * COMPARE_COUNT_PER_BLOCK;
    uint32_t compResultU64Num = Ceil(flagNum, 64); // 64：按照64bit位进行划分
    DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(flagNum), static_cast<uint32_t>(sizeof(float)),
        static_cast<uint32_t>(SPLIT_BLOCK_SIZE - sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> expFlagPadParams{false, 0U, 0U, 0U};
    GlobalTensor<float> dataFlagGlobal;
    GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcExpDataIdx * expertPerSizeOnWin_ + // 拿到第一个起始位置
        beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE;
    dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
    DataCopyPad(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    GatherMask(flagGatherOutTensor_, flagRecvTensor_, flagRecvGatherMask_, true, uint32_t(1),
         {1, (uint16_t)(flagNum), 1, 0}, rsvdCnt); 
    PipeBarrier<PIPE_V>();
    CompareScalar(flagCompResultU8_, flagGatherOutTensor_, float(1), AscendC::CMPMODE::EQ, compareCount);
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t i = 0; i < compResultU64Num; i++) { 
        uint64_t flagCompMask = flagCompResultLtU64_(i);
        int64_t firstValidIdx = ScalarGetSFFValue<0>(flagCompMask); // 找到0则表示数据没到
        if (firstValidIdx == -1) { // 本次数据全到
            arriveFlagNum += 64U; // 64：ScalarGetSFFValue操作单位为64bit位
        } else {
            arriveFlagNum += uint32_t(firstValidIdx);
            break;
        }
    }
    if (arriveFlagNum > flagNum) {
        arriveFlagNum = flagNum;
    }
    return uint32_t(arriveFlagNum / blockCntPerToken_); // 返回token总数
}

__aicore__ inline void MoeDistributeDispatch::CopyInAndOut(
    LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount)
{
    // ===== TODO 3：完成量化接收流程（开发者补全本注释块内的代码） =====
    // 步骤 1：构造两个 GlobalTensor<ExpandXOutType>：
    //         dataFlagGlobal 指向 wAddr（对端 Win 数据区的源地址）；
    //         expandXOutGlobal 指向 expandXOutGM_ + dstPosition * axisH_（目标 token 行）。
    // 步骤 2：构造 4 组拷贝参数（注意单位与 stride）：
    //         a) DataCopyParams srcTokenCopyParams：
    //            blockCount = blockCntPerToken_ * arriveCount, blockLen = SPLIT_BLOCK_DATA_SIZE (480B),
    //            srcStride = UB_ALIGN (32B token-flag 跳过), dstStride = 0。
    //         b) DataCopyExtParams scalesCopyParams：
    //            blockCount = arriveCount, blockLen = scaleOutBytes_,
    //            srcStride = (blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN, dstStride = 0。
    //         c) DataCopyExtParams tokenCopyParams：
    //            blockCount = arriveCount, blockLen = hOutSize_,
    //            srcStride = (blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN, dstStride = 0。
    //         d) DataCopyExtParams expandIdxCopyParams：
    //            blockCount = arriveCount, blockLen = EXPAND_IDX_INFO * sizeof(int32_t)（三元组 12B）,
    //            srcStride = (blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1, dstStride = 0。
    //         e) DataCopyPadParams srcTokenPadParams{false, 0, 0, 0}（紧凑 480B 块无需 pad）。
    // 步骤 3：先把对端 Win 数据搬到 UB 临时缓冲 xTmpTensor_：
    //         DataCopyPad(xTmpTensor_, dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(ExpandXOutType)],
    //                     srcTokenCopyParams, srcTokenPadParams);
    //         SyncFunc<MTE2_MTE3>();
    // 步骤 4：把 scales 拆到 dynamicScalesOut：
    //         quantInst_.CopyScalesToOut(dstPosition, scaleOutBytes_, xTmpTensor_, scalesCopyParams);
    //         （内部已使用 Align256(axisH_) 作为源偏移。）
    // 步骤 5：把 FP8 主体拆到 expandXOut：
    //         DataCopyPad(expandXOutGlobal, xTmpTensor_, tokenCopyParams);
    // 步骤 6：把三元组拆到 assistInfoOut（即 expandIdxGMTensor_）：
    //         DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO],
    //                     xOutInt32Tensor[tokenQuantAlign_], expandIdxCopyParams);
    // ===== TODO 3 区结束 =====
}

__aicore__ inline void MoeDistributeDispatch::WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum)
{
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t maxCopyTokenCnt = tBufRealSize_ / hCommuSize_;
    uint32_t localExpertNum = moeExpertNumPerRank_;
    uint32_t srcExpRankId, dstPosition, arriveCount, copyCnt, srcDataBlockIdx;
    uint32_t flagMaxRecvNum = (blockCntPerToken_ * maxCopyTokenCnt * UB_ALIGN) / sizeof(uint32_t);
    uint32_t gatherOutSize = Ceil(blockCntPerToken_ * maxCopyTokenCnt * sizeof(uint32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    GlobalTensor<int32_t> cleanGlobal;
    flagGatherOutTensor_ = tBuf.GetWithOffset<float>(gatherOutSize / sizeof(float), 0); // buf复用
    flagRecvTensor_ = tBuf.GetWithOffset<float>(flagMaxRecvNum, gatherOutSize);  // buf复用
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    while (true) {
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum; // 轮询查询每个有效的index
            continue;
        }
        srcExpRankId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ? maxCopyTokenCnt : expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx = srcExpRankId % epWorldSize_ * localExpertNum + srcExpRankId / epWorldSize_; // 转换成数据区的排布偏移
        arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, expertFinishNumTensor_(index), copyCnt);
        if (arriveCount == copyCnt) {
            dstPosition = srcExpRankId != 0 ? sendCntTensor_(srcExpRankId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
            CopyInAndOut(xOutInt32Tensor, wAddr, index, dstPosition, arriveCount);
            // finish更新并clean
            expertFinishNumTensor_(index) += arriveCount;
            expertLeftNumTensor_(index) -= arriveCount;
            if (expertLeftNumTensor_(index) == 0) {
                uint32_t cleanUpNum = expertFinishNumTensor_(index) * blockCntPerToken_;
                DataCopyExtParams cleanUoParams = {uint16_t(cleanUpNum), sizeof(int32_t), 0U, SPLIT_BLOCK_SIZE - sizeof(int32_t), 0U};
                LocalTensor<int32_t> cleanTensor = tBuf.GetWithOffset<int32_t>(UB_ALIGN / sizeof(int32_t), 0); // 在0偏移位置存放比较结果
                cleanGlobal.SetGlobalBuffer((__gm__ int32_t *)(wAddr));
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                Duplicate<int32_t>(cleanTensor, 0, UB_ALIGN_DATA_COUNT);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopyPad(cleanGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(int32_t)], cleanTensor, cleanUoParams);
                finishNum++;
            }
            PipeBarrier<PIPE_ALL>();
        } else {
            index = (index + 1) % validNum;
        }
        if (validNum == finishNum) {
            break;
        }
    }
}

__aicore__ inline void MoeDistributeDispatch::LocalWindowCopy()
{
    // 分核负责源专家数量
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    // 软同步
    WaitCumSumFlag();
    if (sendNum_ == 0) {
        return;
    }
    // 连续化
    TBuf<> expertMapBuf, expertFinishBuf, expertLeftBuf, flagMaskBuf, tBuf;
    uint32_t validNum = 0;
    uint32_t expInfoSize = Ceil(sendNum_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);  // max CompareScalar
    tBufRealSize_ = MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) -
        (expInfoSize * 3) - BUFFER_NUM * UB_ALIGN; // 3为expInfoSize大小buffer申请个数
    tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    SetValidExpertInfo(expInfoSize, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
    flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
    flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    xTmpTensor_ = tBuf.Get<ExpandXOutType>();
    LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
    Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_S>();
    flagRecvGatherMask_.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::S_V>();
    WaitAndFormatOutput(tBuf, validNum);
}

__aicore__ inline void MoeDistributeDispatch::ExpIdsCopy()
{
    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    Duplicate<int32_t>(validExpertIdsTensor_, -1, int32_t(expertIdsBufSize_ / sizeof(int32_t)));

    uint32_t expertIdsMask = axisBS_ * axisK_;
    uint32_t expertIdsAlignCnt = Ceil(expertIdsMask, BITS_PER_BYTE) * BITS_PER_BYTE;
    uint32_t rightPadding = expertIdsAlignCnt - expertIdsMask;
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{true, 0U, uint8_t(rightPadding), -1}; // rightPadding字节数不能超过 32，不能超过8个u32
    DataCopyExtParams expertIdsCntParams{1U, static_cast<uint32_t>(expertIdsMask * sizeof(uint32_t)), 0U, 0U, 0U}; //第二个参数blockLen 范围[1, 2097151] 不能为0

    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

__aicore__ inline void MoeDistributeDispatch::Process()
{
    if ASCEND_IS_AIV {          // 全aiv处理
        if (aivId_ < aivUsedAllToAll_) {
            AllToAllDispatch(); // 前面核all2all发送
        } else {
            CalCumSum();        // 后面核发送当前卡给每个专家的tokenCnt，输出epRecvCnt/exportTokenNums
        }

        // localWindowCopy中包含reset操作，需确保前面操作完成
        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy();      // 本卡上专家数据连续化，输出expandX/scales/expandIdx
    }
}

} // DispatchImpl
#endif // MOE_DISTRIBUTE_DISPATCH_H
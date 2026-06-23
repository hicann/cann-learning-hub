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
 * \file moe_distribute_combine.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_H
#define MOE_DISTRIBUTE_COMBINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "shmem.h"
#include "moe_distribute_comm.h"

struct MoeDistributeCombineShmemTilingData {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    uint64_t totalWinSize;
};

namespace MoeDistributeCombineShmemImpl {
constexpr uint8_t BUFFER_NUM = 2;              // 多buf
constexpr uint32_t STATE_OFFSET = 32U;         // 状态空间偏移地址
constexpr uint32_t UB_ALIGN = 32U;             // UB按32字节对齐
constexpr uint32_t COMBINE_STATE_OFFSET =
    64U * 1024U;  // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint8_t EP_DOMAIN = 0;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 975UL * 1024UL;  // 预留48*512内存
constexpr uint32_t EXPAND_IDX_INFO =
    3U;  // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr uint32_t ALIGNED_LEN =
    256U;  // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr float SCALE_PARAM = 127.0;  // 计算量化参数所需的缩放倍数
constexpr uint32_t BLOCK_NUM =
    ALIGNED_LEN /
    UB_ALIGN;  // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t REDUCE_NUM = 8U;

__aicore__ inline int64_t GetShmemDataAddr(__gm__ uint8_t* shmemSpace, int32_t pe) {
    return (int64_t)aclshmem_ptr(shmemSpace, pe);
}

__aicore__ inline int64_t GetShmemSignalAddr(__gm__ uint8_t* shmemSpace, int32_t pe) {
    return (int64_t)aclshmem_ptr(shmemSpace, pe) + 1022 * 1024 * 1024;
}

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass \
    typename ExpandXType, typename XType, typename ExpandIdxType, bool IsInt8Quant

#define TemplateMC2TypeFunc ExpandXType, XType, ExpandIdxType, IsInt8Quant

using namespace Mc2Kernel;
using namespace AscendC;

template <TemplateMC2TypeClass>
class MoeDistributeCombineShmem {
public:
    __aicore__ inline MoeDistributeCombineShmem(){};
    __aicore__ inline void Init(GM_ADDR shmemSpace, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR XOut, TPipe* pipe,
                                const MoeDistributeCombineShmemTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitInputAndOutput(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
                                              GM_ADDR expertScales, GM_ADDR XOut);
    __aicore__ inline void InitAttrs(const MoeDistributeCombineShmemTilingData& tilingData);
    __aicore__ inline void InitInt8Quant();
    __aicore__ inline void Int8DequantProcess(LocalTensor<XType>& src);
    __aicore__ inline void AlltoAllBuffInit();
    __aicore__ inline void SetWaitStatusAndDisPatch();
    __aicore__ inline void ExpertAlltoAllDispatchInnerCopyAdd(uint32_t toRankId, uint32_t tokenId, uint32_t topkId,
                                                              uint32_t tkIndex, AscendC::TEventID copyEventId);
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void Int8QuantProcess();
    __aicore__ inline void ProcessMoeExpert(uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void WaitDispatch(uint32_t tokenIndex);

    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId) {
        return (GM_ADDR)GetShmemDataAddr(shmemContextGM_, rankId) + winDataSizeOffset_;
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId) {
        return (GM_ADDR)GetShmemSignalAddr(shmemContextGM_, rankId) + winStatusOffset_;
    }

    __aicore__ GM_ADDR GetShmemWinAddr(const int32_t rankId) {
        return (GM_ADDR)shmemContextGM_ + winDataSizeOffset_;
    }

    __aicore__ GM_ADDR GetShmemWinStateAddr(const int32_t rankId) {
        return (GM_ADDR)shmemContextGM_ + 1022 * 1024 * 1024 + winStatusOffset_;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y) {
        return (x < y) ? x : y;
    }

    TPipe* tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> rankWindow_;  // 用于存对端window的变量
    GlobalTensor<uint32_t> selfDataStatusTensor_;

    GM_ADDR shmemContextGM_;
    GM_ADDR statusDataSpaceGm_;

    uint64_t rank0_shmem_addr_{0};
    uint64_t pe_stride_{0};
    uint64_t shmem_size_{0};

    LocalTensor<ExpandXType> gmTpSendCountTensor_;
    LocalTensor<XType> outTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> gmTpSendCountFloatTensor_;
    LocalTensor<float> stateResetTensor_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t tpWorldSize_{0};
    uint32_t coreIdx_{0};              // aiv id
    uint32_t moeExpertPerRankNum_{0};  // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};           // moeExpertPerRankNum_ * epWorldSize_
    uint32_t moeExpertNum_{0};
    uint32_t bsKNum_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendCntNum_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t winStatusOffset_{0};
    uint64_t totalWinSize_{0};
    uint32_t selfSendCnt_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hAlign32Size_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t tokenScaleCnt_{0};
    uint32_t scaleNumAlignSize_{0};
    uint32_t flagRcvCount_{0};
    uint32_t opCnt_{0};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gmTpSendCountQueue_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> readStateBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> winTpSendCountFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> stateBuf_;
    TBuf<> stateResetBuf_;

    // int8量化
    TBuf<> xAbsBuf_;
    TBuf<> xMaxBuf_;
    TBuf<> xScaleMulBuf_;

    LocalTensor<int8_t> castLocalTensor_;
    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<XType> scaleDivTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<XType> sendLocalTensor_;
    LocalTensor<float> expertScalesLocal_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;

    uint32_t mask_{0};
    uint32_t repeatNum_{0};
    uint32_t scaleNum_{0};
    float scaleValFloat_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::InitInputAndOutput(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR XOut) {
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType*)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t*)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float*)expertScales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ XType*)XOut);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::InitAttrs(
    const MoeDistributeCombineShmemTilingData& tilingData) {
    axisBS_ = tilingData.bs;
    axisH_ = tilingData.h;
    axisK_ = tilingData.k;
    aivNum_ = tilingData.aivNum;
    epRankId_ = tilingData.epRankId;
    epWorldSize_ = tilingData.epWorldSize;
    moeExpertPerRankNum_ = tilingData.moeExpertPerRankNum;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;

    totalWinSize_ = tilingData.totalWinSize;
    moeExpertNum_ = tilingData.moeExpertNum;

    stateOffset_ = STATE_OFFSET;
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
    hAlignWinSize_ = Ceil(hExpandXTypeSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    bsKNum_ = axisBS_ * axisK_;
    hFloatAlign256Size_ = Ceil(hFloatSize, ALIGNED_LEN) * ALIGNED_LEN;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::InitInt8Quant() {
    scaleValFloat_ = static_cast<float>(1.0f / SCALE_PARAM);
    uint32_t scaleGranu = static_cast<uint32_t>(UB_ALIGN / sizeof(float));
    scaleNum_ = (hExpandXAlign32Size_ / sizeof(ExpandXType)) / scaleGranu;
    repeatNum_ = static_cast<uint32_t>(hFloatAlign256Size_ / ALIGNED_LEN);
    mask_ = static_cast<uint32_t>(ALIGNED_LEN / sizeof(float));
    tokenScaleCnt_ = hAlign32Size_ / sizeof(ExpandXType) + scaleNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::Init(
    GM_ADDR shmemSpace, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
    GM_ADDR expertScales, GM_ADDR XOut, TPipe* pipe,
    const MoeDistributeCombineShmemTilingData& tilingData) {
    tpipe_ = pipe;
    shmemContextGM_ = shmemSpace;

    coreIdx_ = GetBlockIdx();

    InitInputAndOutput(expandX, expertIds, expandIdx, epSendCount, expertScales, XOut);

    InitAttrs(tilingData);

    if constexpr (IsInt8Quant) {
        InitInt8Quant();
    }

    PipeBarrier<PIPE_ALL>();

    winDataSizeOffset_ = static_cast<uint64_t>(dataState_) * (totalWinSize_ / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET;
    DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        epSendCountGM_[moeSendNum_ - 1]);
    selfSendCnt_ = epSendCountGM_(moeSendNum_ - 1);

    SplitCoreCal();
    tpipe_->InitBuffer(gmTpSendCountQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    flagRcvCount_ = axisK_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::BuffInit() {
    tpipe_->Reset();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);

    tpipe_->InitBuffer(gmTpSendCountQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    if constexpr (IsInt8Quant) {
        uint32_t tokenScaleAlign32Size = Ceil(tokenScaleCnt_ * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM,
                           tokenScaleAlign32Size);  // 28K 输出token搬运
        tpipe_->InitBuffer(
            xAbsBuf_,
            hFloatAlign256Size_);  // 28K
                                   // blockReduceMax计算及后续Cast计算，256对齐
        uint32_t hFloatAlign256Cnt = hFloatAlign256Size_ / sizeof(float);
        tpipe_->InitBuffer(xMaxBuf_,
                           (hFloatAlign256Cnt / REDUCE_NUM) *
                               sizeof(float));  // 3.5K 存储ReduceMax结果
        tpipe_->InitBuffer(xScaleMulBuf_,
                           hFloatAlign256Size_);  // 28K 参与Brcb计算，256对齐
        tpipe_->InitBuffer(winTpSendCountFloatBuf_,
                           hFloatAlign32Size_);  // 28K 参与Div等token v核运算

        winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
        absFloatTensor_ = xAbsBuf_.Get<float>();
        reduceMaxFloatTensor_ = xMaxBuf_.Get<float>();
        scaleDupLocalTensor_ = xScaleMulBuf_.Get<float>();
        fp16CastTensor_ = xAbsBuf_.Get<half>();
        Duplicate(absFloatTensor_, float(0), hFloatAlign256Cnt);  // 统一写0
    }
    tpipe_->InitBuffer(indexCountsBuf_, sendCntNum_ * EXPAND_IDX_INFO * sizeof(int32_t));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::AlltoAllBuffInit() {
    tpipe_->Reset();
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size_;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size_;
    tpipe_->InitBuffer(expertScalesBuf_, axisBS_ * axisK_ * sizeof(float));
    tpipe_->InitBuffer(tokenBuf_, maxSizeRowTmpFloatBuf);
    tpipe_->InitBuffer(rowTmpFloatBuf_, maxSizeRowTmpFloatBuf);
    tpipe_->InitBuffer(mulBuf_, hFloatAlign32Size_);
    tpipe_->InitBuffer(sumFloatBuf_, hFloatAlign32Size_);
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    tpipe_->InitBuffer(stateBuf_, (flagRcvCount_) * STATE_OFFSET);
    tpipe_->InitBuffer(stateResetBuf_, (flagRcvCount_) * STATE_OFFSET);
    LocalTensor<float> stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(flagRcvCount_ * FLOAT_PER_UB_ALIGN));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    if constexpr (IsInt8Quant) {
        scaleNumAlignSize_ = Ceil(scaleNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xAbsBuf_, scaleNumAlignSize_);
        fp16CastTensor_ = mulBuf_.Get<half>();
        absFloatTensor_ = rowTmpFloatBuf_.Get<float>();
        scaleDupLocalTensor_ = mulBuf_.Get<float>();
        scaleDivFloatTensor_ = xAbsBuf_.Get<float>();
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::SplitCoreCal() {
    sendCntNum_ = selfSendCnt_ / aivNum_;
    uint32_t remainderRankNum = selfSendCnt_ % aivNum_;

    startTokenId_ = sendCntNum_ * coreIdx_;

    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
    endTokenId_ = startTokenId_ + sendCntNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::SetWaitStatusAndDisPatch() {
    PipeBarrier<PIPE_ALL>();
    if (coreIdx_ >= selfSendCnt_) {
        return;
    }
    ExpertAlltoAllDispatchCopyAdd();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::ExpertAlltoAllDispatchCopyAdd() {
    if (sendCntNum_ == 0U) {
        return;
    }

    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams{1U,
                                      static_cast<uint32_t>(sendCntNum_ * EXPAND_IDX_INFO * sizeof(uint32_t)), 0U, 0U,
                                      0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId_ * EXPAND_IDX_INFO], bskParams, copyPadParams);
    LocalTensor<float> statusTensor = readStateBuf_.AllocTensor<float>();
    Duplicate<float>(statusTensor, (float)1, FLOAT_PER_UB_ALIGN);

    __gm__ aclshmem_device_host_state_t* deviceState = aclshmemi_get_state();
    AscendC::TEventID copyEventId = (AscendC::TEventID)deviceState->mte_config.sync_id;

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t loop = 0; loop < sendCntNum_; loop++) {
        uint32_t tkIndex = startTokenId_ + ((loop + epRankId_) % sendCntNum_);
        uint32_t baseOffset = (tkIndex - startTokenId_) * EXPAND_IDX_INFO;
        uint32_t rankIdExpandIdx = static_cast<uint32_t>(expandIdxLocal(baseOffset));
        uint32_t toRankId = rankIdExpandIdx;
        uint32_t tokenId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 1));
        uint32_t topkId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 2));
        ExpertAlltoAllDispatchInnerCopyAdd(toRankId, tokenId, topkId, tkIndex, copyEventId);
        PipeBarrier<PIPE_MTE3>();
        GM_ADDR stateGM =
            GetShmemWinStateAddr(toRankId) + tokenId * flagRcvCount_ * stateOffset_ + topkId * stateOffset_;
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
        aclshmemx_mte_put_nbi<float>(stateGMTensor, statusTensor, FLOAT_PER_UB_ALIGN, toRankId, copyEventId);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::Int8QuantProcess() {
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    castLocalTensor_ = sendLocalTensor_.template ReinterpretCast<int8_t>();
    scaleDivTensor_ = castLocalTensor_[hAlign32Size_].template ReinterpretCast<XType>();

    Cast(winTpSendCountFloatTensor_, gmTpSendCountTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    Abs(absFloatTensor_, winTpSendCountFloatTensor_, axisH_);
    PipeBarrier<PIPE_V>();

    BlockReduceMax(reduceMaxFloatTensor_, absFloatTensor_, repeatNum_, mask_, 1, 1, BLOCK_NUM);
    PipeBarrier<PIPE_V>();

    Muls(reduceMaxFloatTensor_, reduceMaxFloatTensor_, scaleValFloat_, scaleNum_);

    PipeBarrier<PIPE_V>();

    Cast(scaleDivTensor_, reduceMaxFloatTensor_, RoundMode::CAST_RINT, scaleNum_);
    PipeBarrier<PIPE_V>();

    Brcb(scaleDupLocalTensor_, reduceMaxFloatTensor_, repeatNum_, {1, BLOCK_NUM});
    PipeBarrier<PIPE_V>();

    Div(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, scaleDupLocalTensor_, axisH_);

    PipeBarrier<PIPE_V>();

    Cast(fp16CastTensor_, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();

    Cast(castLocalTensor_, fp16CastTensor_, RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::ExpertAlltoAllDispatchInnerCopyAdd(
    uint32_t toRankId, uint32_t tokenId, uint32_t topkId, uint32_t tkIndex, AscendC::TEventID copyEventId) {
    uint32_t epOffset = tokenId * (axisK_) + topkId;
    uint32_t tokenGMOffset = tkIndex * axisH_;
    uint32_t tokenWinOffset = tkIndex * hAlignWinCnt_;
    GM_ADDR rankGM = GetWinAddrByRankId(toRankId) + epOffset * hAlignWinSize_;
    rankWindow_.SetGlobalBuffer((__gm__ XType*)rankGM);
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyExtParams xScaleCopyParams{1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U,
                                       0U};
    if constexpr (IsInt8Quant) {
        gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
        gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
        gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
        sendLocalTensor_ = xOutQueue_.AllocTensor<ExpandXType>();
        Int8QuantProcess();
        xOutQueue_.EnQue(sendLocalTensor_);
        sendLocalTensor_ = xOutQueue_.DeQue<ExpandXType>();
        DataCopyPad(rankWindow_, sendLocalTensor_, xScaleCopyParams);
        gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
        xOutQueue_.FreeTensor<ExpandXType>(sendLocalTensor_);
    } else {
        gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
        gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
        gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
        DataCopyPad(rankWindow_, gmTpSendCountTensor_, expandXCopyParams);
        gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::Int8DequantProcess(LocalTensor<XType>& src) {
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    castLocalTensor_ = src.template ReinterpretCast<int8_t>();
    scaleDivTensor_ = src[hAlign32Size_ / 2].template ReinterpretCast<XType>();

    SyncFunc<AscendC::HardEvent::S_V>();
    Cast(scaleDivFloatTensor_, scaleDivTensor_, RoundMode::CAST_NONE, scaleNum_);
    Cast(fp16CastTensor_, castLocalTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    Cast(absFloatTensor_, fp16CastTensor_, RoundMode::CAST_NONE, axisH_);
    Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_, repeatNum_, {1, BLOCK_NUM});

    PipeBarrier<PIPE_V>();

    Mul(absFloatTensor_, absFloatTensor_, scaleDupLocalTensor_, axisH_);
    PipeBarrier<PIPE_V>();

    Cast(src, absFloatTensor_, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::WaitDispatch(uint32_t tokenIndex) {
    uint32_t copyCount = flagRcvCount_ * FLOAT_PER_UB_ALIGN;
    uint32_t targetCount = copyCount;
    GM_ADDR stateGM = GetWinStateAddrByRankId(epRankId_) + tokenIndex * flagRcvCount_ * stateOffset_;
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
    float localState = 0;
    float target = (float)1.0 * targetCount;
    float minTarget = target - (float)0.5;
    float maxTarget = target + (float)0.5;
    SumParams sumParams{1, copyCount, copyCount};
    LocalTensor<float> stateTensor = stateBuf_.Get<float>();
    while ((localState < minTarget) || (localState > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(stateTensor, stateGMTensor, copyCount);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(stateTensor, stateTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        localState = stateTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(stateGMTensor, stateResetTensor_, copyCount);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::ProcessMoeExpert(uint32_t tokenIndexOffset,
                                                                                          uint32_t topkId,
                                                                                          float scaleVal) {
    uint32_t processLen = axisH_;
    const DataCopyExtParams xScaleCopyParams{1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U,
                                             0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};

    GM_ADDR wAddr = (__gm__ uint8_t*)(GetShmemWinAddr(epRankId_)) + (tokenIndexOffset + topkId) * hAlignWinSize_;
    rankWindow_.SetGlobalBuffer((__gm__ XType*)wAddr);
    LocalTensor<XType> tmpUb = moeSumQueue_.AllocTensor<XType>();
    if constexpr (IsInt8Quant) {
        DataCopyPad(tmpUb, rankWindow_, xScaleCopyParams, copyPadExtParams);
    } else {
        DataCopyPad(tmpUb, rankWindow_, expandXCopyParams, copyPadExtParams);
    }
    moeSumQueue_.EnQue(tmpUb);
    tmpUb = moeSumQueue_.DeQue<XType>();
    if constexpr (IsInt8Quant) {
        Int8DequantProcess(tmpUb);
    }
    Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, processLen);
    moeSumQueue_.FreeTensor<XType>(tmpUb);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::LocalWindowCopy() {
    if (axisBS_ == 0U) {
        return;
    }
    uint32_t beginIndex = 0U;
    uint32_t endIndex = 0U;
    uint32_t processLen = 0U;
    uint32_t tokenOffset = 0U;
    uint32_t tokenPerAivNum = axisBS_ / aivNum_;
    uint32_t remainderToken = axisBS_ % aivNum_;

    beginIndex = tokenPerAivNum * coreIdx_;
    if (coreIdx_ < remainderToken) {
        tokenPerAivNum++;
        beginIndex += coreIdx_;
    } else {
        beginIndex += remainderToken;
    }
    endIndex = beginIndex + tokenPerAivNum;
    if (tokenPerAivNum == 0U) {
        return;
    }
    processLen = axisH_;
    expertScalesLocal_ = expertScalesBuf_.Get<float>();
    rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    sumFloatBufLocal_ = sumFloatBuf_.Get<float>();

    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(bsKNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyPad(expertScalesLocal_, expertScalesGM_, bskParams, copyPadFloatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t curIdx = beginIndex; curIdx < endIndex; curIdx++) {
        uint32_t tokenIndex = curIdx;
        WaitDispatch(tokenIndex);
        uint32_t index = tokenIndex * axisK_;
        float scaleVal = 0.0;
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(sumFloatBufLocal_, static_cast<float>(0), axisH_);
        uint32_t tokenIndexOffset = tokenIndex * axisK_;
        for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
            uint32_t expert_id = expertIdsGM_.GetValue(tokenIndex * axisK_ + topkId);
            scaleVal = expertScalesLocal_.GetValue(index);

            if (expert_id < moeExpertNum_) {
                ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal);
                index++;
            }
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<XType> sumBufLocal = tokenBuf_.Get<XType>();
        Cast(sumBufLocal, sumFloatBufLocal_, AscendC::RoundMode::CAST_RINT, processLen);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(expandOutGlobal_[tokenIndex * axisH_ + tokenOffset], sumBufLocal, expandXCopyParams);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineShmem<TemplateMC2TypeFunc>::Process() {
    if ASCEND_IS_AIV {
        BuffInit();
        SetWaitStatusAndDisPatch();
        PipeBarrier<PIPE_ALL>(); // AlltoAllBuffInitAndMaskCal中包含reset操作，需确保前面操作完成
        AlltoAllBuffInit();
        LocalWindowCopy();
    }
}

}  // namespace MoeDistributeCombineShmemImpl
#endif  // MOE_DISTRIBUTE_COMBINE_IMPL_H
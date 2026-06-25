#ifndef MTE_COMM_H
#define MTE_COMM_H

#include "utils.h"
#include "kernel_operator.h"

namespace MatmulAllGatherImpl {

using namespace AscendC;

constexpr static uint32_t UB_ALIGN_BYTES = 32U;
constexpr static uint32_t FLOAT_ALIGN_NUM = 8U;
constexpr static uint32_t BUFFER_NUM = 2U;
constexpr static uint64_t STATE_WIN_SIZE = 1024UL * 1024UL;
constexpr static uint64_t SINGLE_STATE_REGION_SIZE = 462UL * 1024UL;
constexpr static uint32_t ZERONE_STATE_POS = 0U;
constexpr static uint64_t WIN_ADDR_ALIGN = 512UL;

template<typename DataType>
class MTECommunication {
public:
    __aicore__ inline MTECommunication() {};
    __aicore__ inline void InitHcclContextByAddr(GM_ADDR mc2Context);
    __aicore__ inline void InitGMTensor(GM_ADDR y, uint64_t winSpaceGmSize);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void CopyDataFromWin(uint64_t perRankM, uint64_t nDim);
    __aicore__ inline GM_ADDR GetWinAddrGm(uint32_t rankId, uint64_t offset = 0);
    __aicore__ inline GM_ADDR GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag);
    __aicore__ inline GM_ADDR GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag);

    __gm__ HcclA5OpResParam *hcclContext_;
    uint32_t aivId_{0};
    uint32_t winBufferFlags_{0};
    uint64_t totalLen_{0};
    GlobalTensor<DataType> inputTensor_;
    GlobalTensor<DataType> outputTensor_;
    GlobalTensor<DataType> localWinOutTensor_;
    GM_ADDR localWinDataAddr_;

private:
    __aicore__ inline void CopyDataBlockFromWin(uint32_t remoteRankId, uint64_t readOffset, uint64_t curOutOffset, uint32_t count);

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> copyQueue_;
    TBuf<> winFlagsBuf_;
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> stateResetBuf_;

    GlobalTensor<uint32_t> selfWinFlagGMTensor_;
    LocalTensor<DataType> copyTmpTensor_;
    LocalTensor<float> stateResetTensor_;
};

template<typename DataType>
__aicore__ inline void MTECommunication<DataType>::InitHcclContextByAddr(GM_ADDR mc2Context)
{
    hcclContext_ = (__gm__ HcclA5OpResParam*)(mc2Context);
    aivId_ = GetBlockIdx();
}

template<typename DataType>
__aicore__ inline void MTECommunication<DataType>::InitGMTensor(GM_ADDR y, uint64_t winSpaceGmSize)
{
    outputTensor_.SetGlobalBuffer((__gm__ DataType*)y);

    uint64_t currCoreFlagOffset = 2UL * SINGLE_STATE_REGION_SIZE + aivId_ * WIN_ADDR_ALIGN;
    selfWinFlagGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)GetWinAddrGm(hcclContext_->rankId, currCoreFlagOffset));
    LocalTensor<uint32_t> winFlagLocalTensor = winFlagsBuf_.Get<uint32_t>();
    DataCopyExtParams flagCopyInParams{1, static_cast<uint32_t>(UB_ALIGN_BYTES), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> flagPadParams{false, 0, 0, 0};
    DataCopyPad(winFlagLocalTensor, selfWinFlagGMTensor_, flagCopyInParams, flagPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    winBufferFlags_ = winFlagLocalTensor.GetValue(ZERONE_STATE_POS);
    winFlagLocalTensor.SetValue(ZERONE_STATE_POS, 1 - winBufferFlags_);

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams flagCopyOutParams{1, static_cast<uint32_t>(UB_ALIGN_BYTES), 0, 0, 0};
    DataCopyPad(selfWinFlagGMTensor_, winFlagLocalTensor, flagCopyOutParams);

    GM_ADDR localWinDataGm = GetWinDataAddrGm(hcclContext_->rankId, winBufferFlags_);
    localWinDataAddr_ = localWinDataGm;
    localWinOutTensor_.SetGlobalBuffer((__gm__ DataType*)localWinDataGm);
}

template<typename DataType>
__aicore__ inline void MTECommunication<DataType>::InitBuffer(TPipe *tPipe)
{
    constexpr uint32_t COPY_BLOCK_SIZE = 1024U;
    tPipe->InitBuffer(copyQueue_, BUFFER_NUM, COPY_BLOCK_SIZE * sizeof(DataType));
    tPipe->InitBuffer(winFlagsBuf_, UB_ALIGN_BYTES);
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES);
    tPipe->InitBuffer(readStateBuf_, hcclContext_->rankDim * UB_ALIGN_BYTES);
    tPipe->InitBuffer(stateResetBuf_, hcclContext_->rankDim * UB_ALIGN_BYTES);

    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(hcclContext_->rankDim * FLOAT_ALIGN_NUM));
}

template<typename DataType>
__aicore__ inline void MTECommunication<DataType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * hcclContext_->rankDim;
    for (uint32_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
        LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
        DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_ALIGN_NUM);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        statusTensor.SetValue(0, (float)1.0);
        GM_ADDR remoteWinStateGM = GetWinStatusAddrGm(curRank, winBufferFlags_);
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
        uint64_t curOffset = (coreOffset + hcclContext_->rankId) * FLOAT_ALIGN_NUM;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams stateCopyOutParams{1, static_cast<uint32_t>(FLOAT_ALIGN_NUM * sizeof(float)), 0, 0, 0};
        DataCopyPad(stateGMTensor[curOffset], statusTensor, stateCopyOutParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template<typename DataType>
__aicore__ inline void MTECommunication<DataType>::ReadStatus()
{
    GM_ADDR stateGM = GetWinStatusAddrGm(hcclContext_->rankId, winBufferFlags_);
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * hcclContext_->rankDim * FLOAT_ALIGN_NUM;
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0;
    uint32_t statusCnt = hcclContext_->rankDim * FLOAT_ALIGN_NUM;
    SumParams sumParams{1, statusCnt, statusCnt};
    float minTarget = hcclContext_->rankDim - (float)0.5;
    float maxTarget = hcclContext_->rankDim + (float)0.5;
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopyExtParams statusCopyInParams{1, static_cast<uint32_t>(statusCnt * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> statusPadParams{false, 0, 0, 0};
        DataCopyPad(statusTensor, selfStatusWinTensor[offset], statusCopyInParams, statusPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusTensor, statusTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        flag = statusTensor.GetValue(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams statusCopyOutParams{1, static_cast<uint32_t>(statusCnt * sizeof(float)), 0, 0, 0};
    DataCopyPad(selfStatusWinTensor[offset], stateResetTensor_, statusCopyOutParams);
}

template<typename DataType>
__aicore__ inline void MTECommunication<DataType>::CopyDataBlockFromWin(uint32_t remoteRankId, uint64_t readOffset, uint64_t curOutOffset, uint32_t count)
{
    GM_ADDR remoteWinDataGm = GetWinDataAddrGm(remoteRankId, winBufferFlags_);
    GlobalTensor<DataType> remoteWinXTensor;
    remoteWinXTensor.SetGlobalBuffer((__gm__ DataType*)remoteWinDataGm);

    copyTmpTensor_ = copyQueue_.AllocTensor<DataType>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(count * sizeof(DataType)), 0, 0, 0};
    DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
    DataCopyPad(copyTmpTensor_, remoteWinXTensor[readOffset], copyInParams, padParams);
    copyQueue_.EnQue(copyTmpTensor_);
    copyTmpTensor_ = copyQueue_.DeQue<DataType>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(count * sizeof(DataType)), 0, 0, 0};
    DataCopyPad(outputTensor_[curOutOffset], copyTmpTensor_, copyOutParams);
    copyQueue_.FreeTensor(copyTmpTensor_);
}

template<typename DataType>
__aicore__ inline void MTECommunication<DataType>::CopyDataFromWin(uint64_t perRankM, uint64_t nDim)
{
    constexpr uint32_t COPY_BLOCK_SIZE = 1024U;
    uint64_t perRankOutSize = perRankM * nDim;
    uint64_t totalOutSize = perRankOutSize * hcclContext_->rankDim;
    
    for (uint32_t i = 0; i < hcclContext_->rankDim; ++i) {
        uint32_t remoteRankId = (hcclContext_->rankId + i) % hcclContext_->rankDim;
        uint64_t outBaseOffset = remoteRankId * perRankOutSize;

        uint64_t remainSize = perRankOutSize;
        uint64_t curOffset = 0;
        while (remainSize > 0) {
            uint32_t copySize = static_cast<uint32_t>(Min(remainSize, COPY_BLOCK_SIZE));
            CopyDataBlockFromWin(remoteRankId, curOffset, outBaseOffset + curOffset, copySize);
            curOffset += copySize;
            remainSize -= copySize;
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template<typename DataType>
__aicore__ inline GM_ADDR MTECommunication<DataType>::GetWinAddrGm(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(hcclContext_->windowsIn[rankId] + offset);
}

template<typename DataType>
__aicore__ inline GM_ADDR MTECommunication<DataType>::GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag)
{
    if (winFlag == 0U) {
        return GetWinAddrGm(rankId, STATE_WIN_SIZE);
    } else {
        return (GM_ADDR)(hcclContext_->windowsOut[rankId]);
    }
}

template<typename DataType>
__aicore__ inline GM_ADDR MTECommunication<DataType>::GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag)
{
    if (winFlag == 0U) {
        return GetWinAddrGm(rankId);
    } else {
        return GetWinAddrGm(rankId, SINGLE_STATE_REGION_SIZE);
    }
}

}

#endif
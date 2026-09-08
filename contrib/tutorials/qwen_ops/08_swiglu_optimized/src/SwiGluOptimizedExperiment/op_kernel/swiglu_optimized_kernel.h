#ifndef SWIGLU_OPTIMIZED_KERNEL_H
#define SWIGLU_OPTIMIZED_KERNEL_H

#include "kernel_operator.h"
#include "swiglu_tiling.h"

using namespace AscendC;

class SwiGluOptimizedKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR gate, GM_ADDR up, GM_ADDR y, const SwiGluTilingData &tiling)
    {
        totalSize_ = tiling.totalSize;
        elementsPerCore_ = tiling.elementsPerCore;
        coreNum_ = tiling.coreNum;
        tileLength_ = tiling.tileLength;

        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gate), totalSize_);
        upGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(up), totalSize_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), totalSize_);

        const uint32_t tileBytes = tileLength_ * sizeof(float);
        pipe_.InitBuffer(gateQueue_, 1, tileBytes);
        pipe_.InitBuffer(upQueue_, 1, tileBytes);
        pipe_.InitBuffer(yQueue_, 1, tileBytes);
        pipe_.InitBuffer(workBuf_, 6U * tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        if (coreId >= coreNum_) {
            return;
        }

        const uint32_t start = coreId * elementsPerCore_;
        uint32_t end = start + elementsPerCore_;
        if (end > totalSize_) {
            end = totalSize_;
        }

        for (uint32_t offset = start; offset < end; offset += tileLength_) {
            uint32_t curLength = tileLength_;
            if (offset + curLength > end) {
                curLength = end - offset;
            }
            ProcessTile(offset, curLength);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t curLength)
    {
        LocalTensor<float> gateLocal = gateQueue_.AllocTensor<float>();
        LocalTensor<float> upLocal = upQueue_.AllocTensor<float>();
        LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();

        DataCopy(gateLocal, gateGm_[offset], curLength);
        DataCopy(upLocal, upGm_[offset], curLength);
        gateQueue_.EnQue(gateLocal);
        upQueue_.EnQue(upLocal);

        gateLocal = gateQueue_.DeQue<float>();
        upLocal = upQueue_.DeQue<float>();

        LocalTensor<float> workLocal = workBuf_.Get<float>();
        LocalTensor<float> negGateLocal = workLocal;
        LocalTensor<float> expLocal = workLocal[tileLength_];
        LocalTensor<float> denomLocal = workLocal[2U * tileLength_];
        LocalTensor<float> oneLocal = workLocal[3U * tileLength_];
        LocalTensor<float> sigmoidLocal = workLocal[4U * tileLength_];
        LocalTensor<float> siluLocal = workLocal[5U * tileLength_];

        Muls(negGateLocal, gateLocal, -1.0f, curLength);
        PipeBarrier<PIPE_V>();
        Exp(expLocal, negGateLocal, curLength);
        PipeBarrier<PIPE_V>();
        Adds(denomLocal, expLocal, 1.0f, curLength);
        PipeBarrier<PIPE_V>();
        Duplicate(oneLocal, 1.0f, curLength);
        PipeBarrier<PIPE_V>();
        Div(sigmoidLocal, oneLocal, denomLocal, curLength);
        PipeBarrier<PIPE_V>();
        Mul(siluLocal, gateLocal, sigmoidLocal, curLength);
        PipeBarrier<PIPE_V>();
        Mul(yLocal, siluLocal, upLocal, curLength);
        PipeBarrier<PIPE_V>();

        gateQueue_.FreeTensor(gateLocal);
        upQueue_.FreeTensor(upLocal);

        yQueue_.EnQue(yLocal);
        yLocal = yQueue_.DeQue<float>();
        DataCopy(yGm_[offset], yLocal, curLength);
        yQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> gateQueue_;
    TQue<TPosition::VECIN, 1> upQueue_;
    TQue<TPosition::VECOUT, 1> yQueue_;
    TBuf<TPosition::VECCALC> workBuf_;
    GlobalTensor<float> gateGm_;
    GlobalTensor<float> upGm_;
    GlobalTensor<float> yGm_;
    uint32_t totalSize_ = 0;
    uint32_t elementsPerCore_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t tileLength_ = 1024;
};

#endif
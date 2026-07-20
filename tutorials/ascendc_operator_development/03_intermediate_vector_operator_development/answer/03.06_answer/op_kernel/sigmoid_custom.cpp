#include "kernel_operator.h"
#include "sigmoid_custom_tiling.h"
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t QUEUE_DEPTH = 2;

template<typename TYPE_X, typename TYPE_Y>
class KernelSigmoid {
public:
    __aicore__ inline KernelSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;

        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.AllocTensor<TYPE_Y>();

        AscendC::Sigmoid(zLocal, xLocal, this->processDataNum);
        outQueueZ.EnQue<TYPE_Y>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.DeQue<TYPE_Y>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX;          // 单输入队列
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;        // 单输出队列
    AscendC::GlobalTensor<TYPE_X> xGm;                                        // 输入全局Tensor
    AscendC::GlobalTensor<TYPE_Y> zGm;                                        // 输出全局Tensor
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tilingData.smallCoreDataNum, 
            tilingData.bigCoreDataNum, tilingData.finalBigTileNum, 
            tilingData.finalSmallTileNum, tilingData.tileDataNum, 
            tilingData.smallTailDataNum, tilingData.bigTailDataNum, 
            tilingData.tailBlockNum);
    op.Process();
}
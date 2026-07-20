#include "kernel_operator.h"
#include "square_diff_tiling.h"
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t QUEUE_DEPTH = 2;

template<typename TYPE_X, typename TYPE_Y, typename TYPE_Z> class KernelSquareDiff {
public:
    __aicore__ inline KernelSquareDiff() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t smallCoreDataNum,
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
        }
        else { 
          this->coreDataNum = smallCoreDataNum;
          this->tileNum = finalSmallTileNum;
          this->tailDataNum = smallTailDataNum;
          globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Z));
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
      AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
      AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
      AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
      inQueueX.EnQue(xLocal);
      inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
      AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
      AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();

      AscendC::Mul(xLocal, xLocal, xLocal, this->processDataNum);
      AscendC::Mul(yLocal, yLocal, yLocal, this->processDataNum);
      AscendC::Sub(zLocal, xLocal, yLocal, this->processDataNum);
      outQueueZ.EnQue<TYPE_Z>(zLocal);
      inQueueX.FreeTensor(xLocal);
      inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();  
      AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
      outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<TYPE_Z> zGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void square_diff(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SquareDiffTilingData);
    GET_TILING_DATA(tiling_data, tiling);
    KernelSquareDiff<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.smallCoreDataNum, 
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
            tiling_data.tailBlockNum);
    op.Process();
}
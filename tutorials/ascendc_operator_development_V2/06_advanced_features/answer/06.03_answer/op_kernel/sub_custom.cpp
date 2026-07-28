#include "kernel_operator.h"
#include "sub_custom_tiling.h"
#include "kernel_operator_dump_tensor_intf_impl.h"


template <class dtypeX, class dtypeY, class dtypeZ>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum;
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, 1, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, 1, this->tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        AscendC::printf("Core %d executed %d times in total\n",  AscendC::GetBlockIdx(), loopCount);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Sub(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


 __global__ __aicore__ void sub_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
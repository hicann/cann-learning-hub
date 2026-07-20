#include "kernel_operator.h"
#include "add_custom_template_tiling.h"
#include "tiling_key_add_custom_template.h"
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t QUEUE_DEPTH = 1;

template <class dtypeX>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum, float max, float min)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->max = max;
        this->min = min;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeX *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeX *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(tmp1, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmp2, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        AscendC::printf("Dtype float32, Core %d executed %d times in total\n",  AscendC::GetBlockIdx(), loopCount);
    }

    __aicore__ inline void Process2()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute2(i);
            CopyOut(i);
        }
        AscendC::printf("Dtype int8, Core %d executed %d times in total\n",  AscendC::GetBlockIdx(), loopCount);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeX> yLocal = inQueueY.AllocTensor<dtypeX>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeX> yLocal = inQueueY.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeX> zLocal = outQueueZ.AllocTensor<dtypeX>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        AscendC::Mins(zLocal, zLocal, this->max, this->tileLength);
        AscendC::Maxs(zLocal, zLocal, this->min, this->tileLength);
        outQueueZ.EnQue<dtypeX>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute2(int32_t progress)
    {
        auto p1 = tmp1.Get<half>();
        auto p2 = tmp2.Get<half>();
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeX> yLocal = inQueueY.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeX> zLocal = outQueueZ.AllocTensor<dtypeX>();

        AscendC::Cast(p1, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
        AscendC::Cast(p2, yLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
        AscendC::Add(p2, p1, p2, this->tileLength);
        AscendC::Mins(p2, p2, (half)this->max, this->tileLength);
        AscendC::Maxs(p2, p2, (half)this->min, this->tileLength);
        AscendC::Cast(zLocal, p2, AscendC::RoundMode::CAST_RINT, this->tileLength);

        outQueueZ.EnQue<dtypeX>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> zLocal = outQueueZ.DeQue<dtypeX>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeX> yGm;
    AscendC::GlobalTensor<dtypeX> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float max, min;
};


template <typename D_T_X>
__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd<D_T_X> op;
    op.Init(x, y , z,  tilingData.totalLength, tilingData.tileNum,tilingData.max, tilingData.min);
    if constexpr (std::is_same_v<D_T_X, float>) {
        op.Process();
    } else {
        op.Process2();
    }
}
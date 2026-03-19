
#ifndef SUB_CUSTOM_H
#define SUB_CUSTOM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sub_custom_tiling_data.h"
#include "sub_custom_tiling_key.h"

namespace NsSub {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class Sub {
public:
    __aicore__ inline Sub(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t tileNum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename T>
__aicore__ inline void Sub<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
{
    this->blockLength = totalLength / AscendC::GetBlockNum();
    this->tileNum = tileNum;
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
    xGm.SetGlobalBuffer((__gm__ T *)x1 + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    x2Gm.SetGlobalBuffer((__gm__ T *)x2 + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    yGm.SetGlobalBuffer((__gm__ T *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
}

template <typename T>
__aicore__ inline void Sub<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    AscendC::DataCopy(x2Local, x2Gm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueX2.EnQue(x2Local);
}

template <typename T>
__aicore__ inline void Sub<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sub<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    AscendC::Sub(yLocal, xLocal, x2Local, this->tileLength);
    outQueueY.EnQue<T>(yLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueX2.FreeTensor(x2Local);
}

template <typename T>
__aicore__ inline void Sub<T>::Process()
{
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsSub
#endif // SUB_CUSTOM_H

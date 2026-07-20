#ifndef __SIGMOID_H__
#define __SIGMOID_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t QUEUE_DEPTH = 1;

template <typename T>
class Sigmoid {
public:
    __aicore__ inline Sigmoid(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum);
    __aicore__ inline void Process();
    __aicore__ inline void Process2();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void Compute2(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> XXX;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> YYY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename T>
__aicore__ inline void Sigmoid<T>::Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
{
    this->blockLength = totalLength / AscendC::GetBlockNum();
    this->tileNum = tileNum;
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
    xGm.SetGlobalBuffer((__gm__ T *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    yGm.SetGlobalBuffer((__gm__ T *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    pipe.InitBuffer(XXX, BUFFER_NUM, this->tileLength * sizeof(T));
    pipe.InitBuffer(YYY, BUFFER_NUM, this->tileLength * sizeof(T));
    pipe.InitBuffer(tmp1, this->tileLength * sizeof(float));
    pipe.InitBuffer(tmp2, this->tileLength * sizeof(float));
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = XXX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    XXX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> yLocal = YYY.DeQue<T>();
    AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
    YYY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = XXX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = YYY.AllocTensor<T>();
    // sigmoid(x) = 1/(1 + exp(-x))
    AscendC::Muls(yLocal, xLocal, (T)-1, this->tileLength);
    AscendC::Exp(yLocal, yLocal, this->tileLength);
    AscendC::Adds(yLocal, yLocal, (T)1, this->tileLength);
    AscendC::Div(xLocal, xLocal, xLocal, this->tileLength);
    AscendC::Div(yLocal, xLocal, yLocal, this->tileLength);
    YYY.EnQue(yLocal);
    XXX.FreeTensor(xLocal);

}

template <typename T>
__aicore__ inline void Sigmoid<T>::Compute2(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = XXX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = YYY.AllocTensor<T>();
    auto p1 = tmp1.Get<float>();
    auto p2 = tmp2.Get<float>();
    AscendC::Cast(p1, xLocal, AscendC::RoundMode::CAST_RINT, this->tileLength);
    AscendC::Muls(p1, p1, (float)-1, this->tileLength);
    AscendC::Exp(p1, p1, this->tileLength);
    AscendC::Adds(p1, p1, (float)1, this->tileLength);
    AscendC::Duplicate<float>(p2, (float)1, this->tileLength);
    AscendC::Div(p1, p2, p1, this->tileLength);
    AscendC::Cast(yLocal, p1, AscendC::RoundMode::CAST_RINT, this->tileLength);
    YYY.EnQue(yLocal);
    XXX.FreeTensor(xLocal);

}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process()
{
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

template <typename T>
__aicore__ inline void Sigmoid<T>::Process2()
{
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute2(i);
        CopyOut(i);
    }
}

} // namespace NsSigmoid
#endif // SIGMOID_H
#ifndef RMSNORM_OPTIMIZED_KERNEL_H
#define RMSNORM_OPTIMIZED_KERNEL_H

#include "kernel_operator.h"
#include "rmsnorm_tiling.h"

using namespace AscendC;

__aicore__ inline float RmsNormInvSqrtApprox(float x)
{
    if (x < 1.0e-30f) {
        x = 1.0e-30f;
    }

    float normalized = x;
    float correction = 1.0f;
    while (normalized > 2.0f) {
        normalized *= 0.25f;
        correction *= 0.5f;
    }
    while (normalized < 0.5f) {
        normalized *= 4.0f;
        correction *= 2.0f;
    }

    float y = 1.0f;
    const float half = 0.5f * normalized;
    for (int32_t i = 0; i < 6; ++i) {
        y *= 1.5f - half * y * y;
    }
    return y * correction;
}

class KernelRmsNormOptimized {
public:
    __aicore__ inline KernelRmsNormOptimized() {}

    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR weight,
        GM_ADDR output,
        uint32_t rows,
        uint32_t hidden,
        uint32_t coreNum,
        uint32_t rowsPerCore,
        float eps,
        float invHidden)
    {
        rows_ = rows;
        hidden_ = hidden;
        coreNum_ = coreNum;
        rowsPerCore_ = rowsPerCore;
        eps_ = eps;
        invHidden_ = invHidden;

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input), rows_ * hidden_);
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), hidden_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), rows_ * hidden_);

        const uint32_t bytes = hidden_ * sizeof(float);
        pipe_.InitBuffer(inputQue_, 1, bytes);
        pipe_.InitBuffer(weightQue_, 1, bytes);
        pipe_.InitBuffer(outputQue_, 1, bytes);
        pipe_.InitBuffer(squareBuf_, bytes);
        pipe_.InitBuffer(reduceBuf_, bytes);
        pipe_.InitBuffer(sumBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        if (coreId >= coreNum_) {
            return;
        }

        const uint32_t rowBegin = coreId * rowsPerCore_;
        uint32_t rowEnd = rowBegin + rowsPerCore_;
        if (rowEnd > rows_) {
            rowEnd = rows_;
        }

        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            CopyIn(row);
            Compute();
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row)
    {
        const uint32_t base = row * hidden_;
        LocalTensor<float> xLocal = inputQue_.AllocTensor<float>();
        LocalTensor<float> wLocal = weightQue_.AllocTensor<float>();

        DataCopy(xLocal, inputGm_[base], hidden_);
        DataCopy(wLocal, weightGm_[0], hidden_);

        inputQue_.EnQue(xLocal);
        weightQue_.EnQue(wLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<float> xLocal = inputQue_.DeQue<float>();
        LocalTensor<float> wLocal = weightQue_.DeQue<float>();
        LocalTensor<float> yLocal = outputQue_.AllocTensor<float>();
        LocalTensor<float> squareLocal = squareBuf_.Get<float>();
        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();

        Mul(squareLocal, xLocal, xLocal, static_cast<int32_t>(hidden_));
        ReduceSum(sumLocal, squareLocal, reduceLocal, static_cast<int32_t>(hidden_));

        const float squareSum = sumLocal.GetValue(0);
        const float scale = RmsNormInvSqrtApprox(squareSum * invHidden_ + eps_);

        Muls(yLocal, xLocal, scale, static_cast<int32_t>(hidden_));
        Mul(yLocal, yLocal, wLocal, static_cast<int32_t>(hidden_));

        outputQue_.EnQue(yLocal);
        inputQue_.FreeTensor(xLocal);
        weightQue_.FreeTensor(wLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        const uint32_t base = row * hidden_;
        LocalTensor<float> yLocal = outputQue_.DeQue<float>();
        DataCopy(outputGm_[base], yLocal, hidden_);
        outputQue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inputQue_;
    TQue<TPosition::VECIN, 1> weightQue_;
    TQue<TPosition::VECOUT, 1> outputQue_;
    TBuf<TPosition::VECCALC> squareBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;

    GlobalTensor<float> inputGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<float> outputGm_;
    uint32_t rows_ = 0;
    uint32_t hidden_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t rowsPerCore_ = 0;
    float eps_ = 1e-6f;
    float invHidden_ = 1.0f;
};

#endif

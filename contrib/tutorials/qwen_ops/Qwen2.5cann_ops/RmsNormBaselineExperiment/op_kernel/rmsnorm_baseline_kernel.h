#ifndef RMSNORM_BASELINE_KERNEL_H
#define RMSNORM_BASELINE_KERNEL_H

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

class KernelRmsNormBaseline {
public:
    __aicore__ inline KernelRmsNormBaseline() {}

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
            const uint32_t base = row * hidden_;
            float squareSum = 0.0f;
            for (uint32_t col = 0; col < hidden_; ++col) {
                const float x = inputGm_.GetValue(base + col);
                squareSum += x * x;
            }

            const float meanSquare = squareSum * invHidden_;
            const float scale = RmsNormInvSqrtApprox(meanSquare + eps_);

            for (uint32_t col = 0; col < hidden_; ++col) {
                const float x = inputGm_.GetValue(base + col);
                const float w = weightGm_.GetValue(col);
                outputGm_.SetValue(base + col, x * scale * w);
            }
        }
    }

private:
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

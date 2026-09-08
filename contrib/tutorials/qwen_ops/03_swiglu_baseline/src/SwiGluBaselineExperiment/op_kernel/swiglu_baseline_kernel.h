#ifndef SWIGLU_BASELINE_KERNEL_H
#define SWIGLU_BASELINE_KERNEL_H

#include "kernel_operator.h"
#include "swiglu_tiling.h"

using namespace AscendC;

__aicore__ inline float SwigluExpApprox(float x)
{
    if (x > 20.0f) {
        x = 20.0f;
    } else if (x < -20.0f) {
        x = -20.0f;
    }

    const float ln2 = 0.6931471805599453f;
    int32_t scale = 0;
    while (x > ln2) {
        x -= ln2;
        ++scale;
    }
    while (x < -ln2) {
        x += ln2;
        --scale;
    }

    float term = 1.0f;
    float sum = 1.0f;
    for (int32_t i = 1; i <= 10; ++i) {
        term = term * x / static_cast<float>(i);
        sum += term;
    }

    while (scale > 0) {
        sum *= 2.0f;
        --scale;
    }
    while (scale < 0) {
        sum *= 0.5f;
        ++scale;
    }
    return sum;
}

class KernelSwiGluBaseline {
public:
    __aicore__ inline KernelSwiGluBaseline() {}

    __aicore__ inline void Init(
        GM_ADDR gate,
        GM_ADDR up,
        GM_ADDR output,
        uint32_t totalSize,
        uint32_t coreNum,
        uint32_t elementsPerCore)
    {
        totalSize_ = totalSize;
        coreNum_ = coreNum;
        elementsPerCore_ = elementsPerCore;
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gate), totalSize_);
        upGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(up), totalSize_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), totalSize_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        if (coreId >= coreNum_) {
            return;
        }

        const uint32_t begin = coreId * elementsPerCore_;
        uint32_t end = begin + elementsPerCore_;
        if (end > totalSize_) {
            end = totalSize_;
        }

        for (uint32_t i = begin; i < end; ++i) {
            const float gate = gateGm_.GetValue(i);
            const float up = upGm_.GetValue(i);
            const float sigmoid = 1.0f / (1.0f + SwigluExpApprox(-gate));
            outputGm_.SetValue(i, gate * sigmoid * up);
        }
    }

private:
    GlobalTensor<float> gateGm_;
    GlobalTensor<float> upGm_;
    GlobalTensor<float> outputGm_;
    uint32_t totalSize_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t elementsPerCore_ = 0;
};

#endif
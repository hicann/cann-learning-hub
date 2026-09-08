#ifndef GEMM_BASELINE_KERNEL_H
#define GEMM_BASELINE_KERNEL_H

#include "kernel_operator.h"
#include "gemm_tiling.h"

using namespace AscendC;

class KernelGemmBaseline {
public:
    __aicore__ inline KernelGemmBaseline() {}

    __aicore__ inline void Init(
        GM_ADDR a,
        GM_ADDR b,
        GM_ADDR c,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        uint32_t coreNum,
        uint32_t rowsPerCore)
    {
        m_ = m;
        n_ = n;
        k_ = k;
        coreNum_ = coreNum;
        rowsPerCore_ = rowsPerCore;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), m_ * k_);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), k_ * n_);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), m_ * n_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        if (coreId >= coreNum_) {
            return;
        }

        const uint32_t rowBegin = coreId * rowsPerCore_;
        uint32_t rowEnd = rowBegin + rowsPerCore_;
        if (rowEnd > m_) {
            rowEnd = m_;
        }

        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            for (uint32_t col = 0; col < n_; ++col) {
                float acc = 0.0f;
                for (uint32_t kk = 0; kk < k_; ++kk) {
                    const float av = aGm_.GetValue(row * k_ + kk);
                    const float bv = bGm_.GetValue(kk * n_ + col);
                    acc += av * bv;
                }
                cGm_.SetValue(row * n_ + col, acc);
            }
        }
    }

private:
    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> cGm_;
    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t rowsPerCore_ = 0;
};

#endif
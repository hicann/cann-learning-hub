#ifndef GEMM_OPTIMIZED_KERNEL_H
#define GEMM_OPTIMIZED_KERNEL_H

#include "kernel_operator.h"
#include "gemm_tiling.h"

using namespace AscendC;

class KernelGemmOptimized {
public:
    __aicore__ inline void Init(
        GM_ADDR a,
        GM_ADDR bTrans,
        GM_ADDR c,
        const GemmOptimizedTiling &tiling)
    {
        m_ = tiling.m;
        n_ = tiling.n;
        k_ = tiling.k;
        coreNum_ = tiling.coreNum;
        tileM_ = tiling.tileM;
        tileN_ = tiling.tileN;
        tileK_ = tiling.tileK;
        totalTiles_ = tiling.totalTiles;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), m_ * k_);
        bTransGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bTrans), n_ * k_);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), m_ * n_);

        const uint32_t tileBytes = tileK_ * sizeof(float);
        pipe_.InitBuffer(aQueue_, 1, tileBytes);
        pipe_.InitBuffer(bQueue_, 1, tileBytes);
        pipe_.InitBuffer(cQueue_, 1, tileN_ * sizeof(float));
        pipe_.InitBuffer(workBuf_, (2U * tileK_ + 8U) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        if (coreId >= coreNum_) {
            return;
        }

        const uint32_t nTiles = (n_ + tileN_ - 1U) / tileN_;

        for (uint32_t tileId = coreId; tileId < totalTiles_; tileId += coreNum_) {
            const uint32_t mTileId = tileId / nTiles;
            const uint32_t nTileId = tileId % nTiles;

            const uint32_t mStart = mTileId * tileM_;
            const uint32_t nStart = nTileId * tileN_;

            const uint32_t curM = Min(tileM_, m_ - mStart);
            const uint32_t curN = Min(tileN_, n_ - nStart);

            ProcessOneCTile(mStart, nStart, curM, curN);
        }
    }

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline void ProcessOneCTile(
        uint32_t mStart,
        uint32_t nStart,
        uint32_t curM,
        uint32_t curN)
    {
        for (uint32_t mi = 0; mi < curM; ++mi) {
            const uint32_t row = mStart + mi;
            LocalTensor<float> cLocal = cQueue_.AllocTensor<float>();
            cLocal.SetSize(curN);

            for (uint32_t nj = 0; nj < curN; ++nj) {
                const uint32_t col = nStart + nj;
                float acc = 0.0f;

                for (uint32_t kStart = 0; kStart < k_; kStart += tileK_) {
                    const uint32_t curK = Min(tileK_, k_ - kStart);
                    acc += DotTile(row, col, kStart, curK);
                }

                cLocal.SetValue(nj, acc);
            }

            cQueue_.EnQue(cLocal);
            cLocal = cQueue_.DeQue<float>();
            DataCopy(cGm_[row * n_ + nStart], cLocal, curN);
            cQueue_.FreeTensor(cLocal);
        }
    }

    __aicore__ inline float DotTile(
        uint32_t row,
        uint32_t col,
        uint32_t kStart,
        uint32_t curK)
    {
        LocalTensor<float> aLocal = aQueue_.AllocTensor<float>();
        LocalTensor<float> bLocal = bQueue_.AllocTensor<float>();

        DataCopy(aLocal, aGm_[row * k_ + kStart], curK);
        DataCopy(bLocal, bTransGm_[col * k_ + kStart], curK);

        aQueue_.EnQue(aLocal);
        bQueue_.EnQue(bLocal);

        aLocal = aQueue_.DeQue<float>();
        bLocal = bQueue_.DeQue<float>();

        LocalTensor<float> workLocal = workBuf_.Get<float>();
        LocalTensor<float> prodLocal = workLocal;
        LocalTensor<float> reduceTmpLocal = workLocal[tileK_];
        LocalTensor<float> reduceResultLocal = workLocal[2U * tileK_];

        prodLocal.SetSize(curK);
        reduceTmpLocal.SetSize(curK);
        reduceResultLocal.SetSize(8U);

        Mul(prodLocal, aLocal, bLocal, curK);
        PipeBarrier<PIPE_V>();

        ReduceSum<float>(
            reduceResultLocal,
            prodLocal,
            reduceTmpLocal,
            static_cast<int32_t>(curK));
        PipeBarrier<PIPE_V>();

        const float partial = reduceResultLocal.GetValue(0);

        aQueue_.FreeTensor(aLocal);
        bQueue_.FreeTensor(bLocal);
        return partial;
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> aQueue_;
    TQue<TPosition::VECIN, 1> bQueue_;
    TQue<TPosition::VECOUT, 1> cQueue_;
    TBuf<TPosition::VECCALC> workBuf_;

    GlobalTensor<float> aGm_;
    GlobalTensor<float> bTransGm_;
    GlobalTensor<float> cGm_;

    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t tileM_ = 8;
    uint32_t tileN_ = 8;
    uint32_t tileK_ = 128;
    uint32_t totalTiles_ = 0;
};

#endif
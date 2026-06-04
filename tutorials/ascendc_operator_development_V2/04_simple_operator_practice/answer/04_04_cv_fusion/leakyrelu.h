#ifndef LEAKYRELU_H
#define LEAKYRELU_H

#include "kernel_operator.h"

constexpr uint16_t SYNC_FLAG = 0;

struct LeakyReLUCustomTilingData
{
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t baseM;
    uint32_t baseN;
    float alpha;
};

class KernelLeakyReLU {
public:
    __aicore__ inline KernelLeakyReLU() {}

    // numElem表示需要计算的元素个数
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, LeakyReLUCustomTilingData tiling)
    {
        this->m = tiling.m;
        this->n = tiling.n;
        this->k = tiling.k;
        this->baseM = tiling.baseM;
        this->baseN = tiling.baseN;
        this->leakyReluAlpha = tiling.alpha;
        uint32_t numElem = this->m * this->n;
        mmadResGm.SetGlobalBuffer((__gm__ float *)x, numElem);
        leakyreluResGm.SetGlobalBuffer((__gm__ float *)z, numElem);
        pipe.InitBuffer(inQueueX, 1, this->baseM * this->baseN * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, this->baseM * this->baseN * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        uint32_t loopNum = (this->m / this->baseM) * (this->n / this->baseN);
        for (int32_t i = 0; i < loopNum; i++) {
            AscendC::CrossCoreWaitFlag(SYNC_FLAG);
            // 一行的矩阵子块中，属于第几块
            uint32_t cubePerRow = n / baseN;
            uint32_t rowIdx = i % cubePerRow;
            uint32_t rowStartIndex = baseN * rowIdx;
            // 多行的矩阵子块中，属于第几行
            uint32_t wholeRowidx = i / cubePerRow;
            uint32_t gmIndex = wholeRowidx * baseM * n + rowStartIndex;
            CopyIn(i, gmIndex);
            Compute(i);
            CopyOut(i, gmIndex);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t gmIndex)
    {
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = this->baseM;
        copyParams.blockLen = this->baseN * sizeof(float) / AscendC::ONE_BLK_SIZE;
        copyParams.srcGap = (this->n - this->baseN) * sizeof(float) / AscendC::ONE_BLK_SIZE;
        copyParams.dstGap = 0;

        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

        AscendC::DataCopy(xLocal, mmadResGm[gmIndex], copyParams);

        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        AscendC::LeakyRelu(zLocal, xLocal, this->leakyReluAlpha, this->baseM * this->baseN);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t gmIndex)
    {
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = this->baseM;
        copyParams.blockLen = this->baseN * sizeof(float) / AscendC::ONE_BLK_SIZE;
        copyParams.srcGap = 0;
        copyParams.dstGap = (this->n - this->baseN) * sizeof(float) / AscendC::ONE_BLK_SIZE;

        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(leakyreluResGm[gmIndex], zLocal, copyParams);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<float> mmadResGm;
    AscendC::GlobalTensor<float> leakyreluResGm;
    uint32_t m, n, k;
    uint32_t baseM, baseN;
    float leakyReluAlpha;
};

#endif // LEAKYRELU_H
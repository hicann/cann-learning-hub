#ifndef MMAD_H
#define MMAD_H

#include "kernel_operator.h"
#include "leakyrelu.h"

// half type, cube block: [16, 16]
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 0;      // Cube通知Vector C侧xx任务已经完成

class KernelMmad {
public:
    __aicore__ inline KernelMmad() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c)
    {
        // m=128, n=128, k=128. 因此原始 矩阵A [128, 128], 矩阵B [128， 128]
        // 现在将矩阵A和矩阵B对半进行切分(k轴不切)，即baseM = 64, baseN = 64, 即分成小块矩阵A[64, 128]，小块矩阵B[128, 64]
        // 因此总共要进行4次小块矩阵A和小块矩阵B的矩阵乘计算
        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        biasGM.SetGlobalBuffer((__gm__ half *)bias);

        uint32_t baseASize = baseM * k * sizeof(half);
        uint32_t baseBSize = k * baseN * sizeof(half);
        uint32_t baseCSize = baseM * baseN * sizeof(float);
        pipe.InitBuffer(inQueueA1, 1, baseASize);      // GM -> L1  左矩阵 [baseM, k]
        pipe.InitBuffer(inQueueA2, 1, baseASize);      // L1 -> L0A 左矩阵 [baseM, k]
        pipe.InitBuffer(inQueueB1, 1, baseBSize);      // GM -> L1  右矩阵 [k, baseN]
        pipe.InitBuffer(inQueueB2, 1, baseBSize);      // L1 -> L0B 右矩阵 [k, baseN]
        pipe.InitBuffer(outQueueCO1, 1, baseCSize);    // L0A + L0B + BT => LOC 矩阵计算结果[baseM, baseN]
        pipe.InitBuffer(inQueueC1, 1, baseN);          // GM -> L1  偏置   [baseN]
        pipe.InitBuffer(outQueueC2, 1, baseN);         // L1 -> BT  偏置   [baseN]
    }

    __aicore__ inline void Process()
    {
        for (int mProgress = 0; mProgress < (m / baseM); mProgress++) {
            for (int nProgress = 0; nProgress < (n / baseN); nProgress++) {
                CopyIn(mProgress, nProgress);    // GM -> L1               搬运左矩阵，右矩阵，偏置
                SplitA();                        // L1 -> L0A              搬运左矩阵
                SplitB();                        // L1 -> L0B              搬运右矩阵
                SplitBias();                     // L1 -> BT               搬运偏置
                Compute();                       // L0A + L0B + BT -> L0C  进行矩阵计算
                CopyOut(mProgress, nProgress);   // L0C -> GM              搬运矩阵计算结果

                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);   // 计算完一次分型，通知Vector可以计算
            }
        }
    }

private:
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len)
    {
        return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
    }

    __aicore__ inline void CopyIn(uint32_t mProgress, uint32_t nProgress)
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        AscendC::LocalTensor<half> bias1Local = inQueueC1.AllocTensor<half>();

        // 左矩阵: GM: [baseM, k], ND format => L1: [baseM, k], Nz format
        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = baseM;
        nd2nzA1Params.dValue = k;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = k;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(baseM) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM[mProgress * baseM * k], nd2nzA1Params);

        // 右矩阵: GM: [k, baseN], ND format => L1: [k, baseN], Nz format
        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = k;
        nd2nzB1Params.dValue = baseN;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = n;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(k) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, bGM[nProgress * baseN], nd2nzB1Params);

        // 偏置: GM: [baseN], ND format => L1: [baseN], ND format
        AscendC::DataCopy(bias1Local, biasGM[nProgress * baseN], baseN);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueC1.EnQue(bias1Local);
    }

    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // 左矩阵: L1: [baseM, k], Nz format => L0A: [baseM, k], Nz format
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = CeilCubeBlock(baseM);
        loadDataParams.kStep = CeilCubeBlock(k);
        loadDataParams.srcStride = CeilCubeBlock(baseM);
        loadDataParams.dstStride = CeilCubeBlock(baseM);
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, loadDataParams);
#else
        // 左矩阵: L1: [baseM, k], Nz format => L0A: [baseM, k], Zz format
        uint32_t dstOffset = CeilCubeBlock(k) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(k);
        loadDataParams.srcStride = CeilCubeBlock(baseM);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        // 一行一行分形的遍历搬运
        for (int i = 0; i < CeilCubeBlock(baseM); ++i) {
            AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
        }
#endif
        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB()
    {
        AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // 右矩阵: L1: [k, baseN], Nz format => L0B: [k, baseN], Zn format
        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CeilCubeBlock(k) * CUBE_BLOCK_SIZE;

        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(k);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = CeilCubeBlock(baseN) - 1;
        loadDataParams.ifTranspose = true;

        // 一列一列分型的遍历搬运
        for (int i = 0; i < CeilCubeBlock(baseN); ++i) {
            AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
        }

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<half>(b2Local);
    }

    __aicore__ inline void SplitBias()
    {
        AscendC::LocalTensor<half> bias1Local = inQueueC1.DeQue<half>();
        AscendC::LocalTensor<float> bias2Local = outQueueC2.AllocTensor<float>();

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // __NPU_ARCH__为3510时，blockLen单位为32B
        AscendC::DataCopy(bias2Local, bias1Local, { 1, (uint16_t)(baseN * sizeof(float) / 32), 0, 0 });
#else
        // __NPU_ARCH__为2201时，blockLen单位为64B
        AscendC::DataCopy(bias2Local, bias1Local, { 1, (uint16_t)(baseN * sizeof(float) / 64), 0, 0 });
#endif

        outQueueC2.EnQue<float>(bias2Local);
        inQueueC1.FreeTensor(bias1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        AscendC::LocalTensor<float> bias2Local = outQueueC2.DeQue<float>();
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = baseM;
        mmadParams.n = baseN;
        mmadParams.k = k;
        mmadParams.cmatrixInitVal = false;
        AscendC::Mmad(c1Local, a2Local, b2Local, bias2Local, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
        outQueueC2.FreeTensor(bias2Local);
    }
    __aicore__ inline void CopyOut(uint32_t mProgress, uint32_t nProgress)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
#else
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
#endif
        fixpipeParams.nSize = baseN;
        fixpipeParams.mSize = baseM;
        fixpipeParams.srcStride = CeilCubeBlock(baseM) * CUBE_BLOCK;
        fixpipeParams.dstStride = n;     // dst GM matrix [m, n]
        AscendC::Fixpipe(cGM[nProgress * baseN + mProgress * baseM * n], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::QuePosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::QuePosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::QuePosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::QuePosition::CO1, 1> outQueueCO1;
    AscendC::TQue<AscendC::QuePosition::C1, 1> inQueueC1;
    AscendC::TQue<AscendC::QuePosition::C2, 1> outQueueC2;

    AscendC::GlobalTensor<half> aGM;
    AscendC::GlobalTensor<half> bGM;
    AscendC::GlobalTensor<float> cGM;
    AscendC::GlobalTensor<half> biasGM;

    uint16_t m = 128, k = 128, n = 128;
    uint16_t baseM = 32, baseN = 32;
};

__global__ __mix__(1,1) void mmad_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR leakyreluRes,
    LeakyReLUCustomTilingData tiling)
{
    if ASCEND_IS_AIC {
        KernelMmad op;
        op.Init(a, b, bias, c);
        op.Process();
    } else { // ASCEND_IS_AIV
        KernelLeakyReLU op;
        op.Init(c, leakyreluRes, tiling);
        op.Process();
    }
}

#endif // MMAD_H
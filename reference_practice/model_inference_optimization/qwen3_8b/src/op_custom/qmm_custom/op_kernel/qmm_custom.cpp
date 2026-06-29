#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "qmm_custom_tiling.h"

using namespace AscendC;

using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using BType = matmul::MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

__aicore__ inline int32_t CeilDiv(int32_t value, int32_t factor)
{
    return (value + factor - 1) / factor;
}

class QmmCubeBasicKernel {
public:
    __aicore__ inline QmmCubeBasicKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, QmmCustomTilingData* tilingData)
    {
        M = tilingData->M;
        N = tilingData->N;
        K = tilingData->K;

        x1GM.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), M * K);
        x2GM.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), K * N);
        yGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y), M * N);

        mm.SetOrgShape(M, N, K);
        mm.SetTensorA(x1GM, false);
        mm.SetTensorB(x2GM, false);
    }

    __aicore__ inline void Process()
    {
        mm.IterateAll(yGM);
        mm.End();
    }

    matmul::Matmul<AType, BType, CType> mm;

private:
    int32_t M;
    int32_t N;
    int32_t K;
    GlobalTensor<int8_t> x1GM;
    GlobalTensor<int8_t> x2GM;
    GlobalTensor<int32_t> yGM;
};

class QmmPertokenKernel {
public:
    __aicore__ inline QmmPertokenKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR tmp, GM_ADDR pertokenScale,
                                GM_ADDR y, QmmCustomTilingData* tilingData, TPipe* tPipe)
    {
        pipe = tPipe;
        M = tilingData->M;
        N = tilingData->N;
        K = tilingData->K;
        singleCoreM = tilingData->singleCoreM;
        singleCoreN = tilingData->singleCoreN;

        x1GM.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1), M * K);
        x2GM.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2), K * N);
        tmpGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tmp), M * N);
        yGM.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y), M * N);
        scaleGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), N);
        pertokenGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), M);

        if ASCEND_IS_AIC {
            mm.SetOrgShape(M, N, K);
            mm.SetTensorA(x1GM, false);
            mm.SetTensorB(x2GM, false);
        }

        if ASCEND_IS_AIV {
            pipe->InitBuffer(intQueue, 1, TILE_N * sizeof(int32_t));
            pipe->InitBuffer(floatBuf, TILE_N * sizeof(float));
            pipe->InitBuffer(scaleQueue, 1, TILE_N * sizeof(float));
            pipe->InitBuffer(outQueue, 1, TILE_N * sizeof(bfloat16_t));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            mm.IterateAll(tmpGM);
            mm.End();
            CrossCoreSetFlag<0x2, PIPE_FIX>(8);
        }

        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(8);

            int32_t pairedAicIdx = static_cast<int32_t>(GetBlockIdx()) / 2;
            int32_t aivSubIdx = static_cast<int32_t>(GetBlockIdx()) % 2;
            int32_t mBlockCount = CeilDiv(M, singleCoreM);
            int32_t mBlockIdx = pairedAicIdx % mBlockCount;
            int32_t nBlockIdx = pairedAicIdx / mBlockCount;
            int32_t coreRowStart = mBlockIdx * singleCoreM;
            int32_t coreRowEnd = (coreRowStart + singleCoreM < M) ? (coreRowStart + singleCoreM) : M;
            int32_t coreColStart = nBlockIdx * singleCoreN;
            int32_t coreColEnd = (coreColStart + singleCoreN < N) ? (coreColStart + singleCoreN) : N;
            int32_t rowPerAiv = CeilDiv(coreRowEnd - coreRowStart, 2);
            int32_t rowStart = coreRowStart + aivSubIdx * rowPerAiv;
            int32_t rowEnd = rowStart + rowPerAiv;
            rowEnd = (rowEnd < coreRowEnd) ? rowEnd : coreRowEnd;

            for (int32_t row = rowStart; row < rowEnd; ++row) {
                float pertokenVal = pertokenGM.GetValue(row);
                for (int32_t col = coreColStart; col < coreColEnd; col += TILE_N) {
                    int32_t curTile = (col + TILE_N <= coreColEnd) ? TILE_N : (coreColEnd - col);

                    LocalTensor<int32_t> intLocal = intQueue.AllocTensor<int32_t>();
                    LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
                    LocalTensor<float> floatLocal = floatBuf.Get<float>();

                    DataCopy(intLocal, tmpGM[row * N + col], curTile);
                    DataCopy(scaleLocal, scaleGM[col], curTile);
                    intQueue.EnQue(intLocal);
                    scaleQueue.EnQue(scaleLocal);

                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    intLocal = intQueue.DeQue<int32_t>();
                    scaleLocal = scaleQueue.DeQue<float>();
                    LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();
                    Cast(floatLocal, intLocal, RoundMode::CAST_NONE, curTile);
                    PipeBarrier<PIPE_V>();
                    Mul(floatLocal, floatLocal, scaleLocal, curTile);
                    PipeBarrier<PIPE_V>();
                    Muls(floatLocal, floatLocal, pertokenVal, curTile);
                    PipeBarrier<PIPE_V>();
                    Cast(outLocal, floatLocal, RoundMode::CAST_RINT, curTile);
                    outQueue.EnQue<bfloat16_t>(outLocal);
                    intQueue.FreeTensor(intLocal);
                    scaleQueue.FreeTensor(scaleLocal);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                    outLocal = outQueue.DeQue<bfloat16_t>();
                    DataCopy(yGM[row * N + col], outLocal, curTile);
                    outQueue.FreeTensor(outLocal);
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                }
            }
        }
    }

    matmul::Matmul<AType, BType, CType> mm;

private:
    static constexpr int32_t TILE_N = 4096;

    TPipe* pipe;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t singleCoreM;
    int32_t singleCoreN;

    TQue<TPosition::VECIN, 1> intQueue;
    TBuf<TPosition::VECCALC> floatBuf;
    TQue<TPosition::VECIN, 1> scaleQueue;
    TQue<TPosition::VECOUT, 1> outQueue;

    GlobalTensor<int8_t> x1GM;
    GlobalTensor<int8_t> x2GM;
    GlobalTensor<int32_t> tmpGM;
    GlobalTensor<bfloat16_t> yGM;
    GlobalTensor<float> scaleGM;
    GlobalTensor<float> pertokenGM;
};

extern "C" __global__ __aicore__ void qmm_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR tmp,
                                                  GM_ADDR pertoken_scale, GM_ADDR y, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    TPipe pipe;
    REGISTER_TILING_DEFAULT(QmmCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.isPertoken == 0) {
        if ASCEND_IS_AIC {
            QmmCubeBasicKernel kernel;
            REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.mm, &tilingData.cubeTilingData);
            kernel.Init(x1, x2, y, &tilingData);
            kernel.Process();
        }
    } else {
        QmmPertokenKernel kernel;
        if ASCEND_IS_AIC {
            REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.mm, &tilingData.cubeTilingData);
        }
        kernel.Init(x1, x2, scale, tmp, pertoken_scale, y, &tilingData, &pipe);
        kernel.Process();
    }

    pipe.Destroy();
}

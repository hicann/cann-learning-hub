#include "kernel_operator.h"
#include "matmul_sinh_tiling.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType, typename biasType>
class MatmulSinhKernel {
public:
    __aicore__ inline MatmulSinhKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void MatmulCompute();
    __aicore__ inline void SinhCompute();  // 修改：Abs → Sinh
    __aicore__ inline void CopyOut(uint32_t count);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    AscendC::LocalTensor<cType> sinhOutLocal;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;

    TCubeTiling tiling;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> sinhOutQueue_;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulSinhKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                                       const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);
    
    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    // 初始化 buffer 名称修改
    pipe->InitBuffer(sinhOutQueue_, 1, tiling.baseM * tiling.baseN * sizeof(cType));
    pipe->InitBuffer(tmp1, tiling.baseM * tiling.baseN * sizeof(cType));
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulSinhKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    uint32_t computeRound = 0;
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    while (matmulObj.template Iterate<true>()) {
        MatmulCompute();
        SinhCompute();
        CopyOut(computeRound);
        computeRound++;
    }
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulSinhKernel<aType, bType, cType, biasType>::MatmulCompute()
{
    sinhOutLocal = sinhOutQueue_.AllocTensor<cType>();
    matmulObj.template GetTensorC<true>(sinhOutLocal, false, true);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulSinhKernel<aType, bType, cType, biasType>::SinhCompute()
{
    auto tmpTensor = tmp1.Get<cType>();
    // sinh(x) = (exp(x) - exp(-x)) * 0.5
    AscendC::Exp(tmpTensor, sinhOutLocal, tiling.baseM * tiling.baseN);
    AscendC::Muls(sinhOutLocal, sinhOutLocal, (cType)-1, tiling.baseM * tiling.baseN);
    AscendC::Exp(sinhOutLocal, sinhOutLocal, tiling.baseM * tiling.baseN);
    AscendC::Sub(sinhOutLocal, tmpTensor, sinhOutLocal, tiling.baseM * tiling.baseN);
    AscendC::Muls(sinhOutLocal, sinhOutLocal, (cType)0.5, tiling.baseM * tiling.baseN);
    sinhOutQueue_.EnQue(sinhOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulSinhKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count)
{
    sinhOutQueue_.DeQue<cType>();
    const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count % roundM * tiling.baseM * tiling.N + count / roundM * tiling.baseN);
    DataCopyParams copyParam = {(uint16_t)tiling.baseM, (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE), 0,
                                (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)};
    DataCopy(cGlobal[startOffset], sinhOutLocal, copyParam);
    sinhOutQueue_.FreeTensor(sinhOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulSinhKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                             int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_sinh(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MatmulSinhTilingData);
    GET_TILING_DATA(tilingData, tiling);
    MatmulSinhKernel<float, float, float, float> op;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tilingData.cubeTilingData);
    op.Init(a, b, bias, c, workspace, tilingData.cubeTilingData, &pipe);
    op.Process(&pipe);
}
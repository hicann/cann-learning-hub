
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;
// 辅助函数：整数除法向上取整，用于分块数量计算
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType, typename biasType> 
class MatmulKernel {
public:
    __aicore__ inline MatmulKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                uint64_t memSize, const TCubeTiling &tiling);
    template <bool setTmpSpace = false> __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    // Matmul 高阶 API 核心对象，模板参数描述 A/B/C/Bias 的类型信息：
    // - 内存位置：GM（全局内存）
    // - 数据格式：ND（行优先，API内部自动转换为分形格式）
    // - 数据类型：由模板参数 aType/bType/cType/biasType 指定
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    // 当前核所负责的全局张量视图（初始化后偏移至子区域）
    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;

    TCubeTiling tiling;      // 保存主机侧传递的切分参数
    uint64_t localMemSize;   // 保存当前平台的UB大小，用于工作空间分配
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                        int32_t &offsetA, int32_t &offsetB,
                                                        int32_t &offsetC, int32_t &offsetBias)
{
    // M 维度上可划分的单核块数（向上取整，处理尾块）
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    // 当前核在 M 方向上的索引（0 ~ mSingleBlocks-1）
    auto mCoreIndx = blockIdx % mSingleBlocks;
    // 当前核在 N 方向上的索引
    auto nCoreIndx = blockIdx / mSingleBlocks;

    // A 矩阵形状 [M, Ka]，行优先存储：偏移 = 起始行号 × Ka
    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    // B 矩阵形状 [Kb, N]，行优先存储：偏移 = 起始列号（单核负责连续 N 方向块）
    offsetB = nCoreIndx * tiling.singleCoreN;
    // C 矩阵形状 [M, N]：偏移 = 起始行号 × N + 起始列号
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    // Bias 向量长度 N，按 N 方向切分：偏移 = 起始列号
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                  GM_ADDR workspace, uint64_t memSize,
                                                  const TCubeTiling &tiling)
{
    // 保存 Tiling 参数和 UB 大小
    this->tiling = tiling;
    this->localMemSize = memSize;

    // 将全局内存地址绑定到 GlobalTensor，并指定完整张量的总元素个数
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    // 计算当前核在全局矩阵中的偏移量
    int32_t offsetA = 0, offsetB = 0, offsetC = 0, offsetBias = 0;
    CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);

    // 通过 GlobalTensor 的下标运算符 [offset] 获取从偏移位置开始的子张量视图
    // 该操作仅记录新地址，不涉及数据搬运
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];

    // 系统工作空间指针为空时直接返回
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType, typename biasType>
template <bool setTmpSpace>
__aicore__ inline void
MatmulKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    // 步骤4.1（可选）：显式分配 UB 工作空间并将其起始物理地址传入给Matmul
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, localMemSize);     // 初始化缓冲区，大小为 localMemSize
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(localMemSize); // 获取指定大小的本地张量
        matmulObj.SetLocalWorkspace(mmformatUb);           // 将 UB 空间设置给 Matmul 对象
    }

    // 步骤4.2：将当前核的输入地址传递给 Matmul 对象。
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);

    // 步骤4.3：执行完整的矩阵乘法，结果直接写入 cGlobal 指向的全局内存
    matmulObj.IterateAll(cGlobal);

    // 步骤5：结束矩阵乘操作，释放Matmul计算资源
    matmulObj.End();
}

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // ----- 步骤1（实例化）：创建 MatmulKernel 对象，内部包含 Matmul 对象 -----
    MatmulKernel<int8_t, int8_t, int32_t, int32_t> matmulKernel;

    // ----- 步骤2：初始化 Matmul 对象 -----
    AscendC::TPipe pipe;
    // REGIST_MATMUL_OBJ 宏完成 Matmul 对象与流水线、Tiling 参数的绑定
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj,
                      &tilingData.cubeTilingData);

    // ----- 步骤3：设置输入数据（初始化 + 偏移切片）-----
    matmulKernel.Init(a, b, bias, c, workspace,
                      tilingData.localMemSize, tilingData.cubeTilingData);

    // ----- 步骤4 & 5：根据 TilingKey 选择执行路径，完成计算与清理 -----
    if (TILING_KEY_IS(1)) {
        // 平台无需显式 UB 空间（如 Ascend910B）
        matmulKernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        // 平台需要显式 UB 空间（如 Ascend310P）
        matmulKernel.Process<true>(&pipe);
    }
}
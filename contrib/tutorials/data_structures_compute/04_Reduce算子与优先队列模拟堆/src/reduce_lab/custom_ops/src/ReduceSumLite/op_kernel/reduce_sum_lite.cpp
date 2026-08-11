/*
 * ============================================================
 * ReduceSumLite — 求和规约算子 Kernel（教学版）
 * ============================================================
 *
 * 【教学目标】
 * 1. 理解 AscendC 编程模型：GM → 计算 → GM 的数据流
 * 2. 理解多核并行：8个 AI Core 各自处理数据的一个切片
 * 3. 理解 Tiling 机制：Host 端计算分块策略，通过 tiling 结构体传递给 kernel
 * 4. 理解输出 tensor vs workspace 的区别
 *
 * 【AscendC 核心概念】
 * - GM (Global Memory): 设备全局内存，所有核共享
 * - UB (Unified Buffer): 核心本地内存，用于计算
 * - GlobalTensor<T>: GM 上的 tensor 视图，通过 SetValue/GetValue 访问
 * - GetBlockIdx(): 获取当前核的编号 (0~7)
 *
 * 【数据流】
 *   Host: 读取输入 x → 分配输出 y[BLOCK_DIM] → 调用 kernel
 *   Kernel: 读取 x[start..end] → 累加 → 写入 y[blockId]
 *   Host: 读取 y[0..7] → 求和 → 最终结果
 */
#include "reduce_sum_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

class KernelReduceSumLite {
public:
    __aicore__ inline KernelReduceSumLite() {}

    /*
     * Init: 初始化 kernel
     *
     * 【关键】GlobalTensor::SetGlobalBuffer 绑定 GM 地址
     * - xGm: 输入数据 (N 个 half)
     * - yGm: 输出数据 (8 个 half，每个核写一个)
     *
     * 【注意】输出 tensor y 的 shape 是 [BLOCK_DIM]，不是 [1]
     * 这是为了解决 310B 的 D-cache/UB 硬件错误：
     * SetValue 写 workspace 会触发错误，写输出 tensor 则安全
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileLength, uint32_t tileNum,
                                uint32_t lastTileLength, uint32_t outputSize) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->lastTileLength = lastTileLength;
        this->outputSize = outputSize;
        xGm.SetGlobalBuffer((__gm__ half *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ half *)y, outputSize);
    }

    /*
     * Process: 执行规约计算
     *
     * 【多核切分】
     *   blockId = 0..7 (310B 有 8 个 AI Core)
     *   start = blockId * blockLength
     *   end = min(start + blockLength, totalLength)
     *   每个核只处理自己负责的数据范围
     *
     * 【规约逻辑】
     *   1. 遍历自己负责的数据范围
     *   2. 用 GetValue() 从 GM 读取每个元素
     *   3. 转换为 float 进行累加（避免 FP16 精度损失）
     *   4. 最终结果转回 half，写入输出 tensor y[blockId]
     *
     * 【Tiling 参数说明】
     *   - tileLength = 1024: 每次循环处理 1024 个元素
     *   - tileNum: 循环次数 = ceil(blockLength / tileLength)
     *   - lastTileLength: 最后一次循环的有效长度
     */
    __aicore__ inline void Process() {
        uint32_t blockId = GetBlockIdx();
        uint32_t start = blockId * blockLength;
        uint32_t end = start + blockLength;
        if (end > totalLength) end = totalLength;

        float localSum = 0.0f;
        for (uint32_t t = 0; t < tileNum; ++t) {
            uint32_t tileStart = start + t * tileLength;
            if (tileStart >= end) break;
            uint32_t validLen = tileLength;
            if (tileStart + validLen > end) validLen = end - tileStart;
            for (uint32_t i = 0; i < validLen; ++i) {
                float v = static_cast<float>(xGm.GetValue(tileStart + i));
                localSum += v;
            }
        }
        // 写入输出 tensor（不是 workspace！）
        // 这是解决 310B D-cache 错误的关键
        yGm.SetValue(blockId, static_cast<half>(localSum));
    }

private:
    GlobalTensor<half> xGm;  // 输入 tensor (GM)
    GlobalTensor<half> yGm;  // 输出 tensor (GM)
        uint32_t totalLength, blockLength, tileLength, tileNum, lastTileLength, outputSize;
};

__aicore__ inline void RunReduceSumLite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelReduceSumLite op;
    op.Init(x, y, workspace, tilingData.totalLength, tilingData.blockLength,
            tilingData.tileLength, tilingData.tileNum, tilingData.lastTileLength,
            tilingData.outputSize);
    op.Process();
}

/*
 * kernel 入口函数
 * 【关键】extern "C" __global__ __aicore__ 是 AscendC kernel 的标准签名
 * 参数：x(输入), y(输出), workspace(未使用), tiling(分块参数)
 */
extern "C" __global__ __aicore__ void reduce_sum_lite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    RunReduceSumLite(x, y, workspace, tiling);
}
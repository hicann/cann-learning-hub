/*
 * ReduceSumLite — ReduceSum 补全题参考答案
 *
 * 这是 ReduceSum 算子的完整 Kernel 实现。
 * 关键逻辑在 Process() 方法中：
 *   1. GetBlockIdx() 获取当前核编号
 *   2. 计算每个核负责的数据范围 [start, end)
 *   3. 用 GetValue() 从 GM 读取数据，float 累加
 *   4. 用 SetValue() 写入输出 tensor y[blockId]
 */
#include "reduce_sum_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

class KernelReduceSumLite {
public:
    __aicore__ inline KernelReduceSumLite() {}

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

    __aicore__ inline void Process() {
        // 【TODO 1】获取当前核编号
        uint32_t blockId = GetBlockIdx();

        // 【TODO 2】计算本核负责的数据范围
        uint32_t start = blockId * blockLength;
        uint32_t end = start + blockLength;
        if (end > totalLength) end = totalLength;

        // 【TODO 3】累加求和（用 float 避免 FP16 精度损失）
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

        // 【TODO 4】写入输出 tensor（不是 workspace！）
        yGm.SetValue(blockId, static_cast<half>(localSum));
    }

private:
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
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

extern "C" __global__ __aicore__ void reduce_sum_lite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    RunReduceSumLite(x, y, workspace, tiling);
}
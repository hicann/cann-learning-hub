/*
 * ============================================================
 * ReduceMaxLite — 最大值规约算子 Kernel（教学版）
 * ============================================================
 *
 * 【与 ReduceSumLite 的区别】
 * 唯一区别：规约操作从 "+" 变成 "max"，单位元从 0 变成 -inf
 * 这展示了如何通过修改规约操作来实现不同的规约算子
 *
 * 【教学目标】
 * 1. 理解"规约操作"的抽象：任何二元结合运算都可以用同样的多核切分框架
 * 2. 理解单位元（identity element）的概念
 *    - 求和：identity = 0  (x + 0 = x)
 *    - 最大值：identity = -∞ (max(x, -∞) = x)
 */
#include "reduce_max_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

class KernelReduceMaxLite {
public:
    __aicore__ inline KernelReduceMaxLite() {}

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
        uint32_t blockId = GetBlockIdx();
        uint32_t start = blockId * blockLength;
        uint32_t end = start + blockLength;
        if (end > totalLength) end = totalLength;

        // 单位元：-1e30f（近似 -∞）
        // 任何实数 x 都满足 max(x, -1e30f) = x
        float localMax = -1e30f;
        for (uint32_t t = 0; t < tileNum; ++t) {
            uint32_t tileStart = start + t * tileLength;
            if (tileStart >= end) break;
            uint32_t validLen = tileLength;
            if (tileStart + validLen > end) validLen = end - tileStart;
            for (uint32_t i = 0; i < validLen; ++i) {
                float v = static_cast<float>(xGm.GetValue(tileStart + i));
                // 规约操作：max（对比 ReduceSumLite 的 +）
                if (v > localMax) localMax = v;
            }
        }
        yGm.SetValue(blockId, static_cast<half>(localMax));
    }

private:
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    uint32_t totalLength, blockLength, tileLength, tileNum, lastTileLength, outputSize;
};

__aicore__ inline void RunReduceMaxLite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelReduceMaxLite op;
    op.Init(x, y, workspace, tilingData.totalLength, tilingData.blockLength,
            tilingData.tileLength, tilingData.tileNum, tilingData.lastTileLength,
            tilingData.outputSize);
    op.Process();
}

extern "C" __global__ __aicore__ void reduce_max_lite(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    RunReduceMaxLite(x, y, workspace, tiling);
}
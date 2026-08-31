/*
 * ReduceMaxLite — ReduceMax 扩展实践参考答案
 *
 * 与 ReduceSum 的区别：
 *   - 规约操作：max(a,b) 替代 a+b
 *   - 单位元：-1e30f（近似 -∞）替代 0.0f
 *   - 其余完全相同
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

        // 【关键区别】单位元为 -inf，而非 0
        float localMax = -1e30f;
        for (uint32_t t = 0; t < tileNum; ++t) {
            uint32_t tileStart = start + t * tileLength;
            if (tileStart >= end) break;
            uint32_t validLen = tileLength;
            if (tileStart + validLen > end) validLen = end - tileStart;
            for (uint32_t i = 0; i < validLen; ++i) {
                float v = static_cast<float>(xGm.GetValue(tileStart + i));
                // 【关键区别】比较取最大值，而非累加
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
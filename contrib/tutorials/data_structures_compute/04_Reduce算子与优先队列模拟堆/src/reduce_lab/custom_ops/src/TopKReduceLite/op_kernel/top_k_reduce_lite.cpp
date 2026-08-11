/*
 * ============================================================
 * TopKReduceLite — TopK 选择算子 Kernel（教学版）
 * ============================================================
 *
 * 【教学目标】
 * 1. 理解小根堆（min-heap）在 TopK 选择中的应用
 * 2. 理解堆的两种操作：insert（sift up）和 replace root（sift down）
 * 3. 理解多核并行 TopK 的合并策略
 *
 * 【算法说明】
 * 目标：从 N 个元素中选出最大的 K 个
 *
 * 方法：每个核维护一个大小为 K 的小根堆
 *   - 堆顶是堆中最小的元素
 *   - 遍历数据时：
 *     * 如果堆未满（size < K）：直接插入，sift up
 *     * 如果新元素 > 堆顶：替换堆顶，sift down
 *   - 遍历结束后，堆中就是该核负责数据的 TopK
 *
 * 【为什么用小根堆？】
 *   - 大根堆：堆顶是最大值，无法快速判断新元素是否应该入堆
 *   - 小根堆：堆顶是最小值，新元素只需和堆顶比较
 *   - 时间复杂度：O(N * logK)，远优于排序的 O(N * logN)
 *
 * 【多核合并】
 *   每个核输出局部 TopK → Host 端合并 8×K 个候选 → 全局 TopK
 */
#include "top_k_reduce_lite_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

static constexpr uint32_t MAX_K = 8;  // 堆的最大容量（编译时常量）

class KernelTopKReduceLite {
public:
    __aicore__ inline KernelTopKReduceLite() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                GM_ADDR workspace,
                                uint32_t totalLength, uint32_t topK,
                                uint32_t blockLength, uint32_t tileLength,
                                uint32_t tileNum, uint32_t lastTileLength,
                                uint32_t blockDim) {
        this->totalLength = totalLength;
        this->topK = topK;
        this->blockLength = blockLength;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->lastTileLength = lastTileLength;
        this->blockDim = blockDim;
        xGm.SetGlobalBuffer((__gm__ half *)x, totalLength);
        // 输出 tensor（不是 workspace！）
        valGm.SetGlobalBuffer((__gm__ half *)values, blockDim * topK);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)indices, blockDim * topK);
    }

    __aicore__ inline void Process() {
        uint32_t blockId = GetBlockIdx();
        uint32_t start = blockId * blockLength;
        uint32_t end = start + blockLength;
        if (end > totalLength) end = totalLength;

        // 【小根堆】用数组表示（parent=i, left=2i+1, right=2i+2）
        float hVal[MAX_K];   // 堆的值数组
        int32_t hIdx[MAX_K]; // 堆的索引数组（记录原始位置）
        uint32_t hN = 0;     // 堆中元素数量

        if (start < end) {
            for (uint32_t t = 0; t < tileNum; ++t) {
                uint32_t tileStart = start + t * tileLength;
                if (tileStart >= end) break;
                uint32_t validLen = tileLength;
                if (tileStart + validLen > end) validLen = end - tileStart;
                for (uint32_t i = 0; i < validLen; ++i) {
                    uint32_t gi = tileStart + i;
                    float cs = static_cast<float>(xGm.GetValue(gi));
                    int32_t ci = static_cast<int32_t>(gi);

                    if (hN < topK) {
                        // 【阶段1：堆未满】直接插入，sift up
                        hVal[hN] = cs; hIdx[hN] = ci;
                        int32_t idx = static_cast<int32_t>(hN);
                        // Sift up：从插入位置向上调整
                        while (idx > 0) {
                            int32_t p = (idx - 1) / 2;  // 父节点
                            // 比较规则：值小的在下，值相同时索引大的在下
                            if (!(hVal[idx] < hVal[p] || (hVal[idx] == hVal[p] && hIdx[idx] > hIdx[p]))) break;
                            // 交换
                            float tv = hVal[idx]; hVal[idx] = hVal[p]; hVal[p] = tv;
                            int32_t ti = hIdx[idx]; hIdx[idx] = hIdx[p]; hIdx[p] = ti;
                            idx = p;
                        }
                        hN++;
                    } else if (cs > hVal[0] || (cs == hVal[0] && ci < hIdx[0])) {
                        // 【阶段2：堆已满】新元素比堆顶大 → 替换堆顶，sift down
                        hVal[0] = cs; hIdx[0] = ci;
                        // Sift down：从堆顶向下调整
                        int32_t idx = 0;
                        while (true) {
                            int32_t l = 2*idx+1, r = 2*idx+2, w = idx;
                            if (l < static_cast<int32_t>(hN) && (hVal[l] < hVal[w] || (hVal[l] == hVal[w] && hIdx[l] > hIdx[w]))) w = l;
                            if (r < static_cast<int32_t>(hN) && (hVal[r] < hVal[w] || (hVal[r] == hVal[w] && hIdx[r] > hIdx[w]))) w = r;
                            if (w == idx) break;  // 已经是最小的
                            float tv = hVal[idx]; hVal[idx] = hVal[w]; hVal[w] = tv;
                            int32_t ti = hIdx[idx]; hIdx[idx] = hIdx[w]; hIdx[w] = ti;
                            idx = w;
                        }
                    }
                }
            }
        }

        // 【排序】将堆中元素按值降序排列（插入排序，K 很小所以足够快）
        for (uint32_t i = 1; i < hN; ++i) {
            float ks = hVal[i]; int32_t ki = hIdx[i];
            int32_t j = static_cast<int32_t>(i) - 1;
            while (j >= 0 && (ks > hVal[j] || (ks == hVal[j] && ki < hIdx[j]))) {
                hVal[j+1] = hVal[j]; hIdx[j+1] = hIdx[j]; j--;
            }
            hVal[j+1] = ks; hIdx[j+1] = ki;
        }

        // 【写输出】每个核写自己的切片 values[core*K .. core*K+K]
        for (uint32_t k = 0; k < topK; ++k) {
            valGm.SetValue(blockId * topK + k, (k < hN) ? static_cast<half>(hVal[k]) : static_cast<half>(-1e30f));
            idxGm.SetValue(blockId * topK + k, (k < hN) ? hIdx[k] : -1);
        }
    }
private:
    GlobalTensor<half> xGm;
    GlobalTensor<half> valGm;      // 输出：TopK 值
    GlobalTensor<int32_t> idxGm;   // 输出：TopK 索引
    uint32_t totalLength, topK, blockLength, tileLength, tileNum, lastTileLength, blockDim;
};

__aicore__ inline void RunTopKReduceLite(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelTopKReduceLite op;
    op.Init(x, values, indices, workspace, tilingData.totalLength, tilingData.topK,
            tilingData.blockLength, tilingData.tileLength, tilingData.tileNum, tilingData.lastTileLength,
            tilingData.blockDim);
    op.Process();
}

extern "C" __global__ __aicore__ void top_k_reduce_lite(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    RunTopKReduceLite(x, values, indices, workspace, tiling);
}
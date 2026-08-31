/*
 * TreeQueuePipelineLite — Ascend C kernel
 *
 * 输入：
 *   parent[N]：树父节点索引，根节点为 -1
 *   cost[N]：每个任务的 Compute 耗时
 *   order[N]：Host 侧 FIFO 或优先队列产生的执行顺序
 * 输出：
 *   stage_end[N]：按原任务编号记录 CopyOut 完成时间
 *   dependency_ok[1]：父节点是否都排在子节点之前
 *
 * 该算子使用一个 control block 保持调度递推顺序，内部模拟两个缓冲槽和
 * 两个 Compute lane。这是便于教学和 910B 迁移的控制流参考实现，不把树堆
 * 调度伪装成可无条件并行的逐元素算子。
 */
#include "tree_queue_pipeline_lite_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

class KernelTreeQueuePipelineLite {
public:
    __aicore__ inline void Init(GM_ADDR parent, GM_ADDR cost, GM_ADDR order,
                                GM_ADDR stageEnd, GM_ADDR dependencyOk,
                                GM_ADDR workspace, uint32_t taskCount,
                                uint32_t queueDepth, uint32_t computeLanes,
                                float copyIn, float copyOut) {
        this->taskCount = taskCount;
        this->queueDepth = queueDepth;
        this->computeLanes = computeLanes;
        this->copyIn = copyIn;
        this->copyOut = copyOut;
        parentGm.SetGlobalBuffer((__gm__ int32_t *)parent, taskCount);
        costGm.SetGlobalBuffer((__gm__ half *)cost, taskCount);
        orderGm.SetGlobalBuffer((__gm__ int32_t *)order, taskCount);
        stageEndGm.SetGlobalBuffer((__gm__ half *)stageEnd, taskCount);
        dependencyGm.SetGlobalBuffer((__gm__ int32_t *)dependencyOk, 1);
    }

    __aicore__ inline void Process() {
        if (queueDepth != 2 || computeLanes != 2 || taskCount == 0) {
            dependencyGm.SetValue(0, 0);
            return;
        }

        // 校验 order 中的任务编号是否合法；910B 上标量写 workspace
        // 会触发 D-cache 错误，因此不建位置表，改用在线扫描检查父子约束。
        int32_t valid = 1;
        for (uint32_t i = 0; i < taskCount; ++i) {
            int32_t task = orderGm.GetValue(i);
            if (task < 0 || task >= static_cast<int32_t>(taskCount)) {
                valid = 0;
                continue;
            }
            int32_t parent = parentGm.GetValue(static_cast<uint32_t>(task));
            if (parent >= 0 && parent < static_cast<int32_t>(taskCount)) {
                bool found = false;
                for (uint32_t j = 0; j < i; ++j) {
                    if (orderGm.GetValue(j) == parent) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    valid = 0;
                }
            } else if (parent >= static_cast<int32_t>(taskCount)) {
                valid = 0;
            }
        }

        float copyFree = 0.0f;
        float computeFree[2] = {0.0f, 0.0f};
        float copyOutFree = 0.0f;
        float slotFree[2] = {0.0f, 0.0f};

        for (uint32_t i = 0; i < taskCount; ++i) {
            int32_t task = orderGm.GetValue(i);
            if (task < 0 || task >= static_cast<int32_t>(taskCount)) {
                continue;
            }

            uint32_t slot = slotFree[0] <= slotFree[1] ? 0 : 1;
            float copyInStart = copyFree > slotFree[slot] ? copyFree : slotFree[slot];
            float copyInEnd = copyInStart + copyIn;

            uint32_t lane = computeFree[0] <= computeFree[1] ? 0 : 1;
            float computeStart = computeFree[lane] > copyInEnd ? computeFree[lane] : copyInEnd;
            float computeEnd = computeStart + static_cast<float>(costGm.GetValue(static_cast<uint32_t>(task)));

            float copyOutStart = copyOutFree > computeEnd ? copyOutFree : computeEnd;
            float copyOutEnd = copyOutStart + copyOut;

            slotFree[slot] = copyOutEnd;
            copyFree = copyInEnd;
            computeFree[lane] = computeEnd;
            copyOutFree = copyOutEnd;
            stageEndGm.SetValue(static_cast<uint32_t>(task), static_cast<half>(copyOutEnd));
        }

        dependencyGm.SetValue(0, valid);
    }

private:
    GlobalTensor<int32_t> parentGm;
    GlobalTensor<half> costGm;
    GlobalTensor<int32_t> orderGm;
    GlobalTensor<half> stageEndGm;
    GlobalTensor<int32_t> dependencyGm;
    uint32_t taskCount = 0;
    uint32_t queueDepth = 0;
    uint32_t computeLanes = 0;
    float copyIn = 0.0f;
    float copyOut = 0.0f;
};

__aicore__ inline void RunTreeQueuePipelineLite(
    GM_ADDR parent, GM_ADDR cost, GM_ADDR order, GM_ADDR stageEnd,
    GM_ADDR dependencyOk, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelTreeQueuePipelineLite op;
    op.Init(parent, cost, order, stageEnd, dependencyOk, workspace,
            tilingData.taskCount, tilingData.queueDepth, tilingData.computeLanes,
            tilingData.copyIn, tilingData.copyOut);
    op.Process();
}

extern "C" __global__ __aicore__ void tree_queue_pipeline_lite(
    GM_ADDR parent, GM_ADDR cost, GM_ADDR order, GM_ADDR stageEnd,
    GM_ADDR dependencyOk, GM_ADDR workspace, GM_ADDR tiling) {
    RunTreeQueuePipelineLite(parent, cost, order, stageEnd, dependencyOk, workspace, tiling);
}

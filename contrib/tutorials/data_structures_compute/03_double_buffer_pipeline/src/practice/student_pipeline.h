#pragma once

#include <cstdint>

// TODO 1：双缓冲需要为每个 TQue 分配两个物理 Buffer。
// 当前值 1 是可运行的单缓冲基线，请改为 2。
constexpr int32_t kStudentBufferNum = 1;

// TODO 2：把当前严格串行调度重构为双缓冲预取调度。
// 要求：
// 1. 循环前 CopyIn(0)；
// 2. 每轮先 DeQue 当前输入；
// 3. 有下一块时 CopyIn(tile + 1)；
// 4. 从第二轮开始 CopyOut(tile - 1)；
// 5. 计算当前块；
// 6. 循环后 CopyOut(tileCount - 1)。
template <typename Pipeline>
__aicore__ inline void StudentProcess(Pipeline &pipeline, uint32_t tileCount)
{
    // 单缓冲 starter：结果正确，但 CopyIn、Compute、CopyOut 严格串行。
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        pipeline.CopyIn(tile);
        auto xLocal = pipeline.DeQueX();
        auto yLocal = pipeline.DeQueY();
        pipeline.Compute(xLocal, yLocal);
        pipeline.CopyOut(tile);
    }
}

#pragma once

#include <cstdint>

constexpr int32_t kStudentBufferNum = 2;

template <typename Pipeline>
__aicore__ inline void StudentProcess(Pipeline &pipeline, uint32_t tileCount)
{
    pipeline.CopyIn(0);
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        auto xLocal = pipeline.DeQueX();
        auto yLocal = pipeline.DeQueY();
        if (tile + 1 < tileCount) {
            pipeline.CopyIn(tile + 1);
        }
        if (tile > 0) {
            pipeline.CopyOut(tile - 1);
        }
        pipeline.Compute(xLocal, yLocal);
    }
    pipeline.CopyOut(tileCount - 1);
}

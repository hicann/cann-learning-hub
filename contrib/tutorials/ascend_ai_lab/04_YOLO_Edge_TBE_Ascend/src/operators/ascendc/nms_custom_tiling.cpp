// Teaching skeleton for a YOLO NMS custom operator on Ascend C.
//
// This file is intentionally compact. A production operator normally requires:
//   1. op proto / shape inference registration
//   2. tiling data definition
//   3. host tiling implementation
//   4. device kernel implementation
//   5. ACL/PyACL or framework binding
//
// Inputs:
//   boxes:  [num_boxes, 4]  float32, xyxy
//   scores: [num_boxes]     float32
// Outputs:
//   keep:   [max_output]    int32
//   count:  [1]             int32

#include <cstdint>

struct YoloNmsTilingData {
    uint32_t numBoxes;
    uint32_t maxOutput;
    float iouThreshold;
    uint32_t blockLength;
};

static inline uint32_t CeilDiv(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

extern "C" int TilingYoloNmsCustom(
    uint32_t numBoxes,
    uint32_t maxOutput,
    float iouThreshold,
    YoloNmsTilingData *tiling)
{
    if (tiling == nullptr || numBoxes == 0 || maxOutput == 0) {
        return -1;
    }
    tiling->numBoxes = numBoxes;
    tiling->maxOutput = maxOutput;
    tiling->iouThreshold = iouThreshold;
    tiling->blockLength = CeilDiv(numBoxes, 32) * 32;
    return 0;
}

// Pseudo kernel body. Replace with real Ascend C kernel code using GlobalTensor,
// LocalTensor, DataCopy, vector compare, and workspace bitmask reduction.
extern "C" void YoloNmsCustomKernel(
    const float *boxes,
    const float *scores,
    int32_t *keep,
    int32_t *count,
    const YoloNmsTilingData *tiling)
{
    if (boxes == nullptr || scores == nullptr || keep == nullptr || count == nullptr || tiling == nullptr) {
        return;
    }

    // Teaching placeholder:
    // The production implementation should:
    //   1. sort candidates by score or consume pre-sorted candidates
    //   2. compute IoU vector blocks on AI Core
    //   3. suppress boxes whose IoU exceeds threshold
    //   4. write kept indices and final count to global memory
    count[0] = 0;
}

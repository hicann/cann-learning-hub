#ifndef ATTENTION_CUSTOM_TILING_H
#define ATTENTION_CUSTOM_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct AttentionCustomTilingData {
    uint32_t seqLen;                                          // 序列长度 S
    uint32_t dim;                                             // 注意力头维度 D
    float scale;                                              // 缩放因子 1/sqrt(D)
};

#endif // ATTENTION_CUSTOM_TILING_H

#pragma once

// TODO 1：计算当前 Block 在展平物理数组中的起始元素下标。
__aicore__ inline uint32_t StudentBlockOffset(uint32_t blockFormer, uint32_t blockIdx)
{
    return 0U;
}

// TODO 2：普通 Block 返回 blockFormer，最后一个 Block 返回 blockTail。
// starter 返回 blockTail，能够安全运行，但多 Block 用例会遗漏数据。
__aicore__ inline uint32_t StudentCurrentUnits(
    uint32_t blockIdx,
    uint32_t blockNum,
    uint32_t blockFormer,
    uint32_t blockTail)
{
    return blockTail;
}

// TODO 3：把占位计算 x & x 改为集合交 x & y。
#define STUDENT_COMPUTE(z, x, y, len) \
    ComputeSetAnd((z), (x), (x), (len))

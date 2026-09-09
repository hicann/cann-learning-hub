#pragma once

__aicore__ inline uint32_t StudentBlockOffset(uint32_t blockFormer, uint32_t blockIdx)
{
    return blockFormer * blockIdx;
}

__aicore__ inline uint32_t StudentCurrentUnits(
    uint32_t blockIdx,
    uint32_t blockNum,
    uint32_t blockFormer,
    uint32_t blockTail)
{
    return (blockIdx + 1U == blockNum) ? blockTail : blockFormer;
}

#define STUDENT_COMPUTE(z, x, y, len) \
    ComputeSetAnd((z), (x), (y), (len))

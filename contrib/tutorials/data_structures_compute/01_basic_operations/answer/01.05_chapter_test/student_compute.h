#pragma once

__aicore__ inline uint32_t StudentBlockOffset(uint32_t blockLength, uint32_t blockIdx)
{
    return blockLength * blockIdx;
}

__aicore__ inline uint32_t StudentTileOffset(int32_t progress, uint32_t tileLength)
{
    return static_cast<uint32_t>(progress) * tileLength;
}

#define STUDENT_COMPUTE(z, x, y, len) AscendC::Add((z), (x), (y), (len))

#pragma once

// TODO 1：计算当前核在全局内存中的起始元素下标。
// 提示：每个核连续处理 blockLength 个元素，blockIdx 从 0 开始。
__aicore__ inline uint32_t StudentBlockOffset(uint32_t blockLength, uint32_t blockIdx)
{
    return 0U;
}

// TODO 2：计算当前 Tile 在“本核数据段”中的起始元素下标。
// 提示：progress 是 Tile 序号，tileLength 是单个 Tile 的元素数。
__aicore__ inline uint32_t StudentTileOffset(int32_t progress, uint32_t tileLength)
{
    return 0U;
}

// TODO 3：把占位计算替换为逐元素向量加法 z = x + y。
// 当前占位版本只复制 x，能够编译，但不会通过数值正确性测试。
#define STUDENT_COMPUTE(z, x, y, len) AscendC::Adds((z), (x), 0.0F, (len))

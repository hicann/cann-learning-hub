// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct AsStridedTilingData {
    uint32_t totalOutputElements;   // 输出总元素数 = product(size)
    uint32_t tileSize;              // 每 tile 处理的元素数
    uint32_t inputTotalElements;    // 输入总元素数
    uint32_t ndim;                  // 维度数 (1-4)
    uint32_t size[4];               // 输出各维度大小
    int32_t stride[4];              // 输出各维度步长（可为负）
    int32_t storageOffset;          // 存储偏移量
    uint32_t dimStride[4];          // 各维度累积步长（Host 预计算 offset 时使用）
    uint32_t pathFlag;              // 0=PathA, 1=PathB
    uint32_t blockDim;              // 使用的核数
    uint32_t offsetTableInTiling;   // 1=偏移表追加在tiling data中, 0=kernel内on-the-fly计算
};

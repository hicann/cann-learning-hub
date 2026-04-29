/**
 * Ascend C 算子 Tiling 数据结构模板
 * 
 * 使用方法：
 * 1. 替换 "Op" 为实际算子名
 * 2. 添加需要的 Tiling 参数
 */

#ifndef OP_TILING_H
#define OP_TILING_H
#include <cstdint>

struct OpTilingData {
    // 基础参数
    uint32_t totalLength;    // 总元素数
    uint32_t blockDim;       // 使用的核数
    int32_t dtype;           // 数据类型（用于 kernel 中选择模板）
    
    // TODO: 添加算子特有的参数
    // uint32_t axisLength;
    // uint32_t axisStride;
    // uint32_t outerLength;
    // uint32_t innerLength;
    // int32_t axis;
    
    // 属性参数（从 Host 传递到 Kernel）
    // int32_t exclusive;
    // int32_t reverse;
};

#endif

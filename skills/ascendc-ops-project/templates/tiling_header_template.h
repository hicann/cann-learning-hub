#pragma once

#ifndef ${OPERATOR_NAME_UPPER}_TILING_H
#define ${OPERATOR_NAME_UPPER}_TILING_H

#include <cstdint>

// 根据需要定义常量
constexpr uint32_t MAX_DIM = 8;
constexpr uint32_t ALIGN_NUM = 8;

struct ${OPERATOR_CLASS}TilingData {
    // 核心参数
    uint32_t blockNum;              // 使用的核数
    uint32_t totalElements;         // 输出总元素数
    uint32_t inputTotalElements;    // 输入总元素数（用于边界检查）
    
    // 切分参数
    uint32_t numPerCore;            // 每个核处理的元素数
    uint32_t tailNumLastCore;       // 最后一个核的尾部元素数
    
    // TODO: 添加其他需要的参数
    // 例如：shape、stride、属性参数等
};

#endif

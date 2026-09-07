// ============================================================================
// Tiling 常量和结构体 - kernel 和 host 共用
// ============================================================================
// 此文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字
//
// rope_optimized 是 RoPE 的部分优化版本（v2）:
//   - 单缓冲（bufferNum=1），禁止 Double Buffer
//   - sTile 动态计算，一次处理多行
//   - baseRows/remainRows 负载均衡多核切分
//   - Muls 高维切分 + DataCopy UB→UB 高维切分
// ============================================================================

#pragma once

#include <cstdint>

// dtype 标识：0 = float (FP32), 1 = half (FP16)
constexpr int32_t DTYPE_FLOAT = 0;
constexpr int32_t DTYPE_HALF = 1;

// 单缓冲配置（v2 核心约束：不使用 Double Buffer）
constexpr uint32_t BUFFER_NUM = 1;

// UB 可用容量常量（字节），DAV_2201 的 TPipe 可用 UB = TMP_UB_OFFSET = 184 KB
// （物理总容量 TOTAL_UB_SIZE = 192 KB，末尾 8KB 预留给 KFC 通信消息）
constexpr int64_t UB_SIZE = 184 * 1024;

// Tiling 数据结构 - 向核函数传递的运行时参数
// 参考 DESIGN.md §6
struct RopeOptimizedTilingData {
    int64_t B;              // Batch size
    int64_t S;              // Sequence length
    int64_t H;              // Head num
    int64_t D;              // Head dim (必须为偶数)
    int64_t BH;             // B * H

    int32_t coreNum;        // 实际使用的核数
    int32_t baseRows;       // 基础行数（每核处理）
    int32_t remainRows;     // 余数行数（前 remainRows 个核多处理 1 行）
    int32_t blockNum;       // 实际 block 数

    int32_t sTile;          // S 维度的 UB 切分大小（一次处理 sTile 行）
    int32_t sLoop;          // S 维度循环次数
    int32_t sTail;          // S 维度尾部大小

    int32_t dtype;          // 0 = FP32 (float), 1 = FP16 (half) — 与 rope_naive 一致
};

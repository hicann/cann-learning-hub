/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT
 * LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the program repository for details regarding the License.
 */

#ifndef MOE_ROUTER_FUSED_TILING_H
#define MOE_ROUTER_FUSED_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// moe_router_fused 的 Tiling 数据（host 侧 TilingFunc 填充，kernel 侧消费）
//
// 设计参考同级章节 08_engineering_deployment_and_perf_analysis 的 attention_custom：
// 纯标量 tiling 字段，不内嵌任何高阶库 tiling 结构（无 TCubeTiling / TopkTiling），
// host 侧不做 MultiCoreMatmulTiling / TopKTilingFunc 计算，kernel 不使用 workspace。
struct MoeRouterFusedTilingData {
    uint32_t N;            // token 数（输出行数）
    uint32_t D;            // 隐层维度（x 列数 / w_gate 行数）
    uint32_t E;            // w_gate 第二维（含 padding 列，<= 32，决定 GM 行距）
    uint32_t realE;        // 真实专家数（<= E，[realE, E) 列不参与 softmax/topk）
    uint32_t K;            // top-K
    uint32_t blockDim;     // 参与计算的核数（按 token 行连续块切分）
    uint32_t rowsPerCore;  // 每核行数（对齐到 64B cache line 边界；尾核承担剩余行）
};
#endif  // MOE_ROUTER_FUSED_TILING_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT
 * LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the program repository for details regarding the License.
 */

#include <algorithm>
#include <cstdio>
#include "../op_kernel/moe_router_fused_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

// 整数最大公约数（host 库编译标准无 std::gcd，自实现）
static inline uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// TilingFunc：纯标量 tiling（参考 08 章 attention_custom 的简化风格）
// 不使用 MultiCoreMatmulTiling / TopKTilingFunc，不申请 workspace。
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    MoeRouterFusedTilingData *tiling = context->GetTilingData<MoeRouterFusedTilingData>();

    // 输入形状: x [N, D], w_gate [D, E]（E 为含 padding 的列数）
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *wShape = context->GetInputShape(1);
    int32_t N = static_cast<int32_t>(xShape->GetStorageShape().GetDim(0));
    int32_t D = static_cast<int32_t>(xShape->GetStorageShape().GetDim(1));
    int32_t E = static_cast<int32_t>(wShape->GetStorageShape().GetDim(1));
    // 属性 k（第 0 个）、e（第 1 个，真实专家数；-1 表示 e=E）
    const int64_t *kAttr = context->GetAttrs()->GetInt(0);
    int32_t K = (kAttr != nullptr) ? static_cast<int32_t>(*kAttr) : 2;
    const int64_t *eAttr = context->GetAttrs()->GetInt(1);
    int32_t realE = (eAttr != nullptr && *eAttr > 0) ? static_cast<int32_t>(*eAttr) : E;

    // 合法性校验（纯标量实现，无对齐要求）
    // - E <= 32：kernel 内 scores/topk 标量数组上限（kMaxE）
    // - realE <= E：[realE, E) 为 padding 列，不参与 softmax/topk
    if (N < 1 || D < 1 || E < 1 || E > 32 || realE < 1 || realE > E || K < 1 || K > realE) {
        fprintf(stderr, "[tiling] validation failed N=%d D=%d E=%d realE=%d K=%d\n", N, D, E, realE, K);
        return ge::GRAPH_FAILED;
    }

    // 多核执行：按 token 行「连续块」切分。
    // 约束来源（910B 实测）：kernel 标量 GM 写经 L2 缓存（line=64B），多核并发
    // 写同一 line 会非确定性丢失部分写。因此块边界必须落在输出 64B 对齐处：
    // 每行输出字节 = K*4(idx) 与 K*2(wt)，取 R_align = 64/gcd(2K,64) 行，
    // 则任意 R_align 整数倍行区间的 wt/idx 字节数均为 64B 整数倍。
    // N 不足一段时退回单核（此时单核串行，正确性等价于 blockDim=1 实测 PASS）。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = ascendcPlatform.GetCoreNum();
    }
    const uint32_t rAlign = 64 / GcdU32(2u * static_cast<uint32_t>(K), 64u);
    uint32_t blockDim;
    uint32_t rowsPerCore;
    if (static_cast<uint32_t>(N) < rAlign) {
        blockDim = 1;
        rowsPerCore = static_cast<uint32_t>(N);
    } else {
        blockDim = std::min(coreNum, static_cast<uint32_t>(N) / rAlign);
        // 向上取整到 rAlign 倍数，再反推核数：保证任意两核行数差 < rAlign，
        // 避免「前 (blockDim-1) 核取整段、尾核收尾」造成的尾核负载不均
        // （实测 N=1024/E=32 下尾核独占 3.7× 行数，总耗时由尾核决定）
        rowsPerCore = rAlign * ((static_cast<uint32_t>(N) + blockDim * rAlign - 1) / (blockDim * rAlign));
        blockDim = (static_cast<uint32_t>(N) + rowsPerCore - 1) / rowsPerCore;
    }

    tiling->N = static_cast<uint32_t>(N);
    tiling->D = static_cast<uint32_t>(D);
    tiling->E = static_cast<uint32_t>(E);
    tiling->realE = static_cast<uint32_t>(realE);
    tiling->K = static_cast<uint32_t>(K);
    tiling->blockDim = blockDim;
    tiling->rowsPerCore = rowsPerCore;

    context->SetBlockDim(blockDim);

    // workspace：中间结果全部留在 UB，不申请 workspace（声明 LibApi workspace
    // 会使算子被标记为 matmul 库依赖，触发 MIX 编译模式）
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    fprintf(stderr, "[tiling] moe_router_fused N=%d D=%d E=%d realE=%d K=%d blockDim=%u rowsPerCore=%u\n", N, D, E,
            realE, K, blockDim, rowsPerCore);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    int32_t N = static_cast<int32_t>(xShape->GetDim(0));
    const int64_t *kAttr = context->GetAttrs()->GetInt(0);
    int32_t K = (kAttr != nullptr) ? static_cast<int32_t>(*kAttr) : 2;
    gert::Shape *outIdxShape = context->GetOutputShape(0);
    outIdxShape->SetDimNum(2);
    outIdxShape->SetDim(0, N);
    outIdxShape->SetDim(1, K);
    gert::Shape *outWtShape = context->GetOutputShape(1);
    outWtShape->SetDimNum(2);
    outWtShape->SetDim(0, N);
    outWtShape->SetDim(1, K);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // topk_idx: INT32; topk_weights: 与输入一致（FP16）
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MoeRouterFused : public OpDef {
public:
    explicit MoeRouterFused(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_gate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("topk_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("topk_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("k").Int(2).Comment("top-K value");
        this->Attr("e").Int(-1).Comment("real expert count (<= w_gate second dim); -1 means equal to E");

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(MoeRouterFused);
}  // namespace ops

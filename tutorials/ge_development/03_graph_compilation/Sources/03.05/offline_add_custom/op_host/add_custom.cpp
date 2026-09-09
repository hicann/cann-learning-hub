/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "../op_kernel/add_custom_tiling.h"

namespace optiling {
namespace {
constexpr uint32_t kTotalLength = 8U * 1024U;
constexpr uint32_t kBlockDim = 8U;
constexpr uint32_t kTileNum = 8U;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  if ((context == nullptr) || (context->GetInputShape(0U) == nullptr) ||
      (context->GetInputShape(1U) == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  const uint64_t element_count = context->GetInputShape(0U)->GetOriginShape().GetShapeSize();
  const uint64_t second_element_count = context->GetInputShape(1U)->GetOriginShape().GetShapeSize();
  if ((element_count != kTotalLength) || (second_element_count != kTotalLength)) {
    return ge::GRAPH_FAILED;
  }
  auto *tiling = context->GetTilingData<AddCustomTilingData>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling->totalLength = kTotalLength;
  tiling->tileNum = kTileNum;
  context->SetBlockDim(kBlockDim);
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
  if ((context == nullptr) || (context->GetInputShape(0U) == nullptr) ||
      (context->GetOutputShape(0U) == nullptr)) {
    return GRAPH_FAILED;
  }
  *context->GetOutputShape(0U) = *context->GetInputShape(0U);
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
  if (context == nullptr) {
    return GRAPH_FAILED;
  }
  return context->SetOutputDataType(0U, context->GetInputDataType(0U));
}
}  // namespace ge

namespace ops {
class AddCustom : public OpDef {
 public:
  explicit AddCustom(const char *name) : OpDef(name) {
    Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    Input("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    Output("z")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
  }
};

OP_ADD(AddCustom);
}  // namespace ops

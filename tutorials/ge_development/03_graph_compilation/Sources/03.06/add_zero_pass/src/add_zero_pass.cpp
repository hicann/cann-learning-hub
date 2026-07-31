/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "es_all_ops.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

using namespace ge;
using namespace fusion;

namespace {
bool IsTensorValueEqualToZero(const Tensor &tensor) {
  const auto data_type = tensor.GetTensorDesc().GetDataType();
  const auto *data = tensor.GetData();
  if (data == nullptr) {
    return false;
  }

  int64_t shape_size = tensor.GetTensorDesc().GetShape().GetShapeSize();
  if (shape_size <= 0) {
    shape_size = 1;
  }

  switch (data_type) {
    case DT_FLOAT: {
      const auto *ptr = reinterpret_cast<const float *>(data);
      for (int64_t i = 0; i < shape_size; ++i) {
        if (ptr[i] != 0.0F) {
          return false;
        }
      }
      return true;
    }
    case DT_DOUBLE: {
      const auto *ptr = reinterpret_cast<const double *>(data);
      for (int64_t i = 0; i < shape_size; ++i) {
        if (ptr[i] != 0.0) {
          return false;
        }
      }
      return true;
    }
    case DT_INT32: {
      const auto *ptr = reinterpret_cast<const int32_t *>(data);
      for (int64_t i = 0; i < shape_size; ++i) {
        if (ptr[i] != 0) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

void WriteExecutionMarker() {
  const char *marker_path = std::getenv("GE_PASS_MARKER");
  if ((marker_path == nullptr) || (marker_path[0] == '\0')) {
    return;
  }
  std::ofstream marker(marker_path, std::ios::out | std::ios::trunc);
  marker << "NotebookAddZeroPass executed\n";
}
}  // namespace

class NotebookAddZeroPass : public PatternFusionPass {
 protected:
  std::vector<PatternUniqPtr> Patterns() override {
    auto builder = es::EsGraphBuilder("notebook_add_zero_pattern");
    auto input = builder.CreateInput(0);
    auto zero = es::Const(builder);
    auto add = es::Add(input, zero);

    std::vector<PatternUniqPtr> patterns;
    auto graph = builder.BuildAndReset({add});
    patterns.emplace_back(std::make_unique<Pattern>(std::move(*graph)));
    return patterns;
  }

  bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
    for (auto node : match_result->GetMatchedNodes()) {
      AscendString type;
      (void)node.GetType(type);
      if (type == "Const") {
        Tensor value;
        if (node.GetAttr("value", value) != GRAPH_SUCCESS) {
          return false;
        }
        return IsTensorValueEqualToZero(value);
      }
    }
    return false;
  }

  GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
    WriteExecutionMarker();
    std::cout << "[PASS] NotebookAddZeroPass matched Add(x, 0) and replaced it with x" << std::endl;
    auto builder = es::EsGraphBuilder("notebook_add_zero_replacement");
    auto input = builder.CreateInput(0);
    return builder.BuildAndReset({input});
  }
};

REG_FUSION_PASS(NotebookAddZeroPass).Stage(CustomPassStage::kBeforeInferShape);

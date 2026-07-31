/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "add_custom.h"
#include "ge/ge_api.h"
#include "graph.h"
#include "ops_proto_legacy.h"
#include "tensor.h"
#include "types.h"

namespace {
constexpr uint32_t kGraphId = 0U;
constexpr int64_t kDim0 = 8;
constexpr int64_t kDim1 = 1024;
constexpr size_t kElementCount = static_cast<size_t>(kDim0 * kDim1);
constexpr float kInputValue = 1.5F;
constexpr float kExpectedValue = 3.0F;

void PrintGeError(const char *operation) {
  const auto error_message = ge::GEGetErrorMsgV2();
  const char *error_text = error_message.GetString();
  if ((error_text != nullptr) && (error_text[0] != '\0')) {
    std::cerr << operation << " GE error: " << error_text << std::endl;
  }
}

std::unique_ptr<ge::Graph> BuildGraph() {
  const ge::TensorDesc tensor_desc(ge::Shape({kDim0, kDim1}), ge::FORMAT_ND, ge::DT_FLOAT);

  auto data_x = ge::op::Data("data_x").set_attr_index(0);
  data_x.update_input_desc_x(tensor_desc);
  data_x.update_output_desc_y(tensor_desc);

  auto data_y = ge::op::Data("data_y").set_attr_index(1);
  data_y.update_input_desc_x(tensor_desc);
  data_y.update_output_desc_y(tensor_desc);

  auto add = ge::op::AddCustom("add").set_input_x1(data_x).set_input_x2(data_y);
  add.update_input_desc_x1(tensor_desc);
  add.update_input_desc_x2(tensor_desc);
  add.update_output_desc_y(tensor_desc);

  auto graph = std::make_unique<ge::Graph>("NotebookAddCustomGraph");
  graph->SetInputs({data_x, data_y}).SetOutputs({add});
  return graph;
}

bool BuildInputTensor(ge::Tensor &input) {
  const ge::TensorDesc tensor_desc(ge::Shape({kDim0, kDim1}), ge::FORMAT_ND, ge::DT_FLOAT);
  input = ge::Tensor(tensor_desc);
  const std::vector<float> values(kElementCount, kInputValue);
  return input.SetData(reinterpret_cast<const uint8_t *>(values.data()), values.size() * sizeof(float)) ==
         ge::GRAPH_SUCCESS;
}

bool VerifyOutput(const ge::Tensor &output) {
  if (output.GetSize() < kElementCount * sizeof(float)) {
    std::cerr << "Unexpected output size: " << output.GetSize() << std::endl;
    return false;
  }
  const auto *values = reinterpret_cast<const float *>(output.GetData());
  if (values == nullptr) {
    std::cerr << "Output data is null" << std::endl;
    return false;
  }
  for (size_t i = 0U; i < kElementCount; ++i) {
    if (!std::isfinite(values[i]) || std::abs(values[i] - kExpectedValue) > 1e-6F) {
      std::cerr << "Output mismatch at " << i << ": expected " << kExpectedValue << ", got " << values[i]
                << std::endl;
      return false;
    }
  }
  std::cout << "First element of output: " << values[0] << std::endl;
  return true;
}
}  // namespace

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  const std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
  };
  const auto init_ret = ge::GEInitialize(options);
  if (init_ret != ge::SUCCESS) {
    std::cerr << "GEInitialize failed, ret=" << init_ret << std::endl;
    return 1;
  }

  int ret_code = 0;
  {
    ge::Session session(options);
    auto graph = BuildGraph();
    const auto add_graph_ret = session.AddGraph(kGraphId, *graph);
    if (add_graph_ret != ge::SUCCESS) {
      std::cerr << "AddGraph failed, ret=" << add_graph_ret << std::endl;
      PrintGeError("AddGraph");
      ret_code = 1;
    } else {
      ge::Tensor input_x;
      ge::Tensor input_y;
      if (!BuildInputTensor(input_x) || !BuildInputTensor(input_y)) {
        std::cerr << "SetData failed" << std::endl;
        ret_code = 1;
      } else {
        std::vector<ge::Tensor> outputs;
        const auto run_ret = session.RunGraph(kGraphId, {input_x, input_y}, outputs);
        if (run_ret != ge::SUCCESS) {
          std::cerr << "RunGraph failed, ret=" << run_ret << std::endl;
          PrintGeError("RunGraph");
          ret_code = 1;
        } else if ((outputs.size() != 1U) || !VerifyOutput(outputs[0])) {
          ret_code = 1;
        } else {
          std::cout << "[OK] AddCustom completed GE online compilation and NPU numerical validation" << std::endl;
        }
      }
    }
  }

  const auto finalize_ret = ge::GEFinalize();
  if (finalize_ret != ge::SUCCESS) {
    std::cerr << "GEFinalize failed, ret=" << finalize_ret << std::endl;
    return 1;
  }
  return ret_code;
}

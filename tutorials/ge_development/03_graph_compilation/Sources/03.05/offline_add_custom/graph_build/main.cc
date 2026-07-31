/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include "add_custom.h"
#include "graph.h"
#include "ops_proto_legacy.h"
#include "tensor.h"
#include "types.h"

namespace {
constexpr int64_t kDim0 = 8;
constexpr int64_t kDim1 = 1024;
}

int main() {
  const ge::TensorDesc desc(ge::Shape({kDim0, kDim1}), ge::FORMAT_ND, ge::DT_FLOAT);
  auto data = ge::op::Data("data").set_attr_index(0);
  data.update_input_desc_x(desc);
  data.update_output_desc_y(desc);

  auto add1 = ge::op::AddCustom("add1").set_input_x(data).set_input_y(data);
  add1.update_input_desc_x(desc);
  add1.update_input_desc_y(desc);
  add1.update_output_desc_z(desc);
  auto add2 = ge::op::AddCustom("add2").set_input_x(add1).set_input_y(data);
  add2.update_input_desc_x(desc);
  add2.update_input_desc_y(desc);
  add2.update_output_desc_z(desc);

  ge::Graph graph("OfflineAddCustomGraph");
  graph.SetInputs({data}).SetOutputs({add2});
  if (graph.SaveToFile("single_add.air") != ge::GRAPH_SUCCESS) {
    std::cerr << "SaveToFile(single_add.air) failed" << std::endl;
    return 1;
  }
  std::cout << "AIR generated: single_add.air" << std::endl;
  return 0;
}

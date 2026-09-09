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
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "all_ops.h"

namespace {
constexpr int32_t kDeviceId = 0;
constexpr uint32_t kGraphId = 1U;
const std::vector<int64_t> kShape = {2, 3};

std::unique_ptr<ge::Graph> BuildAddGraph() {
  ge::TensorDesc desc(ge::Shape(kShape), ge::FORMAT_ND, ge::DT_INT32);

  auto input_x = ge::op::Data("input_x").set_attr_index(0);
  input_x.update_input_desc_x(desc);
  input_x.update_output_desc_y(desc);

  auto input_y = ge::op::Data("input_y").set_attr_index(1);
  input_y.update_input_desc_x(desc);
  input_y.update_output_desc_y(desc);

  auto add = ge::op::Add("add").set_input_x1(input_x).set_input_x2(input_y);
  add.update_input_desc_x1(desc);
  add.update_input_desc_x2(desc);
  add.update_output_desc_y(desc);

  auto graph = std::make_unique<ge::Graph>("ZeroCopyDeviceAddGraph");
  std::vector<ge::Operator> graph_inputs = {input_x, input_y};
  std::vector<ge::Operator> graph_outputs = {add};
  graph->SetInputs(graph_inputs).SetOutputs(graph_outputs);
  return graph;
}

bool MakeDeviceTensor(const std::vector<int32_t> &host_data, ge::Tensor &tensor) {
  const size_t bytes = host_data.size() * sizeof(int32_t);
  ge::TensorDesc desc(ge::Shape(kShape), ge::FORMAT_ND, ge::DT_INT32);
  desc.SetPlacement(ge::kPlacementDevice);
  tensor = ge::Tensor(desc);

  void *addr = nullptr;
  if (aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS || addr == nullptr) {
    std::cerr << "aclrtMalloc failed" << std::endl;
    return false;
  }
  if (!host_data.empty() &&
      aclrtMemcpy(addr, bytes, host_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    (void)aclrtFree(addr);
    std::cerr << "aclrtMemcpy H2D failed" << std::endl;
    return false;
  }

  auto free_device = [](uint8_t *ptr) {
    if (ptr != nullptr) {
      (void)aclrtFree(static_cast<void *>(ptr));
    }
  };
  if (tensor.SetData(static_cast<uint8_t *>(addr), bytes, free_device) != ge::GRAPH_SUCCESS) {
    (void)aclrtFree(addr);
    std::cerr << "SetData failed" << std::endl;
    return false;
  }
  if (tensor.SetPlacement(ge::kPlacementDevice) != ge::GRAPH_SUCCESS) {
    tensor = ge::Tensor();
    std::cerr << "SetPlacement failed" << std::endl;
    return false;
  }
  return true;
}

bool MakeDeviceOutputTensor(ge::Tensor &tensor, void **device_addr) {
  const size_t element_count = 6U;
  const size_t bytes = element_count * sizeof(int32_t);
  ge::TensorDesc desc(ge::Shape(kShape), ge::FORMAT_ND, ge::DT_INT32);
  desc.SetPlacement(ge::kPlacementDevice);
  tensor = ge::Tensor(desc);

  void *addr = nullptr;
  if (aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS || addr == nullptr) {
    std::cerr << "aclrtMalloc output failed" << std::endl;
    return false;
  }
  auto free_device = [](uint8_t *ptr) {
    if (ptr != nullptr) {
      (void)aclrtFree(static_cast<void *>(ptr));
    }
  };
  if (tensor.SetData(static_cast<uint8_t *>(addr), bytes, free_device) != ge::GRAPH_SUCCESS) {
    (void)aclrtFree(addr);
    std::cerr << "SetData for output failed" << std::endl;
    return false;
  }
  if (tensor.SetPlacement(ge::kPlacementDevice) != ge::GRAPH_SUCCESS) {
    tensor = ge::Tensor();
    std::cerr << "SetPlacement for output failed" << std::endl;
    return false;
  }
  *device_addr = addr;
  return true;
}

int CompileAndRun() {
  auto graph = BuildAddGraph();
  std::map<ge::AscendString, ge::AscendString> session_options;
  ge::Session session(session_options);
  if (session.AddGraph(kGraphId, *graph) != ge::SUCCESS) {
    std::cerr << "Session::AddGraph failed" << std::endl;
    return -1;
  }

  std::cout << "[INFO] 正在在线编译静态 Add 图" << std::endl;
  if (session.CompileGraph(kGraphId) != ge::SUCCESS) {
    std::cerr << "Session::CompileGraph failed" << std::endl;
    (void)session.RemoveGraph(kGraphId);
    return -1;
  }

  if (aclrtSetDevice(kDeviceId) != ACL_SUCCESS) {
    std::cerr << "aclrtSetDevice before stream creation failed" << std::endl;
    (void)session.RemoveGraph(kGraphId);
    return -1;
  }
  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
    std::cerr << "aclrtCreateStream failed" << std::endl;
    (void)session.RemoveGraph(kGraphId);
    return -1;
  }

  const std::vector<int32_t> host_x = {1, 2, 3, 4, 5, 6};
  const std::vector<int32_t> host_y = {10, 20, 30, 40, 50, 60};
  const std::vector<int32_t> expected = {11, 22, 33, 44, 55, 66};
  ge::Tensor input_x;
  ge::Tensor input_y;
  ge::Tensor output;
  void *reserved_output_addr = nullptr;
  int result = -1;

  do {
    if (!MakeDeviceTensor(host_x, input_x) ||
        !MakeDeviceTensor(host_y, input_y) ||
        !MakeDeviceOutputTensor(output, &reserved_output_addr)) {
      break;
    }

    std::vector<ge::Tensor> inputs = {input_x, input_y};
    std::vector<ge::Tensor> outputs = {output};
    std::cout << "[INFO] 正在把预分配的 Device 输入/输出地址交给 GE 异步执行" << std::endl;
    if (aclrtSetDevice(kDeviceId) != ACL_SUCCESS) {
      std::cerr << "aclrtSetDevice before graph execution failed" << std::endl;
      break;
    }
    if (session.RunGraphWithStreamAsync(kGraphId, static_cast<void *>(stream), inputs, outputs) != ge::SUCCESS) {
      std::cerr << "Session::RunGraphWithStreamAsync failed" << std::endl;
      break;
    }
    if (aclrtSetDevice(kDeviceId) != ACL_SUCCESS) {
      std::cerr << "aclrtSetDevice before stream synchronization failed" << std::endl;
      break;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
      std::cerr << "aclrtSynchronizeStream failed" << std::endl;
      break;
    }
    if (outputs.size() != 1U ||
        static_cast<const void *>(outputs[0].GetData()) != static_cast<const void *>(reserved_output_addr)) {
      std::cerr << "GE did not keep the caller-provided output address" << std::endl;
      break;
    }

    std::vector<int32_t> actual(expected.size(), 0);
    const size_t bytes = actual.size() * sizeof(int32_t);
    if (aclrtMemcpy(actual.data(), bytes, outputs[0].GetData(), bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
      std::cerr << "aclrtMemcpy D2H failed" << std::endl;
      break;
    }
    if (actual != expected) {
      std::cerr << "numeric validation failed" << std::endl;
      break;
    }

    std::cout << "预分配输出地址：" << reserved_output_addr << std::endl;
    std::cout << "GE 返回输出地址：" << static_cast<const void *>(outputs[0].GetData()) << std::endl;
    std::cout << "输出：[11, 22, 33, 44, 55, 66]" << std::endl;
    std::cout << "[INFO] 地址一致仅验证输出目的缓冲区；请用 msprof --runtime-api=on 排除 D2D 回退拷贝"
              << std::endl;
    std::cout << "[OK] 调用方预分配的 Device 地址已保留为模型输出缓冲区，NPU 数值校验通过"
              << std::endl;
    result = 0;
  } while (false);

  (void)aclrtSetDevice(kDeviceId);
  (void)aclrtSynchronizeStream(stream);
  (void)aclrtDestroyStream(stream);
  (void)session.RemoveGraph(kGraphId);
  return result;
}
}  // namespace

int main() {
  const std::map<ge::AscendString, ge::AscendString> ge_options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
  };
  if (ge::GEInitialize(ge_options) != ge::SUCCESS) {
    std::cerr << "GEInitialize failed" << std::endl;
    return -1;
  }
  if (aclInit(nullptr) != ACL_SUCCESS) {
    std::cerr << "aclInit failed" << std::endl;
    (void)ge::GEFinalize();
    return -1;
  }
  if (aclrtSetDevice(kDeviceId) != ACL_SUCCESS) {
    std::cerr << "aclrtSetDevice failed" << std::endl;
    (void)ge::GEFinalize();
    (void)aclFinalize();
    return -1;
  }

  int result = CompileAndRun();
  if (ge::GEFinalize() != ge::SUCCESS) {
    std::cerr << "GEFinalize failed" << std::endl;
    result = -1;
  }
  (void)aclrtResetDevice(kDeviceId);
  (void)aclFinalize();
  return result;
}

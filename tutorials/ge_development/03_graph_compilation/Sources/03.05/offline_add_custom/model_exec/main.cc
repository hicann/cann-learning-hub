/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"

namespace {
constexpr int32_t kDeviceId = 0;
constexpr size_t kElementCount = 8U * 1024U;
constexpr float kExpected = 3.0F;

bool Check(const char *name, aclError ret) {
  if (ret != ACL_SUCCESS) {
    std::cerr << name << " failed, ret=" << ret << std::endl;
    return false;
  }
  return true;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model.om>" << std::endl;
    return 1;
  }
  if (!Check("aclInit", aclInit(nullptr)) || !Check("aclrtSetDevice", aclrtSetDevice(kDeviceId))) {
    (void)aclFinalize();
    return 1;
  }

  uint32_t model_id = 0U;
  bool model_loaded = false;
  aclmdlDesc *model_desc = nullptr;
  aclmdlDataset *input_dataset = nullptr;
  aclmdlDataset *output_dataset = nullptr;
  void *input_device = nullptr;
  void *output_device = nullptr;
  aclDataBuffer *input_buffer = nullptr;
  aclDataBuffer *output_buffer = nullptr;
  int result = 1;

  do {
    if (!Check("aclmdlLoadFromFile", aclmdlLoadFromFile(argv[1], &model_id))) break;
    model_loaded = true;
    model_desc = aclmdlCreateDesc();
    if (model_desc == nullptr || !Check("aclmdlGetDesc", aclmdlGetDesc(model_desc, model_id))) break;
    if (aclmdlGetNumInputs(model_desc) != 1U || aclmdlGetNumOutputs(model_desc) != 1U) {
      std::cerr << "The sample expects one input and one output" << std::endl;
      break;
    }

    const size_t input_size = aclmdlGetInputSizeByIndex(model_desc, 0U);
    const size_t output_size = aclmdlGetOutputSizeByIndex(model_desc, 0U);
    if ((input_size != kElementCount * sizeof(float)) || (output_size != input_size)) {
      std::cerr << "Unexpected model tensor size" << std::endl;
      break;
    }
    if (!Check("aclrtMalloc(input)", aclrtMalloc(&input_device, input_size, ACL_MEM_MALLOC_NORMAL_ONLY)) ||
        !Check("aclrtMalloc(output)", aclrtMalloc(&output_device, output_size, ACL_MEM_MALLOC_NORMAL_ONLY))) {
      break;
    }
    const std::vector<float> input_host(kElementCount, 1.0F);
    if (!Check("aclrtMemcpy(H2D)", aclrtMemcpy(input_device, input_size, input_host.data(), input_size,
                                                ACL_MEMCPY_HOST_TO_DEVICE))) {
      break;
    }
    input_dataset = aclmdlCreateDataset();
    output_dataset = aclmdlCreateDataset();
    input_buffer = aclCreateDataBuffer(input_device, input_size);
    output_buffer = aclCreateDataBuffer(output_device, output_size);
    if ((input_dataset == nullptr) || (output_dataset == nullptr) || (input_buffer == nullptr) ||
        (output_buffer == nullptr) || (aclmdlAddDatasetBuffer(input_dataset, input_buffer) != ACL_SUCCESS) ||
        (aclmdlAddDatasetBuffer(output_dataset, output_buffer) != ACL_SUCCESS)) {
      std::cerr << "Failed to build ACL datasets" << std::endl;
      break;
    }
    // aclmdlExecute is synchronous for this small sample; synchronize anyway
    // so the lifetime rule is explicit in the tutorial.
    if (!Check("aclmdlExecute", aclmdlExecute(model_id, input_dataset, output_dataset)) ||
        !Check("aclrtSynchronizeDevice", aclrtSynchronizeDevice())) {
      break;
    }
    std::vector<float> output_host(kElementCount, 0.0F);
    if (!Check("aclrtMemcpy(D2H)", aclrtMemcpy(output_host.data(), output_size, output_device, output_size,
                                                ACL_MEMCPY_DEVICE_TO_HOST))) {
      break;
    }
    for (size_t i = 0U; i < output_host.size(); ++i) {
      if (!std::isfinite(output_host[i]) || std::fabs(output_host[i] - kExpected) > 1e-5F) {
        std::cerr << "Output mismatch at " << i << ": " << output_host[i] << std::endl;
        break;
      }
      if (i + 1U == output_host.size()) {
        std::cout << "First element of output: " << output_host[0] << std::endl;
        std::cout << "[OK] AIR -> ATC -> OM -> ACL offline validation" << std::endl;
        result = 0;
      }
    }
  } while (false);

  if (input_buffer != nullptr) (void)aclDestroyDataBuffer(input_buffer);
  if (output_buffer != nullptr) (void)aclDestroyDataBuffer(output_buffer);
  if (input_dataset != nullptr) (void)aclmdlDestroyDataset(input_dataset);
  if (output_dataset != nullptr) (void)aclmdlDestroyDataset(output_dataset);
  if (input_device != nullptr) (void)aclrtFree(input_device);
  if (output_device != nullptr) (void)aclrtFree(output_device);
  if (model_desc != nullptr) (void)aclmdlDestroyDesc(model_desc);
  if (model_loaded) (void)aclmdlUnload(model_id);
  (void)aclrtResetDevice(kDeviceId);
  (void)aclFinalize();
  return result;
}

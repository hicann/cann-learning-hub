/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits.h>
#include <string>
#include <vector>
#include <unistd.h>

#include "acl/acl_rt.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "graph/custom_op.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr size_t kExpectedElementCount = 8U * 1024U;
constexpr uint32_t kBlockDim = 8U;
constexpr const char *kKernelBinaryFileName = "add_custom_kernel.npubin";

std::string GetKernelBinaryPath() {
  Dl_info info{};
  if ((dladdr(reinterpret_cast<void *>(&GetKernelBinaryPath), &info) == 0) || (info.dli_fname == nullptr)) {
    return {};
  }
  // Resolve an absolute path before GE changes the current working directory.
  // dladdr normally returns an absolute path, but realpath also covers
  // libraries loaded through a relative OPP path.
  char resolved_path[PATH_MAX] = {};
  const char *library_name = realpath(info.dli_fname, resolved_path);
  std::string library_path = library_name == nullptr ? info.dli_fname : library_name;
  if (!library_path.empty() && library_path.front() != '/') {
    char current_dir[PATH_MAX] = {};
    if (getcwd(current_dir, sizeof(current_dir)) != nullptr) {
      library_path = std::string(current_dir) + "/" + library_path;
    }
  }
  if (library_path.empty() || library_path.front() != '/') {
    return {};
  }
  const auto separator = library_path.find_last_of('/');
  if (separator == std::string::npos) {
    return kKernelBinaryFileName;
  }
  return library_path.substr(0U, separator + 1U) + kKernelBinaryFileName;
}

std::vector<char> ReadKernelBinary(const std::string &binary_path) {
  std::ifstream input(binary_path, std::ios::in | std::ios::binary);
  if (!input) {
    return {};
  }
  std::vector<char> binary((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  if (binary.empty() || input.bad()) {
    return {};
  }
  return binary;
}

graphStatus InferShapeForAdd(gert::InferShapeContext *ctx) {
  if ((ctx == nullptr) || (ctx->GetInputShape(0U) == nullptr) || (ctx->GetInputShape(1U) == nullptr) ||
      (ctx->GetOutputShape(0U) == nullptr)) {
    return GRAPH_FAILED;
  }
  const auto *input_x_shape = ctx->GetInputShape(0U);
  const auto *input_y_shape = ctx->GetInputShape(1U);
  if (input_x_shape->GetDimNum() != input_y_shape->GetDimNum()) {
    return GRAPH_FAILED;
  }
  for (size_t i = 0U; i < input_x_shape->GetDimNum(); ++i) {
    if (input_x_shape->GetDim(i) != input_y_shape->GetDim(i)) {
      return GRAPH_FAILED;
    }
  }
  *ctx->GetOutputShape(0U) = *input_x_shape;
  return GRAPH_SUCCESS;
}

graphStatus InferDataTypeForAdd(gert::InferDataTypeContext *ctx) {
  if ((ctx == nullptr) || (ctx->GetInputDataType(0U) != DT_FLOAT) || (ctx->GetInputDataType(1U) != DT_FLOAT)) {
    return GRAPH_FAILED;
  }
  return ctx->SetOutputDataType(0U, DT_FLOAT);
}

bool SameShape(const gert::StorageShape &lhs, const gert::StorageShape &rhs) {
  const auto &lhs_shape = lhs.GetStorageShape();
  const auto &rhs_shape = rhs.GetStorageShape();
  if (lhs_shape.GetDimNum() != rhs_shape.GetDimNum()) {
    return false;
  }
  for (size_t i = 0U; i < lhs_shape.GetDimNum(); ++i) {
    if (lhs_shape.GetDim(i) != rhs_shape.GetDim(i)) {
      return false;
    }
  }
  return true;
}

template <typename Context>
auto MallocOutputTensorCompat(Context *ctx, size_t index, const gert::StorageShape &shape,
                              const gert::StorageFormat &format, DataType data_type, size_t tensor_size, int)
    -> decltype(ctx->MallocOutputTensor(index, shape, format, data_type, tensor_size)) {
  // CANN 9.0: tensor_size is required.
  return ctx->MallocOutputTensor(index, shape, format, data_type, tensor_size);
}

template <typename Context>
auto MallocOutputTensorCompat(Context *ctx, size_t index, const gert::StorageShape &shape,
                              const gert::StorageFormat &format, DataType data_type, size_t, long)
    -> decltype(ctx->MallocOutputTensor(index, shape, format, data_type)) {
  // CANN 9.1: output size is inferred from shape and dtype.
  return ctx->MallocOutputTensor(index, shape, format, data_type);
}

}  // namespace

class AddCustom : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    if (ctx == nullptr) {
      std::cerr << "Execute context is null" << std::endl;
      return GRAPH_FAILED;
    }

    const gert::Tensor *input_x = ctx->GetInputTensor(0U);
    const gert::Tensor *input_y = ctx->GetInputTensor(1U);
    if ((input_x == nullptr) || (input_y == nullptr)) {
      std::cerr << "Input tensor is null" << std::endl;
      return GRAPH_FAILED;
    }
    if ((input_x->GetDataType() != DT_FLOAT) || (input_y->GetDataType() != DT_FLOAT) ||
        !SameShape(input_x->GetShape(), input_y->GetShape())) {
      std::cerr << "This minimal kernel expects two same-shaped DT_FLOAT tensors" << std::endl;
      return GRAPH_FAILED;
    }
    if ((input_x->GetShapeSize() != static_cast<int64_t>(kExpectedElementCount)) ||
        (input_y->GetShapeSize() != static_cast<int64_t>(kExpectedElementCount))) {
      std::cerr << "This minimal kernel expects exactly " << kExpectedElementCount << " float elements" << std::endl;
      return GRAPH_FAILED;
    }
    constexpr size_t kExpectedTensorBytes = kExpectedElementCount * sizeof(float);
    if ((input_x->GetAddr() == nullptr) || (input_y->GetAddr() == nullptr) ||
        (input_x->GetSize() < kExpectedTensorBytes) || (input_y->GetSize() < kExpectedTensorBytes)) {
      std::cerr << "Input tensor address or storage size is invalid" << std::endl;
      return GRAPH_FAILED;
    }

    const auto &output_shape = input_x->GetShape();
    const auto &format = input_x->GetFormat();
    gert::Tensor *output_z = MallocOutputTensorCompat(ctx, 0U, output_shape, format, input_x->GetDataType(),
                                                      input_x->GetSize(), 0);
    if (output_z == nullptr) {
      std::cerr << "MallocOutputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }
    if (output_z->GetAddr() == nullptr || output_z->GetSize() < kExpectedTensorBytes) {
      std::cerr << "Output tensor address or storage size is invalid" << std::endl;
      return GRAPH_FAILED;
    }

    const std::string binary_path = GetKernelBinaryPath();
    if (binary_path.empty()) {
      std::cerr << "Failed to resolve kernel binary path" << std::endl;
      return GRAPH_FAILED;
    }

    // aclrtcGetBinData returns an in-memory Device ELF. CANN's RTC contract
    // loads those bytes with aclrtBinaryLoadFromData, and the magic option
    // identifies the binary as Vector Core ELF.
    const std::vector<char> kernel_binary = ReadKernelBinary(binary_path);
    if (kernel_binary.empty()) {
      std::cerr << "Failed to read kernel binary: " << binary_path << std::endl;
      return GRAPH_FAILED;
    }
    aclrtBinaryLoadOption load_option{};
    load_option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    load_option.value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
    aclrtBinaryLoadOptions load_options{};
    load_options.numOpt = 1U;
    load_options.options = &load_option;

    aclrtBinHandle binary_handle = nullptr;
    aclrtFuncHandle function_handle = nullptr;
    auto unload_binary = [&]() {
      if (binary_handle != nullptr) {
        const aclError unload_ret = aclrtBinaryUnLoad(binary_handle);
        binary_handle = nullptr;
        return unload_ret;
      }
      return ACL_ERROR_NONE;
    };
    const aclError load_ret = aclrtBinaryLoadFromData(kernel_binary.data(), kernel_binary.size(), &load_options,
                                                      &binary_handle);
    if (load_ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError=" << load_ret
                << " while loading RTC binary bytes from " << binary_path << std::endl;
      return GRAPH_FAILED;
    }
    const aclError function_ret = aclrtBinaryGetFunction(binary_handle, "add_custom", &function_handle);
    if (function_ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError=" << function_ret
                << " while resolving add_custom" << std::endl;
      (void)unload_binary();
      return GRAPH_FAILED;
    }

    // Keep the argument layout identical to the three-pointer kernel ABI.
    struct __attribute__((packed)) KernelArgs {
      const void *x __attribute__((aligned(8)));
      const void *y __attribute__((aligned(8)));
      void *z __attribute__((aligned(8)));
    } args{input_x->GetAddr(), input_y->GetAddr(), output_z->GetAddr()};

    const aclError launch_ret = aclrtLaunchKernelWithHostArgs(
        function_handle, kBlockDim, ctx->GetStream(), nullptr, static_cast<void *>(&args), sizeof(args), nullptr, 0U);
    if (launch_ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError=" << launch_ret
                << " while launching add_custom" << std::endl;
      (void)unload_binary();
      return GRAPH_FAILED;
    }

    // The launch is asynchronous.  Unloading the binary immediately after
    // enqueueing it races with the device and can corrupt the next graph node;
    // synchronize this tiny teaching kernel before releasing its handle.
    const aclError sync_ret = aclrtSynchronizeStream(ctx->GetStream());
    if (sync_ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError=" << sync_ret
                << " while synchronizing add_custom" << std::endl;
      (void)unload_binary();
      return GRAPH_FAILED;
    }
    const aclError unload_ret = unload_binary();
    if (unload_ret != ACL_ERROR_NONE) {
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError=" << unload_ret
                << " while unloading add_custom" << std::endl;
      return GRAPH_FAILED;
    }

    std::cout << "AddCustom launched, binary=" << binary_path << ", elements=" << kExpectedElementCount << std::endl;
    return GRAPH_SUCCESS;
  }
};

IMPL_OP(AddCustom).InferShape(InferShapeForAdd).InferDataType(InferDataTypeForAdd);
REG_AUTO_MAPPING_OP(AddCustom);

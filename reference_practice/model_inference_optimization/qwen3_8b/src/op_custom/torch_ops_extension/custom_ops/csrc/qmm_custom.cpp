/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// 工具函数：推导输出 shape 和 dtype
at::Tensor construct_qmm_custom_output_tensor(
    const at::Tensor& x1, const at::Tensor& x2,
    const c10::optional<at::Tensor>& pertoken_scale, int64_t dtype)
{
    auto M = x1.size(0);
    auto N = x2.size(1);

    at::ScalarType output_dtype;
    if (pertoken_scale.has_value() && pertoken_scale.value().defined()) {
        output_dtype = at::ScalarType::BFloat16;
    } else {
        output_dtype = at::ScalarType::Int;
    }

    at::Tensor result = at::empty({M, N}, x1.options().dtype(output_dtype));
    return result;
}

// 工具函数：将 at::Tensor 转为指定 format 的 aclTensor
// ops_common.h 的 ConvertType 只支持单参数（自动推导 format），
// 对于 x2 这种需要 ACL_FORMAT_FRACTAL_NZ 的场景，需手动构造 aclTensor
inline aclTensor* ConvertTypeWithFormat(const at::Tensor& at_tensor, aclFormat format)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }
    if (!at_tensor.defined()) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type =
        kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(
        acl_data_type != ACL_DT_UNDEFINED,
        std::string(c10::toString(scalar_data_type)) + " has not been supported")

    // NZ 格式需要计算 storage dims
    c10::SmallVector<int64_t, SIZE> storageDims;
    if (format == ACL_FORMAT_FRACTAL_NZ) {
        // NZ format: storage shape is [ceil(N/16), ceil(M/16), 16, 16]
        // 对于 2D [M, K] 的 weight tensor，NZ storage shape:
        // [ceil(K/32), ceil(M/16), 16, 32]
        auto M = at_tensor.size(0);
        auto K = at_tensor.size(1);
        storageDims.push_back((K + 31) / 32);
        storageDims.push_back((M + 15) / 16);
        storageDims.push_back(16);
        storageDims.push_back(32);
    } else {
        TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }

    auto acl_tensor = aclCreateTensor(
        at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
        at_tensor.strides().data(), at_tensor.storage_offset(), format,
        storageDims.data(), storageDims.size(),
        const_cast<void*>(at_tensor.storage().data()));
    return acl_tensor;
}

// step2: 为NPU设备实现前向接口
at::Tensor qmm_custom_npu(const at::Tensor& x1, const at::Tensor& x2,
                           const at::Tensor& scale,
                           const c10::optional<at::Tensor>& pertoken_scale,
                           int64_t dtype)
{
    at::Tensor result = construct_qmm_custom_output_tensor(x1, x2, pertoken_scale, dtype);
    at::Tensor tmp = at::empty({x1.size(0), x2.size(1)}, x1.options().dtype(at::ScalarType::Int));

    // 创建 aclTensor 参数
    // x2 使用 ConvertTypeWithFormat 显式指定 ACL_FORMAT_FRACTAL_NZ，其余使用 ND 格式
    aclTensor* x1_acl = ConvertType(x1);
    aclTensor* x2_acl = ConvertTypeWithFormat(x2, ACL_FORMAT_FRACTAL_NZ);
    aclTensor* scale_acl = ConvertType(scale);
    aclTensor* tmp_acl = ConvertType(tmp);
    aclTensor* pertoken_acl = ConvertType(pertoken_scale);
    aclTensor* result_acl = ConvertType(result);

    // 获取 aclnn 函数地址
    static const auto getWorkspaceSizeFuncAddr =
        GetOpApiFuncAddr("aclnnQmmCustomGetWorkspaceSize");
    static const auto opApiFuncAddr = GetOpApiFuncAddr("aclnnQmmCustom");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnQmmCustom or aclnnQmmCustomGetWorkspaceSize not found in libopapi.so.");

    // 获取 NPU stream
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    // 调用 GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    typedef int (*GetWorkspaceSizeFunc)(
        aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*,
        aclTensor*, uint64_t*, aclOpExecutor**);
    auto getWorkspaceSizeFunc =
        reinterpret_cast<GetWorkspaceSizeFunc>(getWorkspaceSizeFuncAddr);
    auto workspace_status = getWorkspaceSizeFunc(
        x1_acl, x2_acl, scale_acl, tmp_acl, pertoken_acl,
        result_acl, &workspace_size, &executor);
    TORCH_CHECK(workspace_status == 0,
                "call aclnnQmmCustomGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());

    // 分配 workspace
    void* workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size != 0) {
        at::TensorOptions options =
            at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kByte);
        workspace_tensor = at::empty({workspace_size}, options);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    // 执行算子
    typedef int (*OpApiFunc)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);

    auto acl_call = [=]() -> int {
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        TORCH_CHECK(api_ret == 0, "call aclnnQmmCustom failed, detail:", aclGetRecentErrMsg());
        Release(x1_acl);
        Release(x2_acl);
        Release(scale_acl);
        Release(tmp_acl);
        if (pertoken_acl != nullptr) {
            Release(pertoken_acl);
        }
        Release(result_acl);
        return api_ret;
    };

    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnQmmCustom");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();

    return result;
}

// step3: 为META设备实现前向接口（图模式需要）
at::Tensor qmm_custom_meta(const at::Tensor& x1, const at::Tensor& x2,
                             const at::Tensor& scale,
                             const c10::optional<at::Tensor>& pertoken_scale,
                             int64_t dtype)
{
    return construct_qmm_custom_output_tensor(x1, x2, pertoken_scale, dtype);
}

} // namespace custom

// step4: 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("qmm_custom", &custom::qmm_custom_npu);
}

// step5: 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("qmm_custom", &custom::qmm_custom_meta);
}

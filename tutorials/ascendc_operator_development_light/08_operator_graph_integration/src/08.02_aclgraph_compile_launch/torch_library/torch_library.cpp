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
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void add_custom_impl(uint32_t numBlocks, void* aclStream, uint8_t* x, uint8_t* y, uint8_t* z,
                                uint32_t totalLength);

namespace ascendc_ops {
at::Tensor ascendc_add(const at::Tensor& x, const at::Tensor& y)
{
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);
    constexpr uint32_t numBlocks = 8;
    uint32_t totalLength = static_cast<uint32_t>(x.numel());
    add_custom_impl(numBlocks, aclStream, reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
                    reinterpret_cast<uint8_t*>(y.mutable_data_ptr()), reinterpret_cast<uint8_t*>(z.mutable_data_ptr()),
                    totalLength);
    return z;
}
} // namespace ascendc_ops

TORCH_LIBRARY(ascendc_ops, m) { m.def("ascendc_add(Tensor x, Tensor y) -> Tensor"); }

TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m) { m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add)); }

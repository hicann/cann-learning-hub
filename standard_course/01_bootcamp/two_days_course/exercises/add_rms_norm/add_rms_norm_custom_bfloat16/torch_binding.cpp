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
#include <limits>
#include <vector>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void add_rms_norm_custom_impl(
    void *aclStream,
    uint8_t *x1, uint8_t *x2, uint8_t *gamma,
    uint8_t *y, uint8_t *rstd, uint8_t *x,
    uint32_t m, uint32_t n, float epsilon);

namespace ascendc_ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> ascendc_add_rms_norm_custom(
    const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &gamma, double epsilon)
{
    TORCH_CHECK(x1.scalar_type() == at::kBFloat16 && x2.scalar_type() == at::kBFloat16 &&
                    gamma.scalar_type() == at::kBFloat16,
                "x1/x2/gamma must be bfloat16");
    TORCH_CHECK(x1.dim() >= 2 && x2.dim() == x1.dim(),
                "x1/x2 must be at least 2D and have the same rank");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == x1.size(-1),
                "gamma must be 1D [N] matching the last dim of x1");

    const int64_t n = x1.size(-1);
    const int64_t m = x1.numel() / n;
    TORCH_CHECK(n > 0, "last dim must be non-empty");
    TORCH_CHECK(n <= 4096,
                "N must be <= 4096 for whole-row processing (UB capacity limit)");
    TORCH_CHECK(m <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "too many rows to index in uint32_t");

    // Kernel consumes row-major contiguous [M, N] buffers. Materialize
    // contiguous views before passing raw pointers.
    auto x1c = x1.contiguous();
    auto x2c = x2.contiguous();
    auto gammac = gamma.contiguous();

    std::vector<int64_t> rstd_shape(x1c.dim() - 1);
    for (int64_t i = 0; i < x1c.dim() - 1; ++i) {
        rstd_shape[i] = x1c.size(i);
    }

    at::Tensor y = at::empty_like(x1c);                                       // bfloat16
    at::Tensor rstd = at::empty(rstd_shape, x1c.options().dtype(at::kFloat)); // float32
    at::Tensor x = at::empty_like(x1c);                                       // bfloat16

    auto aclStream = c10_npu::getCurrentNPUStream().stream();
    add_rms_norm_custom_impl(
        aclStream,
        reinterpret_cast<uint8_t *>(x1c.data_ptr()),
        reinterpret_cast<uint8_t *>(x2c.data_ptr()),
        reinterpret_cast<uint8_t *>(gammac.data_ptr()),
        reinterpret_cast<uint8_t *>(y.data_ptr()),
        reinterpret_cast<uint8_t *>(rstd.data_ptr()),
        reinterpret_cast<uint8_t *>(x.data_ptr()),
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<float>(epsilon));
    return std::make_tuple(y, rstd, x);
}

}  // namespace ascendc_ops

TORCH_LIBRARY(ascendc_ops, m)
{
    m.def("ascendc_add_rms_norm_custom(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
{
    m.impl("ascendc_add_rms_norm_custom", TORCH_FN(ascendc_ops::ascendc_add_rms_norm_custom));
}

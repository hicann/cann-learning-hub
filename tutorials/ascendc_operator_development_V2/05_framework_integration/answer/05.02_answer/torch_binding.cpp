#include <cstdint>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void sub_custom_impl(uint32_t numBlocks, void* aclStream, uint8_t* x, uint8_t* y, uint8_t* z,
                                uint32_t totalLength);

namespace {
constexpr int64_t SUB_ROWS = 8;
constexpr int64_t SUB_COLS = 2048;
}

namespace practice_sub_ops {
at::Tensor practice_sub(const at::Tensor& x, const at::Tensor& y)
{
    TORCH_CHECK(x.dim() == 2 && x.size(0) == SUB_ROWS && x.size(1) == SUB_COLS,
        "practice_sub only supports lhs shape [8, 2048].");
    TORCH_CHECK(y.dim() == 2 && y.size(0) == SUB_ROWS && y.size(1) == SUB_COLS,
        "practice_sub only supports rhs shape [8, 2048].");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half && y.scalar_type() == at::ScalarType::Half,
        "practice_sub only supports float16 tensors.");
    TORCH_CHECK(x.device() == y.device(), "lhs and rhs must be on the same device.");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "practice_sub only supports contiguous tensors.");

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);

    uint32_t numBlocks = 8;
    uint32_t totalLength = SUB_ROWS * SUB_COLS;
    sub_custom_impl(numBlocks, aclStream, reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()), reinterpret_cast<uint8_t*>(z.mutable_data_ptr()),
        totalLength);
    return z;
}
}  // namespace practice_sub_ops

TORCH_LIBRARY(practice_sub_ops, m)
{
    m.def("practice_sub(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(practice_sub_ops, PrivateUse1, m)
{
    m.impl("practice_sub", TORCH_FN(practice_sub_ops::practice_sub));
}

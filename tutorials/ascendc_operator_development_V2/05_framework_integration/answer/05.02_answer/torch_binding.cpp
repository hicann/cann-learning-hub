#include <cstdint>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void matmul_custom_impl(void* aclStream, uint8_t* a, uint8_t* b, uint8_t* c);

namespace {
constexpr int64_t MATMUL_M = 256;
constexpr int64_t MATMUL_K = 64;
constexpr int64_t MATMUL_N = 256;
}

namespace practice_matmul_ops {
at::Tensor practice_matmul(const at::Tensor& a, const at::Tensor& b)
{
    TORCH_CHECK(a.dim() == 2 && a.size(0) == MATMUL_M && a.size(1) == MATMUL_K,
        "practice_matmul only supports lhs shape [256, 64].");
    TORCH_CHECK(b.dim() == 2 && b.size(0) == MATMUL_K && b.size(1) == MATMUL_N,
        "practice_matmul only supports rhs shape [64, 256].");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half && b.scalar_type() == at::ScalarType::Half,
        "practice_matmul only supports float16 tensors.");
    TORCH_CHECK(a.device() == b.device(), "lhs and rhs must be on the same device.");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "practice_matmul only supports contiguous tensors.");

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor c = at::empty({MATMUL_M, MATMUL_N}, a.options());
    matmul_custom_impl(aclStream, reinterpret_cast<uint8_t*>(a.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(b.mutable_data_ptr()), reinterpret_cast<uint8_t*>(c.mutable_data_ptr()));
    return c;
}
}  // namespace practice_matmul_ops

TORCH_LIBRARY(practice_matmul_ops, m)
{
    m.def("practice_matmul(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(practice_matmul_ops, PrivateUse1, m)
{
    m.impl("practice_matmul", TORCH_FN(practice_matmul_ops::practice_matmul));
}

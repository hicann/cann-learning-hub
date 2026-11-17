#include <cstdint>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void matmul_custom_impl(void* stream, uint8_t* x, uint8_t* bt, uint8_t* y);

namespace {
void check_input(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "Expected an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, "Unexpected dtype");
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == 8192 && tensor.size(1) == 8192,
                "Expected shape (8192, 8192)");
    TORCH_CHECK(tensor.is_contiguous(), "Expected a contiguous tensor");
    TORCH_CHECK(!tensor.requires_grad(), "This exercise supports forward inference only");
}

at::Tensor matmul(const at::Tensor& x, const at::Tensor& bt)
{
    check_input(x);
    check_input(bt);
    TORCH_CHECK(x.device() == bt.device(), "Inputs must be on the same NPU");
    const c10_npu::NPUGuard guard(x.device());
    auto output = at::empty(x.sizes(), x.options());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    matmul_custom_impl(stream, reinterpret_cast<uint8_t*>(x.mutable_data_ptr()), reinterpret_cast<uint8_t*>(bt.mutable_data_ptr()), reinterpret_cast<uint8_t*>(output.mutable_data_ptr()));
    return output;
}
} // namespace

TORCH_LIBRARY(ascendc_matmul, m)
{
    m.def("matmul(Tensor x, Tensor bt) -> Tensor");
}

TORCH_LIBRARY_IMPL(ascendc_matmul, PrivateUse1, m)
{
    m.impl("matmul", TORCH_FN(matmul));
}

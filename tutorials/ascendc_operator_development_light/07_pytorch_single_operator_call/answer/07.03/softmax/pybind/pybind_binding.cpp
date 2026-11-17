#include <cstdint>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void softmax_custom_impl(void* stream, uint8_t* x, uint8_t* y);

namespace {
void check_input(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "Expected an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, "Unexpected dtype");
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == 128 && tensor.size(1) == 128,
                "Expected shape (128, 128)");
    TORCH_CHECK(tensor.is_contiguous(), "Expected a contiguous tensor");
    TORCH_CHECK(!tensor.requires_grad(), "This exercise supports forward inference only");
}

at::Tensor softmax(const at::Tensor& x)
{
    check_input(x);
    const c10_npu::NPUGuard guard(x.device());
    auto output = at::empty(x.sizes(), x.options());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    softmax_custom_impl(stream, reinterpret_cast<uint8_t*>(x.mutable_data_ptr()), reinterpret_cast<uint8_t*>(output.mutable_data_ptr()));
    return output;
}
} // namespace

PYBIND11_MODULE(ascendc_softmax, m)
{
    m.def("softmax", &softmax);
}

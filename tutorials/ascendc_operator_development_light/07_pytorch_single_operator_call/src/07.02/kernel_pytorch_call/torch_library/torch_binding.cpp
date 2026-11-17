#include <cstdint>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void add_custom_impl(uint32_t numBlocks, void* aclStream, uint8_t* x, uint8_t* y, uint8_t* z,
                                uint32_t totalLength);

namespace ascendc_ops {
at::Tensor ascendc_add(const at::Tensor& x, const at::Tensor& y)
{
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty(x.sizes(), x.options());

    uint32_t numBlocks = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }

    add_custom_impl(numBlocks, aclStream, reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
                    reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
                    reinterpret_cast<uint8_t*>(z.mutable_data_ptr()), totalLength);
    return z;
}
}  // namespace ascendc_ops

TORCH_LIBRARY(ascendc_ops, m)
{
    m.def("ascendc_add(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
{
    m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
}

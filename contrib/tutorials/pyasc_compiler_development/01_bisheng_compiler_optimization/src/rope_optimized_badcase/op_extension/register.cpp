#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

// rope_optimized 算子注册
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("rope_optimized(Tensor x, Tensor cos, Tensor sin) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("rope_optimized", TORCH_FN(ascend_kernel::rope_optimized_torch));
}

// Meta 后端实现（torch.compile / fx 需要）
at::Tensor rope_optimized_meta(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin)
{
    // 输出 shape 与 x 相同
    return at::empty_like(x);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("rope_optimized", &rope_optimized_meta);
}

} // namespace

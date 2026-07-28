#include <torch/extension.h>

namespace {
constexpr int64_t ROW_NUM = 8;
constexpr int64_t COL_NUM = 2048;

at::Tensor LaunchAddMeta(const at::Tensor &x, const at::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape.");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == ROW_NUM && x.size(1) == COL_NUM,
        "launch_add only supports shape [8, 2048] in this sample.");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float && y.scalar_type() == at::ScalarType::Float,
        "launch_add only supports float tensors in this sample.");
    return at::empty_like(x);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(ge_launch_samples, m)
{
    m.def("launch_add(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(ge_launch_samples, Meta, m)
{
    m.impl("launch_add", TORCH_FN(LaunchAddMeta));
}

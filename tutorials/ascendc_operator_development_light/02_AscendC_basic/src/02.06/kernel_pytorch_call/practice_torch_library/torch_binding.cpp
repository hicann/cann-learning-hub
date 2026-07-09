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

    // TODO: 获取当前 NPU stream。
    // TODO: 按输入 shape 和 options 创建输出 Tensor。
    // TODO: 调用 sub_custom_impl(...)，传入核数、stream、x/y/z 的 device 指针和 totalLength。
    // TODO: 返回输出 Tensor。
}
}  // namespace practice_sub_ops

// TODO: 使用 TORCH_LIBRARY 定义 practice_sub_ops::practice_sub 的 schema。

// TODO: 使用 TORCH_LIBRARY_IMPL 将 practice_sub 注册到 PrivateUse1。

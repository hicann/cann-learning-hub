#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

// rope_optimized 算子 PyTorch 接口
// x: [B, S, H, D], cos: [S, D], sin: [S, D] -> output: [B, S, H, D]
at::Tensor rope_optimized_torch(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin);

} // namespace ascend_kernel

#endif // OPS_H

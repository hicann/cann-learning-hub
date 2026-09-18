"""PyPTO Sigmoid 激活层 (fwd + bwd) — 7.5 节等用到的激活函数

结构同 PyPTODropoutModule: kernel (vec tile 128,128) + autograd Function
+ nn.Module 封装, 可直接放进 nn.Sequential 与 PyPTOConv2d/PyPTOLinear 混用。

- 前向: y = σ(x) = 1 / (1 + e^(-x)), kernel 内 ``pypto.sigmoid``。
- 反向: grad_x = y · (1-y) · grad_y, 由前向保存的 y 重算 (同 09/10 章
  ``make_pypto_sigmoid``, 无需要额外保存 x)。
- 任意维度输入, 内部展平为 2D 后调用 kernel, 输出 reshape 回原形状。

限制: FP32; 输入需连续 (函数内 ``.contiguous()`` 防御); 无参数, 无 dropout
式的 training/eval 区分 (sigmoid 与 torch.nn.Sigmoid 语义一致)。
"""

import torch
from torch import nn

import pypto


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sigmoid_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    b.move(pypto.sigmoid(a))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sigmoid_bwd_kernel(
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    # grad_x = y · (1-y) · grad_y
    one_minus_b = pypto.neg(pypto.sub(b, 1.0))
    g_mul_b = pypto.mul(grad_b, b)
    grad_a.move(pypto.mul(g_mul_b, one_minus_b))


class PyPTOSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        orig_shape = a.shape
        a_2d = a.reshape(-1, a.shape[-1]).contiguous()
        b_2d = torch.empty_like(a_2d)
        sigmoid_fwd_kernel(a_2d, b_2d)
        b = b_2d.reshape(orig_shape)
        ctx.save_for_backward(b)  # 反向由 y 重算, 无需保存 x
        return b

    @staticmethod
    def backward(ctx, grad_b):
        (b,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None
        b_2d = b.reshape(-1, b.shape[-1]).contiguous()
        grad_b_2d = grad_b.reshape(-1, b.shape[-1]).contiguous()
        grad_a_2d = torch.empty_like(b_2d)
        sigmoid_bwd_kernel(b_2d, grad_b_2d, grad_a_2d)
        return grad_a_2d.reshape(b.shape)


class PyPTOSigmoid(nn.Module):
    """NPU 上的 Sigmoid 激活层, 等价于 torch.nn.Sigmoid。"""

    def forward(self, x):
        return PyPTOSigmoidFunction.apply(x)

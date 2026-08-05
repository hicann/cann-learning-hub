"""PyPTO NPU 算子库（第 8 章共享模块）。

本模块把第 8 章后续各节需要用到的 pypto Kernel 与
``torch.autograd.Function`` 封装收集到一处，供本章后续各节通过
``from src.pypto_ops import ...`` 直接复用，避免每节都把同一份 Kernel 代码
原样复制一遍。

提供的算子：

| 算子 | 前向 PyPTO API | 反向 PyPTO API | tiling |
|------|---------------|----------------|--------|
| matmul | pypto.matmul | pypto.matmul (transpose variants) | cube [16,16] x3 |
| bias_add | pypto.add (broadcast) | sum over batch dim | vec (128,128) |
| add | pypto.add (same shape) | identity | vec (128,128) |
| tanh | pypto.exp/div/sub | (1-y²)*grad | vec (128,128) |
| relu | pypto.maximum | pypto.where | vec (128,128) |
| softmax+CE | pypto.softmax + gather/log | softmax - one_hot | vec (8, aligned) |

每个算子采用工厂函数模式：
1. 定义 @pypto.frontend.jit 内核 (fwd + bwd)
2. 包装为 torch.autograd.Function
3. 导出为 PyPTO<Op> 类，通过 .apply() 调用
4. 对有需要的算子提供 nn.Module 封装

导出清单：
- PyPTOMatmul / PyPTOBiasAdd / PyPTOAdd / PyPTOTanh / PyPTOReLUOp
- PyPTOReLU / PyPTOLinear (nn.Module 封装)
- loss_fn (softmax + cross-entropy 损失入口)
"""

import pypto
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# matmul：前向 C = A @ B，反向 grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    c.move(pypto.matmul(a, b, pypto.DT_FP32))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
    grad_a.move(
        pypto.matmul(grad_c, b, pypto.DT_FP32, b_trans=True))
    grad_b.move(
        pypto.matmul(a, grad_c, pypto.DT_FP32, a_trans=True))


def make_pypto_matmul(fwd_kernel, bwd_kernel):
    """创建 PyPTO 矩阵乘法算子。"""

    class PyPTOMatmulImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty(a.shape[0], b.shape[1],
                            device=a.device, dtype=a.dtype)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            a, b = ctx.saved_tensors
            need_a = ctx.needs_input_grad[0]
            need_b = ctx.needs_input_grad[1]
            grad_a = torch.empty_like(a) if need_a else None
            grad_b = torch.empty_like(b) if need_b else None
            tmp_a = grad_a if need_a else torch.empty_like(a)
            tmp_b = grad_b if need_b else torch.empty_like(b)
            bwd_kernel(a.contiguous(), b.contiguous(),
                       grad_c.contiguous(), tmp_a, tmp_b)
            return grad_a, grad_b

    return PyPTOMatmulImpl


# ---------------------------------------------------------------------------
# bias_add：MxN + N → MxN，广播
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bias_add_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.add(a, b))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bias_add_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_b.move(pypto.sum(grad_c, 0))


def make_pypto_bias_add(fwd_kernel, bwd_kernel):
    """创建 PyPTO 偏置加法算子 (MxN + N -> MxN)。"""

    class PyPTOBiasAddImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            a, _ = ctx.saved_tensors
            need_grad_a = ctx.needs_input_grad[0]
            need_grad_b = ctx.needs_input_grad[1]
            grad_a = grad_c if need_grad_a else None
            grad_b = torch.empty(a.shape[1], device=a.device,
                                 dtype=a.dtype) if need_grad_b else None
            if need_grad_b:
                bwd_kernel(a.contiguous(), grad_c.contiguous(), grad_b)
            return grad_a, grad_b

    return PyPTOBiasAddImpl


# ---------------------------------------------------------------------------
# add：MxN + MxN → MxN，同形状逐元素加法（反向恒等映射，无需 NPU 内核）
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.add(a, b))


def make_pypto_add(fwd_kernel):
    """创建 PyPTO 同形状逐元素加法算子 (MxN + MxN -> MxN)。
       add(a,b) 的反向：grad_a = grad_c, grad_b = grad_c。"""

    class PyPTOAddImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            grad_a = grad_c if ctx.needs_input_grad[0] else None
            grad_b = grad_c if ctx.needs_input_grad[1] else None
            return grad_a, grad_b

    return PyPTOAddImpl


# ---------------------------------------------------------------------------
# tanh：tanh(x) = 2·sigmoid(2x) - 1
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tanh_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    two_a = pypto.mul(a, 2.0)
    s = pypto.sigmoid(two_a)
    two_s = pypto.mul(s, 2.0)
    b.move(pypto.sub(two_s, 1.0))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tanh_bwd_kernel(
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    b_sq = pypto.mul(b, b)
    one_minus_b2 = pypto.neg(pypto.sub(b_sq, 1.0))
    grad_a.move(pypto.mul(grad_b, one_minus_b2))


def make_pypto_tanh(fwd_kernel, bwd_kernel):
    """创建 PyPTO tanh 激活算子。"""

    class PyPTOTanhImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            b = torch.empty_like(a)
            fwd_kernel(a, b)
            ctx.save_for_backward(b)
            return b

        @staticmethod
        def backward(ctx, grad_b):
            (b,) = ctx.saved_tensors
            grad_a = torch.empty_like(b) if ctx.needs_input_grad[0] else None
            if grad_a is not None:
                bwd_kernel(b.contiguous(), grad_b.contiguous(), grad_a)
            return grad_a

    return PyPTOTanhImpl


# ---------------------------------------------------------------------------
# relu：y = max(x, 0)，反向梯度 = grad_out if y > 0 else 0
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def relu_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    y.move(pypto.maximum(x, 0.0))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def relu_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    grad_in: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_in.move(pypto.where(y > 0.0, grad_out, 0.0))


def make_pypto_relu(fwd_kernel, bwd_kernel):
    """创建 PyPTO ReLU 激活算子。"""

    class PyPTOReLUOpImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            y = torch.empty_like(x)
            fwd_kernel(x, y)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            (y,) = ctx.saved_tensors
            grad_in = torch.empty_like(y) if ctx.needs_input_grad[0] else None
            if grad_in is not None:
                bwd_kernel(y.contiguous(), grad_out.contiguous(), grad_in)
            return grad_in

    return PyPTOReLUOpImpl


# ---------------------------------------------------------------------------
# softmax + cross_entropy：组合损失函数
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    y.move(pypto.softmax(x, dim=-1))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_y: pypto.Tensor([], pypto.DT_FP32),
    grad_x: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    diag = pypto.mul(y, grad_y)
    s = pypto.sum(diag, 1, True)
    grad_x.move(pypto.sub(diag, pypto.mul(y, s)))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def cross_entropy_fwd_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    indices: pypto.Tensor([], pypto.DT_INT32),
    out: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    gathered = pypto.gather(y_hat, 1, pypto.reshape(indices, (-1, 1)))
    gathered = pypto.reshape(gathered, (-1,))
    out.move(pypto.neg(pypto.log(pypto.maximum(gathered, 1e-12))))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def cross_entropy_bwd_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    indices: pypto.Tensor([], pypto.DT_INT32),
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    grad_y_hat: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    one_hot_vec = pypto.one_hot(indices, ((num_classes + 7) // 8) * 8)
    one_hot_vec = pypto.cast(one_hot_vec, pypto.DT_FP32)
    neg_grad = pypto.mul(pypto.reshape(grad_out, (-1, 1)), one_hot_vec)
    grad_y_hat.move(pypto.div(neg_grad, y_hat))


class PyPTOSoftmaxCrossEntropyLossFunction(torch.autograd.Function):
    """组合 Softmax + CrossEntropy 损失的自定义 autograd 函数。"""

    @staticmethod
    def forward(ctx, x, y, num_classes):
        ctx.num_classes = num_classes

        softmax_out = torch.empty_like(x)
        softmax_fwd_kernel(x, softmax_out, num_classes)

        y_i32 = y.to(torch.int32).reshape(-1)
        ce_out = torch.empty(y_i32.shape[0], device=x.device, dtype=x.dtype)
        cross_entropy_fwd_kernel(softmax_out, y_i32, ce_out, num_classes)

        ctx.save_for_backward(softmax_out, y_i32)
        return ce_out.view(-1, 1)

    @staticmethod
    def backward(ctx, grad_out):
        softmax_out, y_i32 = ctx.saved_tensors
        num_classes = ctx.num_classes

        one_hot = F.one_hot(
            y_i32.long(), num_classes).to(dtype=softmax_out.dtype)
        grad_x = grad_out * (softmax_out - one_hot)

        return grad_x, None, None


# =========================================================================
# 实例化所有算子（工厂函数调用）
# =========================================================================

PyPTOMatmul = make_pypto_matmul(matmul_fwd_kernel, matmul_bwd_kernel)
PyPTOBiasAdd = make_pypto_bias_add(bias_add_fwd_kernel, bias_add_bwd_kernel)
PyPTOAdd = make_pypto_add(add_fwd_kernel)
PyPTOTanh = make_pypto_tanh(tanh_fwd_kernel, tanh_bwd_kernel)
PyPTOReLUOp = make_pypto_relu(relu_fwd_kernel, relu_bwd_kernel)
PyPTOSoftmaxCrossEntropyLoss = PyPTOSoftmaxCrossEntropyLossFunction


# =========================================================================
# nn.Module 封装
# =========================================================================

class PyPTOLinear(nn.Module):
    """用 PyPTO 算子实现的线性层 (与 nn.Linear 接口兼容)。"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        y = PyPTOMatmul.apply(x, self.weight)
        if self.bias is not None:
            y = PyPTOBiasAdd.apply(y, self.bias)
        return y


class PyPTOReLU(nn.Module):
    """用 PyPTO ReLU 算子实现的 ReLU 激活层。"""

    def forward(self, x):
        return PyPTOReLUOp.apply(x)


class PyPTORNN(nn.Module):
    """PyPTO RNN 隐层（单层单向）。

    仅做隐状态递推：h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)。
    不含输出投影，与 nn.RNN 的隐藏层接口兼容，
    搭配 PyPTOLinear 即可组成完整语言模型。
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.bidirectional = False
        self.W_xh = nn.Parameter(
            torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, X, state=None):
        """前向传播。

        Args:
            X: 输入，shape (seq_len, batch, input_size)
            state: 初始隐状态 (1, batch, hidden_size) 或 None

        Returns:
            output: (seq_len, batch, hidden_size) 各时间步隐状态
            h_n: (1, batch, hidden_size) 最终隐状态
        """
        batch_size = X.shape[1]
        if state is None:
            h_flat = torch.zeros(batch_size, self.hidden_size,
                                 device=X.device, dtype=X.dtype)
        else:
            h_flat = state.squeeze(0)

        outputs = []
        for t in range(X.shape[0]):
            xw = PyPTOMatmul.apply(X[t], self.W_xh)
            hw = PyPTOMatmul.apply(h_flat, self.W_hh)
            summed = PyPTOAdd.apply(xw, hw)
            biased = PyPTOBiasAdd.apply(summed, self.b_h)
            h_flat = PyPTOTanh.apply(biased)
            outputs.append(h_flat)

        Y = torch.stack(outputs, dim=0)
        h_n = h_flat.unsqueeze(0)
        return Y, h_n


# =========================================================================
# 便利函数
# =========================================================================

def loss_fn(logits, y, num_classes=10):
    """计算 softmax cross-entropy 损失。

    Args:
        logits: shape [N, num_classes]
        y: shape [N] 整数标签
        num_classes: 类别数

    Returns:
        标量损失
    """
    return PyPTOSoftmaxCrossEntropyLossFunction.apply(
        logits, y, num_classes).mean()


__all__ = [
    "PyPTOMatmul", "PyPTOBiasAdd", "PyPTOAdd",
    "PyPTOTanh", "PyPTOReLUOp", "PyPTOSoftmaxCrossEntropyLoss",
    "PyPTOLinear", "PyPTOReLU", "PyPTORNN",
    "loss_fn",
]

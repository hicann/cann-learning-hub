"""PyPTO NPU 算子库（第 9 章共享模块）。

本模块继承第 8 章的算子（matmul / bias_add / add / tanh / relu / softmax+CE /
PyPTOLinear / PyPTORNN），并新增第 9 章所需的算子：sigmoid / mul / sub /
PyPTOGRU / PyPTOLSTM。

提供的算子：

| 算子 | 前向 PyPTO API | 反向 | tiling |
|------|---------------|------|--------|
| matmul | pypto.matmul | transpose variants | cube [16,16] x3 |
| bias_add | pypto.add (broadcast) | sum over batch dim | vec (128,128) |
| add | pypto.add (same shape) | identity | vec (128,128) |
| tanh | pypto.sigmoid 组合 2σ(2x)-1 | (1-y²)*grad | vec (128,128) |
| relu | pypto.maximum | pypto.where | vec (128,128) |
| softmax+CE | pypto.softmax + gather/log | softmax - one_hot | vec (8, aligned) |
| sigmoid | pypto.sigmoid | y*(1-y)*grad | vec (128,128) |
| mul | pypto.mul | swap operands | vec (128,128) |
| sub | pypto.sub | identity / neg | vec (128,128) |

每个算子采用工厂函数模式：
1. 定义 @pypto.frontend.jit 内核 (fwd + bwd)
2. 包装为 torch.autograd.Function
3. 导出为 PyPTO<Op> 类，通过 .apply() 调用
4. 对有需要的算子提供 nn.Module 封装

导出清单：
- PyPTOMatmul / PyPTOBiasAdd / PyPTOAdd / PyPTOTanh / PyPTOReLUOp
- PyPTOSigmoid / PyPTOMul / PyPTOSub
- PyPTOReLU / PyPTOLinear / PyPTORNN / PyPTOGRU / PyPTOLSTM
- loss_fn / PyPTOSoftmaxCrossEntropyLoss
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
# sigmoid：y = 1 / (1 + e^(-x))，反向：grad_x = y · (1-y) · grad_y
# ---------------------------------------------------------------------------

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
    one_minus_b = pypto.neg(pypto.sub(b, 1.0))
    g_mul_b = pypto.mul(grad_b, b)
    grad_a.move(pypto.mul(g_mul_b, one_minus_b))


def make_pypto_sigmoid(fwd_kernel, bwd_kernel):
    """创建 PyPTO sigmoid 激活算子。"""

    class PyPTOSigmoidImpl(torch.autograd.Function):
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

    return PyPTOSigmoidImpl


# ---------------------------------------------------------------------------
# mul：逐元素乘法 c = a * b，反向：grad_a = grad_c * b, grad_b = grad_c * a
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def mul_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.mul(a, b))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def mul_bwd_kernel(
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_a.move(pypto.mul(grad_c, b))
    grad_b.move(pypto.mul(grad_c, a))


def make_pypto_mul(fwd_kernel, bwd_kernel):
    """创建 PyPTO 逐元素乘法算子 c = a * b。"""

    class PyPTOMulImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            a, b = ctx.saved_tensors
            need_grad_a = ctx.needs_input_grad[0]
            need_grad_b = ctx.needs_input_grad[1]
            grad_a = torch.empty_like(a) if need_grad_a else None
            grad_b = torch.empty_like(b) if need_grad_b else None
            tmp_a = grad_a if need_grad_a else torch.empty_like(a)
            tmp_b = grad_b if need_grad_b else torch.empty_like(b)
            if need_grad_a or need_grad_b:
                bwd_kernel(grad_c.contiguous(), a.contiguous(),
                           b.contiguous(), tmp_a, tmp_b)
            return grad_a, grad_b

    return PyPTOMulImpl


# ---------------------------------------------------------------------------
# sub：逐元素减法 c = a - b，反向：grad_a = grad_c, grad_b = -grad_c
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sub_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    c.move(pypto.sub(a, b))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sub_bwd_grad_b_kernel(
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_b.move(pypto.neg(grad_c))


def make_pypto_sub(fwd_kernel, bwd_grad_b_kernel):
    """创建 PyPTO 逐元素减法算子 c = a - b。

       反向：grad_a = grad_c (恒等), grad_b = -grad_c。
    """

    class PyPTOSubImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            fwd_kernel(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            need_grad_a = ctx.needs_input_grad[0]
            need_grad_b = ctx.needs_input_grad[1]
            grad_a = grad_c if need_grad_a else None
            grad_b = torch.empty_like(grad_c) if need_grad_b else None
            if need_grad_b:
                bwd_grad_b_kernel(grad_c.contiguous(), grad_b)
            return grad_a, grad_b

    return PyPTOSubImpl


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
PyPTOSigmoid = make_pypto_sigmoid(sigmoid_fwd_kernel, sigmoid_bwd_kernel)
PyPTOMul = make_pypto_mul(mul_fwd_kernel, mul_bwd_kernel)
PyPTOSub = make_pypto_sub(sub_fwd_kernel, sub_bwd_grad_b_kernel)
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
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        orig_m = x.shape[0]
        padded = False
        if orig_m % 16 != 0:
            pad_m = 16 - orig_m % 16
            x = F.pad(x, (0, 0, 0, pad_m))
            padded = True
        y = PyPTOMatmul.apply(x, self.weight)
        if self.bias is not None:
            y = PyPTOBiasAdd.apply(y, self.bias)
        if padded:
            y = y[:orig_m]
        if len(orig_shape) > 2:
            y = y.reshape(orig_shape[:-1] + (y.shape[-1],))
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
        orig_batch = X.shape[1]
        batch_size = orig_batch
        if orig_batch % 16 != 0:
            batch_size += 16 - orig_batch % 16
            X = F.pad(X, (0, 0, 0, batch_size - orig_batch, 0, 0))
        H = hidden_size = self.hidden_size
        if state is None:
            h_flat = torch.zeros(batch_size, H,
                                 device=X.device, dtype=X.dtype)
        else:
            h_flat = state.squeeze(0)
            if batch_size != orig_batch:
                h_flat = F.pad(h_flat, (0, 0, 0, batch_size - orig_batch))

        outputs = []
        for t in range(X.shape[0]):
            xw = PyPTOMatmul.apply(X[t], self.W_xh)
            hw = PyPTOMatmul.apply(h_flat, self.W_hh)
            h_flat = PyPTOTanh.apply(PyPTOBiasAdd.apply(PyPTOAdd.apply(xw, hw), self.b_h))
            outputs.append(h_flat)

        Y = torch.stack(outputs, dim=0)
        if batch_size != orig_batch:
            Y = Y[:, :orig_batch]
            h_flat = h_flat[:orig_batch]
        return Y, h_flat.unsqueeze(0)


class PyPTOGRU(nn.Module):
    """PyPTO GRU 隐层（支持多层单向）。

    单层隐状态递推公式：
      R_t = σ(X_t @ W_xr + H_{t-1} @ W_hr + b_r)
      Z_t = σ(X_t @ W_xz + H_{t-1} @ W_hz + b_z)
      H~_t = tanh(X_t @ W_xh + (R_t ⊙ H_{t-1}) @ W_hh + b_h)
      H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ H~_t

    多层时逐层堆叠：第 l 层的输入是第 l-1 层全部时间步的输出，
    层间 dropout 仅在训练时生效（与 nn.GRU 语义一致，不对输出层施加）。
    输出为最后一层全部时间步的隐状态 (seq, batch, hidden)，
    state 为逐层最终隐状态堆叠 (num_layers, batch, hidden)。

    不含输出投影，与 nn.GRU 的隐藏层接口兼容。

    优化：将 X_t 对 (W_xz|W_xr|W_xh) 三次 matmul 合并为一次大 matmul，
    将 H 对 (W_hz|W_hr) 两次 matmul 合并为一次，核函数调用 6→3。
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = False
        # 逐层参数：第 0 层命名不带后缀（W_x 等），后续层带 _l 后缀，
        # 参数名均含 "W_" 前缀，兼容按 "W_" 匹配的 Xavier 初始化逻辑。
        for l in range(num_layers):
            suffix = '' if l == 0 else f'_{l}'
            in_size = input_size if l == 0 else hidden_size
            # 合并 X 权重: [W_xz | W_xr | W_xh] shape (in_size, 3*hidden_size)
            setattr(self, f'W_x{suffix}',
                    nn.Parameter(torch.randn(in_size, hidden_size * 3) * 0.01))
            # 合并 H 权重 (z, r): [W_hz | W_hr] shape (hidden_size, 2*hidden_size)
            setattr(self, f'W_hzr{suffix}',
                    nn.Parameter(torch.randn(hidden_size, hidden_size * 2) * 0.01))
            # 候选隐状态 H 权重（保持独立，因为乘的是 R_t ⊙ H）
            setattr(self, f'W_hh{suffix}',
                    nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01))
            setattr(self, f'b_z{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_r{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_h{suffix}', nn.Parameter(torch.zeros(hidden_size)))

    def _step(self, x_t, H_state, l):
        """单层单时间步递推（层 l）。"""
        suffix = '' if l == 0 else f'_{l}'
        H = self.hidden_size
        W_x = getattr(self, f'W_x{suffix}')
        W_hzr = getattr(self, f'W_hzr{suffix}')
        W_hh = getattr(self, f'W_hh{suffix}')

        # 一次 matmul 计算 X_t @ [W_xz|W_xr|W_xh]
        xw = PyPTOMatmul.apply(x_t, W_x)
        xw_z = xw[:, :H].contiguous()
        xw_r = xw[:, H:2 * H].contiguous()
        xw_h = xw[:, 2 * H:].contiguous()

        # 一次 matmul 计算 H @ [W_hz|W_hr]
        hw_zr = PyPTOMatmul.apply(H_state, W_hzr)
        hw_z = hw_zr[:, :H].contiguous()
        hw_r = hw_zr[:, H:].contiguous()

        Z = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_z, hw_z),
                               getattr(self, f'b_z{suffix}')))
        R = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_r, hw_r),
                               getattr(self, f'b_r{suffix}')))

        # 候选隐状态 (R ⊙ H) @ W_hh
        rh = PyPTOMul.apply(R, H_state)
        rhw = PyPTOMatmul.apply(rh, W_hh)
        H_tilde = PyPTOTanh.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_h, rhw),
                               getattr(self, f'b_h{suffix}')))

        # 最终状态更新
        z_h = PyPTOMul.apply(Z, H_state)
        inv_z_h_tilde = PyPTOMul.apply(
            PyPTOSub.apply(torch.ones_like(Z), Z), H_tilde)
        return PyPTOAdd.apply(z_h, inv_z_h_tilde)

    def forward(self, X, state=None):
        orig_batch = X.shape[1]
        batch_size = orig_batch
        if orig_batch % 16 != 0:
            batch_size += 16 - orig_batch % 16
            X = F.pad(X, (0, 0, 0, batch_size - orig_batch, 0, 0))
        H = hidden_size = self.hidden_size
        num_layers = self.num_layers
        if state is None:
            states = [torch.zeros(batch_size, H,
                                  device=X.device, dtype=X.dtype)
                      for _ in range(num_layers)]
        else:
            states = [state[l] for l in range(num_layers)]
            if batch_size != orig_batch:
                states = [F.pad(s, (0, 0, 0, batch_size - orig_batch))
                          for s in states]

        inputs = X
        Y = None
        final_states = []
        for l in range(num_layers):
            H_state = states[l]
            outputs = []
            for t in range(inputs.shape[0]):
                H_state = self._step(inputs[t], H_state, l)
                outputs.append(H_state)
            layer_out = torch.stack(outputs, dim=0)
            Y = layer_out
            final_states.append(H_state)
            # 层间 dropout：仅训练时、仅在层之间施加（与 nn.GRU 一致）
            if l < num_layers - 1:
                if self.dropout > 0 and self.training:
                    inputs = F.dropout(layer_out, p=self.dropout)
                else:
                    inputs = layer_out

        if batch_size != orig_batch:
            Y = Y[:, :orig_batch]
            final_states = [s[:orig_batch] for s in final_states]
        return Y, torch.stack(final_states, dim=0)


class PyPTOLSTM(nn.Module):
    """PyPTO LSTM 隐层（支持多层单向）。

    单层隐状态递推公式：
      I_t = σ(X_t @ W_xi + H_{t-1} @ W_hi + b_i)
      F_t = σ(X_t @ W_xf + H_{t-1} @ W_hf + b_f)
      O_t = σ(X_t @ W_xo + H_{t-1} @ W_ho + b_o)
      C~_t = tanh(X_t @ W_xc + H_{t-1} @ W_hc + b_c)
      C_t = F_t ⊙ C_{t-1} + I_t ⊙ C~_t
      H_t = O_t ⊙ tanh(C_t)

    多层时逐层堆叠：第 l 层的输入是第 l-1 层全部时间步的输出，
    层间 dropout 仅在训练时生效（与 nn.LSTM 语义一致，不对输出层施加）。
    输出为最后一层全部时间步的隐状态 (seq, batch, hidden)，
    state 为 (H, C) 二元组，H/C 各为逐层最终状态堆叠 (num_layers, batch, hidden)。

    不含输出投影，与 nn.LSTM 的隐藏层接口兼容。

    优化：将 4 次 X matmul (W_xi|W_xf|W_xo|W_xc) 合并为 1 次大 matmul，
    将 4 次 H matmul (W_hi|W_hf|W_ho|W_hc) 合并为 1 次，核函数调用 8→2。
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = False
        # 逐层参数：第 0 层命名不带后缀（W_x 等），后续层带 _l 后缀，
        # 参数名均含 "W_" 前缀，兼容按 "W_" 匹配的 Xavier 初始化逻辑。
        for l in range(num_layers):
            suffix = '' if l == 0 else f'_{l}'
            in_size = input_size if l == 0 else hidden_size
            # 合并 X 权重: [W_xi | W_xf | W_xo | W_xc] shape (in_size, 4*hidden_size)
            setattr(self, f'W_x{suffix}',
                    nn.Parameter(torch.randn(in_size, hidden_size * 4) * 0.01))
            # 合并 H 权重: [W_hi | W_hf | W_ho | W_hc] shape (hidden_size, 4*hidden_size)
            setattr(self, f'W_h{suffix}',
                    nn.Parameter(torch.randn(hidden_size, hidden_size * 4) * 0.01))
            setattr(self, f'b_i{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_f{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_o{suffix}', nn.Parameter(torch.zeros(hidden_size)))
            setattr(self, f'b_c{suffix}', nn.Parameter(torch.zeros(hidden_size)))

    def _step(self, x_t, H_state, C, l):
        """单层单时间步递推（层 l）。"""
        suffix = '' if l == 0 else f'_{l}'
        H = self.hidden_size
        W_x = getattr(self, f'W_x{suffix}')
        W_h = getattr(self, f'W_h{suffix}')

        # 一次 matmul：X_t @ [W_xi|W_xf|W_xo|W_xc]
        xw = PyPTOMatmul.apply(x_t, W_x)
        xw_i = xw[:, :H].contiguous()
        xw_f = xw[:, H:2 * H].contiguous()
        xw_o = xw[:, 2 * H:3 * H].contiguous()
        xw_c = xw[:, 3 * H:].contiguous()

        # 一次 matmul：H @ [W_hi|W_hf|W_ho|W_hc]
        hw = PyPTOMatmul.apply(H_state, W_h)
        hw_i = hw[:, :H].contiguous()
        hw_f = hw[:, H:2 * H].contiguous()
        hw_o = hw[:, 2 * H:3 * H].contiguous()
        hw_c = hw[:, 3 * H:].contiguous()

        I = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_i, hw_i),
                               getattr(self, f'b_i{suffix}')))
        F = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_f, hw_f),
                               getattr(self, f'b_f{suffix}')))
        O = PyPTOSigmoid.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_o, hw_o),
                               getattr(self, f'b_o{suffix}')))
        C_tilde = PyPTOTanh.apply(
            PyPTOBiasAdd.apply(PyPTOAdd.apply(xw_c, hw_c),
                               getattr(self, f'b_c{suffix}')))

        f_c = PyPTOMul.apply(F, C)
        i_c_tilde = PyPTOMul.apply(I, C_tilde)
        C = PyPTOAdd.apply(f_c, i_c_tilde)

        H_state = PyPTOMul.apply(O, PyPTOTanh.apply(C))
        return H_state, C

    def forward(self, X, state=None):
        orig_batch = X.shape[1]
        batch_size = orig_batch
        if orig_batch % 16 != 0:
            batch_size += 16 - orig_batch % 16
            X = F.pad(X, (0, 0, 0, batch_size - orig_batch, 0, 0))
        num_layers = self.num_layers
        if state is None:
            H_states = [torch.zeros(batch_size, self.hidden_size,
                                    device=X.device, dtype=X.dtype)
                        for _ in range(num_layers)]
            C_states = [torch.zeros(batch_size, self.hidden_size,
                                    device=X.device, dtype=X.dtype)
                        for _ in range(num_layers)]
        else:
            H_stack, C_stack = state
            if batch_size != orig_batch:
                H_stack = F.pad(H_stack, (0, 0, 0, batch_size - orig_batch, 0, 0))
                C_stack = F.pad(C_stack, (0, 0, 0, batch_size - orig_batch, 0, 0))
            H_states = [H_stack[l] for l in range(num_layers)]
            C_states = [C_stack[l] for l in range(num_layers)]

        inputs = X
        final_H, final_C = [], []
        for l in range(num_layers):
            H_state, C = H_states[l], C_states[l]
            outputs = []
            for t in range(inputs.shape[0]):
                H_state, C = self._step(inputs[t], H_state, C, l)
                outputs.append(H_state)
            layer_out = torch.stack(outputs, dim=0)
            final_H.append(H_state)
            final_C.append(C)
            # 层间 dropout：仅训练时、仅在层之间施加（与 nn.LSTM 一致）
            if l < num_layers - 1:
                if self.dropout > 0 and self.training:
                    inputs = F.dropout(layer_out, p=self.dropout)
                else:
                    inputs = layer_out

        if batch_size != orig_batch:
            layer_out = layer_out[:, :orig_batch]
            final_H = [h[:orig_batch] for h in final_H]
            final_C = [c[:orig_batch] for c in final_C]
        return layer_out, (torch.stack(final_H, dim=0),
                           torch.stack(final_C, dim=0))


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
    "PyPTOTanh", "PyPTOReLUOp",
    "PyPTOSigmoid", "PyPTOMul", "PyPTOSub",
    "PyPTOSoftmaxCrossEntropyLoss",
    "PyPTOLinear", "PyPTOReLU",
    "PyPTORNN", "PyPTOGRU", "PyPTOLSTM",
    "loss_fn",
]

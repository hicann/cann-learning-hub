"""PyPTO NPU 算子库（第 8 章共享模块）。

本模块把第 8 章后续各节需要用到的 pypto Kernel 与
``torch.autograd.Function`` 封装收集到一处，供本章后续各节通过
``from src.pypto_ops import ...`` 直接复用，避免每节都把同一份 Kernel 代码
原样复制一遍。

提供的算子：

| 算子 | 前向 PyPTO API | 反向 PyPTO API | tiling |
|------|---------------|----------------|--------|
| matmul | pypto.matmul | pypto.matmul (transpose variants) | cube [32,32] x3 |
| bias_add | pypto.add (broadcast) | sum over batch dim | vec (128,128) |
| add | pypto.add (same shape) | identity | vec (128,128) |
| tanh | pypto.tanh (v0.2.1+ 原生) | (1-y²)*grad | vec (128,64) |
| relu | pypto.maximum | pypto.where | vec (128,128) |
| softmax+CE | pypto.softmax + gather/log | softmax - one_hot | vec (rows 自适应, aligned) |
| linear 融合 | pypto.matmul + add | matmul_bwd + sum | cube+vec (0.2.1+) |
| rnn_hidden 融合 | tanh(add(add(a,b),bias)) | 两路梯度 + sum | vec (128,64) (0.2.1+) |

每个算子采用工厂函数模式：
1. 定义 @pypto.frontend.jit 内核 (fwd + bwd)
2. 包装为 torch.autograd.Function
3. 导出为 PyPTO<Op> 类，通过 .apply() 调用
4. 对有需要的算子提供 nn.Module 封装

导出清单：
- PyPTOMatmul / PyPTOBiasAdd / PyPTOAdd / PyPTOTanh / PyPTOReLUOp
- PyPTOLinearFused / PyPTORNNHidden（融合算子，0.2.1+ 启用）
- PyPTOReLU / PyPTOLinear (nn.Module 封装)
- loss_fn (softmax + cross-entropy 损失入口)
"""

import importlib.metadata

import pypto
import torch
from torch import nn
from torch.nn import functional as F

try:
    _PYPTO_VERSION = tuple(
        int(x) for x in importlib.metadata.version("pypto").split(".")[:3])
except Exception:
    _PYPTO_VERSION = (0, 2, 0)
# pypto 0.2.1+ 提供原生 pypto.tanh，且 matmul+bias 融合路径数值稳定
# （0.2.0 下该融合存在调用序列相关的数值不稳定，见第 10 章 KNOWN_ISSUES）
_PYPTO_GE_021 = _PYPTO_VERSION >= (0, 2, 1)


# ---------------------------------------------------------------------------
# matmul：前向 C = A @ B，反向 grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    # cube tile 16→32/64（第 4 章实测：tile 过小导致切分次数过多，-45%）
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    c.move(pypto.matmul(a, b, pypto.DT_FP32))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
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
# linear 融合版：c = matmul(a, w) + bias，单 kernel（少一次 kernel 启动）。
#    实测 (640,512)x(512,512) 从分两次的 0.32ms 降到 0.14ms（约 2.3x）。
#    注意：pypto 0.2.0 下该融合存在调用序列相关的数值不稳定
#    （详见第 10 章 KNOWN_ISSUES 问题 D），仅在 0.2.1+ 启用。
#    反向复用 matmul_bwd_kernel + bias_add_bwd_kernel。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def linear_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(64, 128)
    c.move(pypto.add(pypto.matmul(a, w, pypto.DT_FP32), bias))


def make_pypto_linear_fused(fwd_kernel, matmul_bwd, bias_bwd):
    """创建融合 matmul+bias 的线性层算子（y = x @ W + b）。"""

    class PyPTOLinearFusedImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias):
            ctx.save_for_backward(x, weight)
            c = torch.empty(x.shape[0], weight.shape[1],
                            device=x.device, dtype=x.dtype)
            fwd_kernel(x, weight, bias, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            x, weight = ctx.saved_tensors
            need_x = ctx.needs_input_grad[0]
            need_w = ctx.needs_input_grad[1]
            need_b = ctx.needs_input_grad[2]
            grad_x = torch.empty_like(x) if need_x else None
            grad_w = torch.empty_like(weight) if need_w else None
            if need_x or need_w:
                tmp_x = grad_x if need_x else torch.empty_like(x)
                tmp_w = grad_w if need_w else torch.empty_like(weight)
                matmul_bwd(x.contiguous(), weight.contiguous(),
                           grad_c.contiguous(), tmp_x, tmp_w)
            grad_b = None
            if need_b:
                grad_b = torch.empty(weight.shape[1], device=x.device,
                                     dtype=x.dtype)
                bias_bwd(grad_c.contiguous(), grad_c.contiguous(), grad_b)
            return grad_x, grad_w, grad_b

    return PyPTOLinearFusedImpl


# ---------------------------------------------------------------------------
# rnn_hidden 融合版：c = tanh(a + b + bias)，单 kernel（替代 add+bias_add+tanh
#    三次启动）。反向：grad_a = grad_b = grad_c·(1-y²)（逐元素），
#    grad_bias = sum 行归约。
#    依赖 pypto.tanh（0.2.1+），仅在 0.2.1+ 启用融合路径。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rnn_hidden_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    # pypto.tanh 内部临时 workspace 约 4 倍 tile 数据量，行 tile 收窄为 64
    pypto.set_vec_tile_shapes(128, 64)
    summed = pypto.add(a, b)
    biased = pypto.add(summed, bias)
    c.move(pypto.tanh(biased))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rnn_hidden_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_bias: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 64)
    # move 会消费中间张量（move 后原 tensor 为 nullptr），
    # 三个输出各计算一份 dy = grad_c·(1-y²)，避免复用
    one_minus_y2 = pypto.neg(pypto.sub(pypto.mul(y, y), 1.0))
    grad_a.move(pypto.mul(grad_c, one_minus_y2))
    one_minus_y2_2 = pypto.neg(pypto.sub(pypto.mul(y, y), 1.0))
    grad_b.move(pypto.mul(grad_c, one_minus_y2_2))
    one_minus_y2_3 = pypto.neg(pypto.sub(pypto.mul(y, y), 1.0))
    grad_bias.move(pypto.sum(pypto.mul(grad_c, one_minus_y2_3), 0))


def make_pypto_rnn_hidden(fwd_kernel, bwd_kernel):
    """创建 RNN 隐层融合算子：c = tanh(a + b + bias)。"""

    class PyPTORNNHiddenImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, bias):
            c = torch.empty_like(a)
            fwd_kernel(a, b, bias, c)
            ctx.save_for_backward(c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            (c,) = ctx.saved_tensors
            need_a = ctx.needs_input_grad[0]
            need_b = ctx.needs_input_grad[1]
            need_bias = ctx.needs_input_grad[2]
            grad_a = torch.empty_like(c) if need_a else None
            grad_b = torch.empty_like(c) if need_b else None
            grad_bias = torch.empty(c.shape[1], device=c.device,
                                    dtype=c.dtype) if need_bias else None
            tmp_a = grad_a if need_a else torch.empty_like(c)
            tmp_b = grad_b if need_b else torch.empty_like(c)
            tmp_bias = grad_bias if need_bias else torch.empty(
                c.shape[1], device=c.device, dtype=c.dtype)
            bwd_kernel(c.contiguous(), grad_c.contiguous(),
                       tmp_a, tmp_b, tmp_bias)
            return grad_a, grad_b, grad_bias

    return PyPTORNNHiddenImpl


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
# tanh：原生 pypto.tanh（v0.2.1+），早期版本用 2·sigmoid(2x)-1 组合
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tanh_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    # pypto.tanh 内部临时 workspace 约 4 倍 tile 数据量，
    # 行 tile 收窄为 64 控制 UB 占用（同第 4 章实测结论）
    pypto.set_vec_tile_shapes(128, 64)
    b.move(pypto.tanh(a))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def tanh_bwd_kernel(
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 64)
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
    # 行 tile 8→128（第 4/10 章实测 4.9x）；UB 护栏（0.2.0/0.2.1 实测）：
    # softmax 内部 DIV 约需 2×rows×cols×4B，
    # cols ≤ 160 → 128 行，cols ≤ 320 → 64 行，否则退回 8 行
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
    y.move(pypto.softmax(x, dim=-1))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def cross_entropy_fwd_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    indices: pypto.Tensor([], pypto.DT_INT32),
    out: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
    gathered = pypto.gather(y_hat, 1, pypto.reshape(indices, (-1, 1)))
    gathered = pypto.reshape(gathered, (-1,))
    out.move(pypto.neg(pypto.log(pypto.maximum(gathered, 1e-12))))


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
# 融合算子（0.2.1+ 启用：原生 tanh + 数值稳定；0.2.0 回退分解路径）
PyPTOLinearFused = make_pypto_linear_fused(
    linear_fwd_kernel, matmul_bwd_kernel, bias_add_bwd_kernel)
PyPTORNNHidden = make_pypto_rnn_hidden(
    rnn_hidden_fwd_kernel, rnn_hidden_bwd_kernel)


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
        if _PYPTO_GE_021:
            # 融合路径：matmul+bias 单 kernel（少一次启动，第 10 章实测约 2.3x）
            return PyPTOLinearFused.apply(x, self.weight, self.bias)
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
            if _PYPTO_GE_021:
                # 融合路径（0.2.1+，原生 pypto.tanh）：
                # matmul×2 + tanh(add(add(xw, hw), b_h)) = 每步 3 次启动
                xw = PyPTOMatmul.apply(X[t], self.W_xh)
                hw = PyPTOMatmul.apply(h_flat, self.W_hh)
                h_flat = PyPTORNNHidden.apply(xw, hw, self.b_h)
            else:
                # 0.2.0 分解路径：matmul×2 + add + bias_add + tanh = 每步 5 次启动
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
    "PyPTOLinearFused", "PyPTORNNHidden",
    "PyPTOLinear", "PyPTOReLU", "PyPTORNN",
    "loss_fn",
]

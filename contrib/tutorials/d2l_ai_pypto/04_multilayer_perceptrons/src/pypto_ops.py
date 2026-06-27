"""PyPTO NPU 算子库（第 4 章共享模块）。

本模块把 [4.2 节](./04.02_mlp_scratch.ipynb) 中逐行讲解过的 pypto Kernel 与
`torch.autograd.Function` 封装收集到一处，供本章后续各节（4.3 起）通过
``from src.pypto_ops import ...`` 直接复用，避免每节都把同一份 Kernel 代码
原样复制一遍。

导出清单见 ``__all__``。其中三个 ``autograd.Function`` 子类
（``PyPTOMatmul`` / ``PyPTOBiasAdd`` / ``PyPTOReLUOp``）可被 ``nn.Module``
进一步封装；``loss_fn`` 是带 ``.mean()`` 聚合的 softmax+交叉熵损失入口。
"""

import pypto
import torch
from torch import nn


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
    grad_a.move(pypto.matmul(grad_c, b, pypto.DT_FP32, b_trans=True))
    grad_b.move(pypto.matmul(a, grad_c, pypto.DT_FP32, a_trans=True))


# ---------------------------------------------------------------------------
# bias_add：前向 Y = X + b（逐行广播），反向 grad_b = sum(grad_y, dim=0)
# ---------------------------------------------------------------------------
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bias_add_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    y.move(x + b)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bias_add_bwd_kernel(
    grad_y: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_b.move(pypto.sum(grad_y, 0, True))


# ---------------------------------------------------------------------------
# relu：前向 y = max(x, 0)，反向 grad_in = grad_out * (y > 0)
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


# ---------------------------------------------------------------------------
# softmax：前向 out = softmax(x, dim=-1)
#           反向 grad_x = y * (grad_out - sum(grad_out * y, dim=-1, keepdim=True))
# ---------------------------------------------------------------------------
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_forward_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(16, 16)
    out.move(pypto.softmax(x, dim=-1))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_backward_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_x: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(16, 16)
    grad_x.move(y * (grad_out - pypto.sum(grad_out * y, dim=-1, keepdim=True)))


# ---------------------------------------------------------------------------
# cross_entropy：前向 out = -log(gather(y_hat, labels))（逐样本，reduction='none'）
#                 反向 grad_y_hat = -(grad_out / selected) * one_hot(y)
# ---------------------------------------------------------------------------
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def cross_entropy_forward_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    y_1d: pypto.Tensor([], pypto.DT_INT32),
    out: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    y_2d = pypto.reshape(y_1d, [-1, 1])
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    selected = pypto.gather(y_hat, -1, y_2d)
    # 裁剪到一个极小正数，避免 softmax 下溢出 0 时 log(0) = -inf 传播为 NaN 梯度
    out.move(pypto.neg(pypto.log(pypto.maximum(selected, 1e-12))))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def cross_entropy_backward_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    y_1d: pypto.Tensor([], pypto.DT_INT32),
    grad_y_hat: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    y_2d = pypto.reshape(y_1d, [-1, 1])
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    selected = pypto.gather(y_hat, -1, y_2d)
    # one_hot 要求 tile 末维严格等于 num_classes（TiledOneHot 约束），不可对齐到 8/16 的倍数
    pypto.set_vec_tile_shapes(8, num_classes)
    one_hot_y = pypto.cast(pypto.one_hot(y_1d, num_classes), pypto.DT_FP32)
    pypto.set_vec_tile_shapes(8, 8)
    grad_y_hat.move(-(grad_out / selected) * one_hot_y)


# ---------------------------------------------------------------------------
# 工厂函数：把 (fwd_kernel, bwd_kernel) 包装成 torch.autograd.Function 子类
# ---------------------------------------------------------------------------
def make_pypto_matmul(fwd, bwd):
    class Impl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
            fwd(a, b, c)
            return c

        @staticmethod
        def backward(ctx, grad_c):
            a, b = ctx.saved_tensors
            grad_a, grad_b = torch.empty_like(a), torch.empty_like(b)
            # kernel 要求连续内存：autograd 传入的 grad_c 可能非连续
            bwd(a.contiguous(), b.contiguous(), grad_c.contiguous(), grad_a, grad_b)
            return grad_a, grad_b

    return Impl


def make_pypto_bias_add(fwd, bwd):
    class Impl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, b):
            ctx.save_for_backward(x)
            y = torch.empty_like(x)
            fwd(x, b, y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            grad_x = grad_y
            grad_b_2d = torch.empty(1, grad_y.shape[1], device=grad_y.device, dtype=grad_y.dtype)
            # kernel 要求连续内存：autograd 传入的 grad_y 可能非连续
            bwd(grad_y.contiguous(), grad_b_2d)
            return grad_x, grad_b_2d.squeeze(0)

    return Impl


def make_pypto_relu(fwd, bwd):
    class Impl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            y = torch.empty_like(x)
            fwd(x, y)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            (y,) = ctx.saved_tensors
            grad_in = torch.empty_like(y)
            bwd(y, grad_out, grad_in)
            return grad_in

    return Impl


# ---------------------------------------------------------------------------
# softmax + 交叉熵：前向 PyTorch 调度 + 两个 NPU kernel，反向链式法则手动串联
# ---------------------------------------------------------------------------
class PyPTOSoftmaxCrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, num_classes):
        x_c = x.contiguous()
        # softmax
        softmax_out = torch.empty_like(x_c)
        softmax_forward_kernel(x_c, softmax_out)
        # cross_entropy on softmax output
        y_i32_1d = y.to(torch.int32).contiguous()
        ce_out = torch.empty((x.shape[0], 1), dtype=x.dtype, device=x.device)
        cross_entropy_forward_kernel(softmax_out, y_i32_1d, ce_out, num_classes)
        ctx.save_for_backward(softmax_out, y_i32_1d)
        ctx.num_classes = num_classes
        return ce_out.view(-1)

    @staticmethod
    def backward(ctx, grad_out):
        softmax_out, y_i32_1d = ctx.saved_tensors
        num_classes = ctx.num_classes
        # gradient of cross_entropy w.r.t. softmax_out
        grad_out_c = grad_out.view(-1, 1).contiguous()
        grad_ce = torch.empty_like(softmax_out)
        cross_entropy_backward_kernel(
            grad_out_c, softmax_out, y_i32_1d, grad_ce, num_classes
        )
        # gradient of softmax w.r.t. x
        grad_x = torch.empty_like(softmax_out)
        softmax_backward_kernel(grad_ce.contiguous(), softmax_out, grad_x)
        return grad_x, None, None


def loss_fn(logits, y, num_classes=10):
    return PyPTOSoftmaxCrossEntropyLossFunction.apply(logits, y, num_classes).mean()


# ---------------------------------------------------------------------------
# 实例化算子类
# ---------------------------------------------------------------------------
PyPTOMatmul = make_pypto_matmul(matmul_fwd_kernel, matmul_bwd_kernel)
PyPTOBiasAdd = make_pypto_bias_add(bias_add_fwd_kernel, bias_add_bwd_kernel)
PyPTOReLUOp = make_pypto_relu(relu_fwd_kernel, relu_bwd_kernel)


# ===========================================================================
# 4.4-4.6 新增算子
# ===========================================================================

# ---------------------------------------------------------------------------
# squared_loss：前向 out = (y_hat - y)^2 / 2
#               反向 grad_y_hat = grad_out * (y_hat - y)
#
# 与 [3.3 节](../03_pypto_linear_networks/03.03_linear_regression_scratch.ipynb)
# 中 inline 的平方损失实现功能等价，此处抽取到共享模块供 4.4 / 4.5 节复用。
# ---------------------------------------------------------------------------
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def squared_loss_fwd_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    diff = y_hat - y
    out.move(diff * diff / 2.0)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def squared_loss_bwd_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_y_hat: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_y_hat.move(grad_out * (y_hat - y))


class PyPTOSquaredLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_hat, y):
        y_hat = y_hat.contiguous()
        y_reshaped = y.reshape(y_hat.shape).contiguous()
        out = torch.empty_like(y_hat)
        squared_loss_fwd_kernel(y_hat, y_reshaped, out)
        ctx.save_for_backward(y_hat, y_reshaped)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        y_hat, y = ctx.saved_tensors
        # .sum()/.mean() 的反向会把上游梯度 expand 成非连续张量，kernel 要求连续
        grad_out = grad_out.contiguous()
        grad_y_hat = torch.empty_like(y_hat)
        squared_loss_bwd_kernel(grad_out, y_hat, y, grad_y_hat)
        return grad_y_hat, None


def mse_loss(y_hat, y, reduction='mean'):
    """平方损失，等价于 d2l 的 ``squared_loss``（含 1/2 因子），即 ``0.5 × nn.MSELoss``。

    前向计算 ``(y_hat - y)^2 / 2``（见 ``squared_loss_fwd_kernel``），因此**不**等价于
    ``nn.MSELoss``——后者无 1/2 因子。与 04.05 从零实现复用的 d2l ``squared_loss`` 一致，
    但相对 04.04 / 04.05-简洁 中的 d2l ``nn.MSELoss`` 参照，报出的损失值为其 0.5 倍。

    前向与反向均在 NPU 上通过 pypto kernel 计算。
    """
    out = PyPTOSquaredLossFunction.apply(y_hat, y)
    if reduction == 'mean':
        return out.mean()
    elif reduction == 'sum':
        return out.sum()
    elif reduction == 'none':
        return out
    raise ValueError(f"invalid reduction: {reduction!r}（仅支持 'none' / 'mean' / 'sum'）")


# ---------------------------------------------------------------------------
# nn.Module 用户层封装：从 [4.3 节](./04.03_mlp_concise.ipynb) inline 移植，
# 并扩展 PyPTOLinear 支持 bias 开关（4.4 多项式回归用 bias=False）。
# ---------------------------------------------------------------------------
class PyPTOLinear(nn.Module):
    """用 NPU 矩阵乘 + 偏置加实现的全连接层，接口等价于 torch.nn.Linear。

    参数
    ----
    in_features : int
        输入特征维度
    out_features : int
        输出特征维度
    bias : bool, 默认 True
        是否包含偏置项。False 时偏置被吸收进特征（如多项式回归中 x^0=1）

    注意：``weight`` 形状为 ``[in_features, out_features]``，与
    ``PyPTOMatmul.apply(x, w)`` 的约定一致（``x`` 是 ``[batch, in]``、
    ``w`` 是 ``[in, out]``、输出 ``[batch, out]``），这与 ``torch.nn.Linear``
    的 ``[out_features, in_features]`` 相反。元素级操作（如
    ``nn.init.normal_(m.weight, std=0.01)``、``m.weight.norm()``）不受影响。
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        y = PyPTOMatmul.apply(x, self.weight)
        if self.bias is not None:
            y = PyPTOBiasAdd.apply(y, self.bias)
        return y


class PyPTOReLU(nn.Module):
    """用 NPU ReLU kernel 实现的激活层，等价于 torch.nn.ReLU。"""
    def forward(self, x):
        return PyPTOReLUOp.apply(x)


# ---------------------------------------------------------------------------
# dropout：前向 y = where(uniform > p, x / (1-p), 0)
#          反向 grad_x = where(uniform > p, grad_y / (1-p), 0)
# 注意（均经 NPU 实测）：
#   1. pypto 有 uniform/normal，但为计数器型 RNG（需管理 key/counter）且设备支持
#      受限，故随机数 uniform 由 torch.rand_like 在 NPU 上生成后传入 kernel。
#   2. p 必须在编译期固化为常量（用闭包捕获），不能作为 kernel 形参被 trace——
#      因为 p 要进入 `uniform > p` 与 `x / (1-p)` 作为张量运算操作数，trace 期无法
#      求值 1.0 - p。cross_entropy 的 num_classes:int 先例不适用（它仅用于编译期
#      shape 算术，从不作为张量运算操作数）。
#   3. 用 `pypto.where(uniform > p, ...)`（与本模块 relu 反向同款）实现掩码 + 缩放，
#      而非 `pypto.cast(pypto.gt(uniform, p), FP32)`——后者经 NPU 实测掩码全 0（错误）。
#   4. 单输出 kernel：前向只输出 y、反向只输出 grad_x。掩码不单独输出（双输出
#      kernel 经实测会 segfault），改为保存 uniform、在反向 kernel 内由
#      `uniform > p` 重算掩码。
# ---------------------------------------------------------------------------
def make_dropout_kernels(p: float):
    """每个 p 值编译一份专用 kernel，p 在编译期即为常量。

    返回 (fwd_kernel, bwd_kernel) 元组。调用方应按 p 值缓存对应的
    kernel 对（见 ``get_dropout_kernels``）。
    """
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def dropout_fwd_kernel(
        x: pypto.Tensor([], pypto.DT_FP32),
        uniform: pypto.Tensor([], pypto.DT_FP32),
        y: pypto.Tensor([], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(128, 128)
        # uniform > p 处保留并按 1/(1-p) 缩放，否则置 0；p 为编译期常量
        y.move(pypto.where(uniform > p, x / (1.0 - p), 0.0))

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def dropout_bwd_kernel(
        uniform: pypto.Tensor([], pypto.DT_FP32),
        grad_y: pypto.Tensor([], pypto.DT_FP32),
        grad_x: pypto.Tensor([], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(128, 128)
        # 由 uniform 重算掩码：保留位透传 grad_y / (1-p)，丢弃位置 0
        grad_x.move(pypto.where(uniform > p, grad_y / (1.0 - p), 0.0))

    return dropout_fwd_kernel, dropout_bwd_kernel


# kernel 缓存：按 p 值缓存编译好的 kernel 对，避免重复编译
_dropout_kernel_cache = {}


def get_dropout_kernels(p: float):
    """按 p 值获取（必要时编译并缓存）dropout 的前向/反向 kernel 对。

    返回 (fwd_kernel, bwd_kernel) 元组。
    """
    # 用定点舍入后的 p 作为缓存键，避免浮点精度差异（如 0.1+0.2 != 0.3）造成重复编译
    key = round(p, 6)
    if key not in _dropout_kernel_cache:
        _dropout_kernel_cache[key] = make_dropout_kernels(p)
    return _dropout_kernel_cache[key]


class PyPTODropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        if not (0 <= p <= 1):
            raise ValueError(f"p must be in [0, 1], got {p}")
        if p == 0:
            ctx.mode = 0  # 全保留：前向恒等，反向 grad_x = grad_y
            return x
        if p == 1:
            ctx.mode = 1  # 全丢弃：前向置 0，反向 grad_x = 0
            return torch.zeros_like(x)
        ctx.mode = None
        fwd_kernel, bwd_kernel = get_dropout_kernels(p)
        uniform = torch.rand_like(x)
        y = torch.empty_like(x)
        fwd_kernel(x, uniform, y)
        ctx.save_for_backward(uniform)  # 反向 kernel 由 uniform 重算掩码
        ctx.bwd_kernel = bwd_kernel
        return y

    @staticmethod
    def backward(ctx, grad_y):
        if ctx.mode == 0:
            return grad_y, None
        if ctx.mode == 1:
            return torch.zeros_like(grad_y), None
        (uniform,) = ctx.saved_tensors
        grad_y = grad_y.contiguous()  # autograd 传入的 grad 可能非连续，kernel 要求连续
        grad_x = torch.empty_like(grad_y)
        ctx.bwd_kernel(uniform.contiguous(), grad_y, grad_x)
        return grad_x, None


class PyPTODropout(nn.Module):
    """NPU 上的 Dropout 层，等价于 torch.nn.Dropout。

    参数
    ----
    p : float, 默认 0.5
        暂退概率。前向时以概率 p 将元素置零，其余元素乘以 1/(1-p)
        保持期望不变。``self.training`` 为 False（``net.eval()``）时直接返回 x。
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        return PyPTODropoutFunction.apply(x, self.p)


__all__ = [
    "PyPTOMatmul",
    "PyPTOBiasAdd",
    "PyPTOReLUOp",
    "PyPTOSoftmaxCrossEntropyLossFunction",
    "loss_fn",
    "matmul_fwd_kernel",
    "matmul_bwd_kernel",
    "bias_add_fwd_kernel",
    "bias_add_bwd_kernel",
    "relu_fwd_kernel",
    "relu_bwd_kernel",
    "softmax_forward_kernel",
    "softmax_backward_kernel",
    "cross_entropy_forward_kernel",
    "cross_entropy_backward_kernel",
    "make_pypto_matmul",
    "make_pypto_bias_add",
    "make_pypto_relu",
    # 新增（4.4-4.6）
    "PyPTOLinear",
    "PyPTOReLU",
    "PyPTOSquaredLossFunction",
    "mse_loss",
    "squared_loss_fwd_kernel",
    "squared_loss_bwd_kernel",
    "PyPTODropoutFunction",
    "PyPTODropout",
    "make_dropout_kernels",
    "get_dropout_kernels",
]

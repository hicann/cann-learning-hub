import torch
from torch import nn
import pypto

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
        # tile 维度必须与输入维度匹配 (4D 特征图用 4D tile, 否则 DIVS 报错)
        pypto.set_vec_tile_shapes(*(8, 8, 8, 8) if len(x.shape) == 4 else (128, 128))
        # uniform > p 处保留并按 1/(1-p) 缩放，否则置 0；p 为编译期常量
        y.move(pypto.where(uniform > p, x / (1.0 - p), 0.0))

    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
    def dropout_bwd_kernel(
        uniform: pypto.Tensor([], pypto.DT_FP32),
        grad_y: pypto.Tensor([], pypto.DT_FP32),
        grad_x: pypto.Tensor([], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(*(8, 8, 8, 8) if len(grad_y.shape) == 4 else (128, 128))
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
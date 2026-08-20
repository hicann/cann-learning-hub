"""PyPTO NPU 算子库（第 10 章共享模块）。

本模块继承第 9 章的算子（matmul / bias_add / add / tanh / relu / sigmoid /
mul / sub / softmax+CE / PyPTOLinear / PyPTORNN / PyPTOGRU / PyPTOLSTM），
并新增第 10 章（注意力机制）所需的算子：bmm / softmax（独立版）/ exp /
layer_norm / fused attention。

提供的算子：

| 算子 | 前向 PyPTO API | 反向 | tiling |
|------|---------------|------|--------|
| matmul | pypto.matmul | transpose variants | cube [32,32] / [64,64] |
| bias_add | pypto.add (broadcast) | sum over batch dim | vec (128,128) |
| add | pypto.add (same shape) | identity | vec (128,128) |
| tanh | pypto.sigmoid 组合 2σ(2x)-1 | (1-y²)*grad | vec (128,128) |
| relu | pypto.maximum | pypto.where | vec (128,128) |
| softmax+CE | pypto.softmax + gather/log | softmax - one_hot | vec (128, aligned) |
| sigmoid | pypto.sigmoid | y*(1-y)*grad | vec (128,128) |
| mul | pypto.mul | swap operands | vec (128,128) |
| sub | pypto.sub | identity / neg | vec (128,128) |
| bmm | pypto.matmul（2D 逐 batch 切片） | transpose variants | loop+view，cube [32,32] / [64,64] |
| softmax | pypto.softmax (dim=-1) | diag - y·sum | vec (128, aligned) |
| exp | pypto.exp | y*grad | vec (128,128) |
| layer_norm | sum/rsqrt 组合 | 标准 LN 反向三式 | vec (8, aligned) |
| fused attention | QK^T+softmax+PV 单 kernel | softmax 梯度三式 | loop+view，cube [32,32] / [64,64] |
| conv2d | pypto.conv（L1/L0/vec tile 显式配置） | unfold/matmul/fold 等价式 | cube，10.8 节图像块嵌入 |

每个算子采用工厂函数模式：
1. 定义 @pypto.frontend.jit 内核 (fwd + bwd)
2. 包装为 torch.autograd.Function
3. 导出为 PYPTO<Op> 类，通过 .apply() 调用
4. 对有需要的算子提供 nn.Module 封装

导出清单：
- PyPTOMatmul / PyPTOBiasAdd / PyPTOAdd / PyPTOTanh / PyPTOReLUOp
- PyPTOSigmoid / PyPTOMul / PyPTOSub
- PyPTOBMM / PyPTOSoftmax / PyPTOExp / PyPTOLayerNorm
- PyPTOFusedAttention
- PyPTOReLU / PyPTOLinear / PyPTORNN / PyPTOGRU / PyPTOLSTM
- PyPTOConv2d（10.8 节图像块嵌入）
- loss_fn / PyPTOSoftmaxCrossEntropyLoss

性能调优说明（2026-08-08，详见 PERF_TUNING_REPORT.md）：
- bmm 由 3D 批量 matmul 改为 2D loop+view 逐 batch 切片（实测 12x）；
- softmax/CE 行 tile 8→128（实测 4.9x）；
- 融合注意力 QK^T+softmax+PV 单 kernel（实测前向 13x）；
- PyPTOLinear 在 pypto 0.2.1+ 启用 matmul+bias 融合（实测 2.3x）。
"""

import importlib.metadata
import pypto
from pypto import pypto_impl
import torch
from torch import nn
from torch.nn import functional as F

# pypto 版本判断：0.2.1+ 启用 linear 融合（0.2.0 下存在数值不稳定，
# 见 KNOWN_ISSUES 问题 D）
try:
    _PYPTO_VERSION = tuple(
        int(x) for x in importlib.metadata.version("pypto").split(".")[:3])
except Exception:
    _PYPTO_VERSION = (0, 2, 0)
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
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 128])
    c.move(pypto.matmul(a, b, pypto.DT_FP32))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def matmul_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 128])
    grad_a.move(
        pypto.matmul(grad_c, b, pypto.DT_FP32, b_trans=True))
    grad_b.move(
        pypto.matmul(a, grad_c, pypto.DT_FP32, a_trans=True))


# ---------------------------------------------------------------------------
# linear 融合版：c = matmul(a, w) + bias，单 kernel（少一次 kernel 启动）。
#    实测 (640,512)x(512,512) 从分两次的 0.32ms 降到 0.14ms（约 2.3x）。
#    注意：pypto 0.2.0 下该融合存在调用序列相关的数值不稳定
#    （详见 KNOWN_ISSUES 问题 D），仅在 0.2.1+ 启用（0.2.1 下经
#    repro_s3 序列 3 轮验证 max_diff=0，且双 kernel 路径在 0.2.1 下
#    反而会触发该问题，融合路径更稳）。
#    反向复用 matmul_bwd_kernel + bias_add_bwd_kernel。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def linear_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    bias: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 128])
    pypto.set_vec_tile_shapes(64, 128)
    c.move(pypto.add(pypto.matmul(a, w, pypto.DT_FP32), bias))


def make_pypto_linear_fused(fwd_kernel, matmul_bwd_kernel, bias_bwd_kernel):
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
                matmul_bwd_kernel(x.contiguous(), weight.contiguous(),
                                  grad_c.contiguous(), tmp_x, tmp_w)
            grad_b = None
            if need_b:
                grad_b = torch.empty(weight.shape[1], device=x.device,
                                     dtype=x.dtype)
                # bias_add_bwd_kernel 的 a 参数未被 kernel 使用，传占位
                bias_bwd_kernel(grad_c.contiguous(), grad_c.contiguous(),
                                grad_b)
            return grad_x, grad_w, grad_b

    return PyPTOLinearFusedImpl


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
    """创建 PyPTO tanh 激活算子（支持任意维度，内部展平为 2D）。"""

    class PyPTOTanhImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            orig_shape = a.shape
            a_2d = a.reshape(-1, a.shape[-1]).contiguous()
            b_2d = torch.empty_like(a_2d)
            fwd_kernel(a_2d, b_2d)
            b = b_2d.reshape(orig_shape)
            ctx.save_for_backward(b)
            return b

        @staticmethod
        def backward(ctx, grad_b):
            (b,) = ctx.saved_tensors
            b_2d = b.reshape(-1, b.shape[-1]).contiguous()
            grad_b_2d = grad_b.reshape(-1, b.shape[-1]).contiguous()
            grad_a_2d = torch.empty_like(b_2d)
            if ctx.needs_input_grad[0]:
                bwd_kernel(b_2d, grad_b_2d, grad_a_2d)
            return grad_a_2d.reshape(b.shape)

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
    """创建 PyPTO ReLU 激活算子（支持任意维度，内部展平为 2D）。"""

    class PyPTOReLUOpImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            y_2d = torch.empty_like(x_2d)
            fwd_kernel(x_2d, y_2d)
            y = y_2d.reshape(orig_shape)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_out):
            (y,) = ctx.saved_tensors
            y_2d = y.reshape(-1, y.shape[-1]).contiguous()
            grad_out_2d = grad_out.reshape(-1, y.shape[-1]).contiguous()
            grad_in_2d = torch.empty_like(y_2d)
            if ctx.needs_input_grad[0]:
                bwd_kernel(y_2d, grad_out_2d, grad_in_2d)
            return grad_in_2d.reshape(y.shape)

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
    """创建 PyPTO sigmoid 激活算子（支持任意维度，内部展平为 2D）。"""

    class PyPTOSigmoidImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            orig_shape = a.shape
            a_2d = a.reshape(-1, a.shape[-1]).contiguous()
            b_2d = torch.empty_like(a_2d)
            fwd_kernel(a_2d, b_2d)
            b = b_2d.reshape(orig_shape)
            ctx.save_for_backward(b)
            return b

        @staticmethod
        def backward(ctx, grad_b):
            (b,) = ctx.saved_tensors
            b_2d = b.reshape(-1, b.shape[-1]).contiguous()
            grad_b_2d = grad_b.reshape(-1, b.shape[-1]).contiguous()
            grad_a_2d = torch.empty_like(b_2d)
            if ctx.needs_input_grad[0]:
                bwd_kernel(b_2d, grad_b_2d, grad_a_2d)
            return grad_a_2d.reshape(b.shape)

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
    """创建 PyPTO 逐元素乘法算子 c = a * b（支持任意维度，内部展平为 2D）。"""

    class PyPTOMulImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            orig_shape = a.shape
            a_2d = a.reshape(-1, a.shape[-1]).contiguous()
            b_2d = b.reshape(-1, b.shape[-1]).contiguous()
            c_2d = torch.empty_like(a_2d)
            fwd_kernel(a_2d, b_2d, c_2d)
            return c_2d.reshape(orig_shape)

        @staticmethod
        def backward(ctx, grad_c):
            a, b = ctx.saved_tensors
            need_grad_a = ctx.needs_input_grad[0]
            need_grad_b = ctx.needs_input_grad[1]
            a_2d = a.reshape(-1, a.shape[-1]).contiguous()
            b_2d = b.reshape(-1, b.shape[-1]).contiguous()
            grad_c_2d = grad_c.reshape(-1, grad_c.shape[-1]).contiguous()
            grad_a = torch.empty_like(a) if need_grad_a else None
            grad_b = torch.empty_like(b) if need_grad_b else None
            tmp_a = grad_a.reshape(-1, grad_a.shape[-1]).contiguous() \
                if need_grad_a else torch.empty_like(a_2d)
            tmp_b = grad_b.reshape(-1, grad_b.shape[-1]).contiguous() \
                if need_grad_b else torch.empty_like(b_2d)
            if need_grad_a or need_grad_b:
                bwd_kernel(grad_c_2d, a_2d, b_2d, tmp_a, tmp_b)
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
    # 行 tile 8→128（实测 (5120,10) softmax 0.47ms→0.097ms，4.9x）；
    # UB 保护（0.2.0/0.2.1 实测修正）：softmax 内部 DIV 约需 2×rows×cols×4B，
    # 128 行在 cols=208（V=201）时溢出（217088B > 192KB）；
    # 规则：cols ≤ 160 → 128 行，cols ≤ 320 → 64 行，否则退回 8 行。
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
    y.move(pypto.softmax(x, dim=-1))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def softmax_bwd_kernel(
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_y: pypto.Tensor([], pypto.DT_FP32),
    grad_x: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
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
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
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
    cols = ((num_classes + 7) // 8) * 8
    rows = 128 if cols <= 160 else (64 if cols <= 320 else 8)
    pypto.set_vec_tile_shapes(rows, cols)
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
PyPTOLinearFused = make_pypto_linear_fused(
    linear_fwd_kernel, matmul_bwd_kernel, bias_add_bwd_kernel)
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
        if orig_m % 32 != 0:
            pad_m = 32 - orig_m % 32
            x = F.pad(x, (0, 0, 0, pad_m))
            padded = True
        if self.bias is not None:
            if _PYPTO_GE_021:
                # 0.2.1+：matmul+bias 融合单 kernel（约 2.3x，且规避问题 D）
                y = PyPTOLinearFused.apply(x, self.weight, self.bias)
            else:
                # 0.2.0：融合 kernel 存在调用序列相关的数值不稳定（问题 D），
                # 保持 matmul + bias_add 两次启动的稳妥实现。
                y = PyPTOMatmul.apply(x, self.weight)
                y = PyPTOBiasAdd.apply(y, self.bias)
        else:
            y = PyPTOMatmul.apply(x, self.weight)
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
        if orig_batch % 32 != 0:
            batch_size += 32 - orig_batch % 32
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
        if orig_batch % 32 != 0:
            batch_size += 32 - orig_batch % 32
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
        if orig_batch % 32 != 0:
            batch_size += 32 - orig_batch % 32
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
# 第 10 章新增算子（注意力机制）
# =========================================================================

# ---------------------------------------------------------------------------
# bmm：批量矩阵乘法 C = A @ B
#    A: (batch, M, K)  B: (batch, K, N)  C: (batch, M, N)
#    反向：grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
# 性能说明：pypto 0.2.x 对 3 维批量 matmul 直接传 batch 维时，batch 循环
#   开销随 batch 数超线性增长（实测 B=512 时 (512,10,64)x(512,64,10) 约
#   2.5ms，torch 同形状仅 0.035ms）。参照官方仓库（models/glm_v4_5、
#   examples/03_advanced/advanced_nn/attention）的成熟写法：把 3 维张量在
#   宿主侧 reshape 为 2 维（纯视图，无拷贝），在 kernel 内用 pypto.loop +
#   pypto.view 逐 batch 取 2 维切片做 2D matmul，再 assemble 回输出。
#   实测同形状降到约 0.21ms（12 倍），数值精确（最大误差 0）。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bmm_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    c: pypto.Tensor([], pypto.DT_FP32),
    bsz: int,
    m: int,
    k: int,
    n: int,
):
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 128])
    pypto.set_vec_tile_shapes(128, 128)
    for i in pypto.loop(bsz, name="LOOP_b", idx_name="i"):
        ai = pypto.view(a, [m, k], [i * m, 0])
        bi = pypto.view(b, [k, n], [i * k, 0])
        ci = pypto.matmul(ai, bi, pypto.DT_FP32)
        pypto.assemble(ci, [i * m, 0], c)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def bmm_bwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_c: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    bsz: int,
    m: int,
    k: int,
    n: int,
):
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 128])
    pypto.set_vec_tile_shapes(128, 128)
    for i in pypto.loop(bsz, name="LOOP_b", idx_name="i"):
        ai = pypto.view(a, [m, k], [i * m, 0])
        bi = pypto.view(b, [k, n], [i * k, 0])
        gci = pypto.view(grad_c, [m, n], [i * m, 0])
        gai = pypto.matmul(gci, bi, pypto.DT_FP32, b_trans=True)
        gbi = pypto.matmul(ai, gci, pypto.DT_FP32, a_trans=True)
        pypto.assemble(gai, [i * m, 0], grad_a)
        pypto.assemble(gbi, [i * k, 0], grad_b)


def make_pypto_bmm(fwd_kernel, bwd_kernel):
    """创建 PyPTO 批量矩阵乘法算子（与 torch.bmm 语义一致）。"""

    class PyPTOBMMImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            bsz, m, k = a.shape
            n = b.shape[-1]
            a2 = a.reshape(bsz * m, k).contiguous()
            b2 = b.reshape(bsz * k, n).contiguous()
            c2 = torch.empty(bsz * m, n, device=a.device, dtype=a.dtype)
            fwd_kernel(a2, b2, c2, bsz, m, k, n)
            return c2.reshape(bsz, m, n)

        @staticmethod
        def backward(ctx, grad_c):
            a, b = ctx.saved_tensors
            need_a = ctx.needs_input_grad[0]
            need_b = ctx.needs_input_grad[1]
            bsz, m, k = a.shape
            n = b.shape[-1]
            a2 = a.reshape(bsz * m, k).contiguous()
            b2 = b.reshape(bsz * k, n).contiguous()
            gc2 = grad_c.reshape(bsz * m, n).contiguous()
            grad_a = torch.empty_like(a) if need_a else None
            grad_b = torch.empty_like(b) if need_b else None
            tmp_a = grad_a.reshape(bsz * m, k) if need_a \
                else torch.empty_like(a2)
            tmp_b = grad_b.reshape(bsz * k, n) if need_b \
                else torch.empty_like(b2)
            if need_a or need_b:
                bwd_kernel(a2, b2, gc2, tmp_a, tmp_b, bsz, m, k, n)
            return grad_a, grad_b

    return PyPTOBMMImpl


# ---------------------------------------------------------------------------
# 融合缩放点积注意力（官方仓库模式，参照 models/glm_v4_5/glm_attention.py
#   与 examples/03_advanced/advanced_nn/attention/attention.py）：
#   把 QK^T + 缩放 + softmax + PV 全部放进同一个 kernel，用 pypto.loop +
#   pypto.view 逐 batch 计算 2D matmul。
#   实测 (512,10,64) 前向 0.42ms vs 分解式（2 次 bmm + softmax + 缩放）
#   5.55ms（13 倍）；反向（softmax 梯度三式 + 3 个梯度 matmul）0.55ms。
#   语义与 torch: softmax(Q K^T / sqrt(d)) @ V 一致（无 mask）。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def attention_fwd_kernel(
    q: pypto.Tensor([], pypto.DT_FP32),
    k: pypto.Tensor([], pypto.DT_FP32),
    v: pypto.Tensor([], pypto.DT_FP32),
    p: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
    bsz: int,
    seq: int,
    dim: int,
    scale: float,
):
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 128])
    cols = ((seq + 7) // 8) * 8
    rows = 128 if cols <= 160 else 64  # UB 保护（同 softmax：2×rows×cols×4B < 192KB）
    pypto.set_vec_tile_shapes(rows, cols)
    for b in pypto.loop(bsz, name="LOOP_b", idx_name="b"):
        qi = pypto.view(q, [seq, dim], [b * seq, 0])
        ki = pypto.view(k, [seq, dim], [b * seq, 0])
        vi = pypto.view(v, [seq, dim], [b * seq, 0])
        sij = pypto.matmul(qi, ki, pypto.DT_FP32, b_trans=True)
        sij = pypto.mul(sij, scale)
        pi = pypto.softmax(sij, dim=-1)
        pypto.assemble(pi, [b * seq, 0], p)
        oi = pypto.matmul(pi, vi, pypto.DT_FP32)
        pypto.assemble(oi, [b * seq, 0], out)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def attention_bwd_kernel(
    q: pypto.Tensor([], pypto.DT_FP32),
    k: pypto.Tensor([], pypto.DT_FP32),
    v: pypto.Tensor([], pypto.DT_FP32),
    p: pypto.Tensor([], pypto.DT_FP32),
    grad_o: pypto.Tensor([], pypto.DT_FP32),
    grad_q: pypto.Tensor([], pypto.DT_FP32),
    grad_k: pypto.Tensor([], pypto.DT_FP32),
    grad_v: pypto.Tensor([], pypto.DT_FP32),
    bsz: int,
    seq: int,
    dim: int,
    scale: float,
):
    pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 128])
    cols = ((seq + 7) // 8) * 8
    rows = 128 if cols <= 160 else 64  # UB 保护（同 softmax：2×rows×cols×4B < 192KB）
    pypto.set_vec_tile_shapes(rows, cols)
    for b in pypto.loop(bsz, name="LOOP_b", idx_name="b"):
        qi = pypto.view(q, [seq, dim], [b * seq, 0])
        ki = pypto.view(k, [seq, dim], [b * seq, 0])
        vi = pypto.view(v, [seq, dim], [b * seq, 0])
        pi = pypto.view(p, [seq, seq], [b * seq, 0])
        goi = pypto.view(grad_o, [seq, dim], [b * seq, 0])
        # dV = P^T @ dO
        dv = pypto.matmul(pi, goi, pypto.DT_FP32, a_trans=True)
        # dP = dO @ V^T；dS = P ⊙ (dP - sum(dP ⊙ P))，再乘 scale
        dpi = pypto.matmul(goi, vi, pypto.DT_FP32, b_trans=True)
        dsi = pypto.mul(pi, pypto.sub(dpi,
                                      pypto.sum(pypto.mul(dpi, pi), 1, True)))
        dsi = pypto.mul(dsi, scale)
        dqi = pypto.matmul(dsi, ki, pypto.DT_FP32)
        dki = pypto.matmul(dsi, qi, pypto.DT_FP32, a_trans=True)
        pypto.assemble(dv, [b * seq, 0], grad_v)
        pypto.assemble(dqi, [b * seq, 0], grad_q)
        pypto.assemble(dki, [b * seq, 0], grad_k)


def make_pypto_fused_attention(fwd_kernel, bwd_kernel):
    """创建融合缩放点积注意力算子。

    输入 q/k/v: (batch, seq, dim) 连续 FP32；返回 (out, p)：
      out: (batch, seq, dim)
      p:   (batch, seq, seq) softmax 概率（供 attention_weights 可视化）
    """

    class PyPTOFusedAttentionImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, scale):
            bsz, seq, dim = q.shape
            q2 = q.reshape(bsz * seq, dim).contiguous()
            k2 = k.reshape(bsz * seq, dim).contiguous()
            v2 = v.reshape(bsz * seq, dim).contiguous()
            p2 = torch.empty(bsz * seq, seq, device=q.device, dtype=q.dtype)
            o2 = torch.empty(bsz * seq, dim, device=q.device, dtype=q.dtype)
            fwd_kernel(q2, k2, v2, p2, o2, bsz, seq, dim, scale)
            ctx.save_for_backward(q2, k2, v2, p2)
            ctx.bsz, ctx.seq, ctx.dim, ctx.scale = bsz, seq, dim, scale
            return o2.reshape(bsz, seq, dim), p2.reshape(bsz, seq, seq)

        @staticmethod
        def backward(ctx, grad_o, _grad_p):
            q2, k2, v2, p2 = ctx.saved_tensors
            bsz, seq, dim, scale = ctx.bsz, ctx.seq, ctx.dim, ctx.scale
            go2 = grad_o.reshape(bsz * seq, dim).contiguous()
            gq2 = torch.empty_like(q2)
            gk2 = torch.empty_like(k2)
            gv2 = torch.empty_like(v2)
            bwd_kernel(q2, k2, v2, p2, go2, gq2, gk2, gv2,
                       bsz, seq, dim, scale)
            return (gq2.reshape(bsz, seq, dim),
                    gk2.reshape(bsz, seq, dim),
                    gv2.reshape(bsz, seq, dim), None)

    return PyPTOFusedAttentionImpl


# ---------------------------------------------------------------------------
# softmax（独立版）：y = softmax(x, dim=-1)，任意维度，内部展平为 2D。
#    反向：grad_x = y ⊙ grad_y - y ⊙ sum(y ⊙ grad_y, axis=-1)
#    复用第 9 章融合 Softmax+CE 中的 fwd/bwd kernel（tiling (8, aligned)）。
# ---------------------------------------------------------------------------

def make_pypto_softmax(fwd_kernel, bwd_kernel):
    """创建 PyPTO 独立 softmax 算子（dim=-1）。"""

    class PyPTOSoftmaxImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            orig_shape = x.shape
            num_classes = x.shape[-1]
            x_2d = x.reshape(-1, num_classes).contiguous()
            y_2d = torch.empty_like(x_2d)
            fwd_kernel(x_2d, y_2d, num_classes)
            y = y_2d.reshape(orig_shape)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            (y,) = ctx.saved_tensors
            num_classes = y.shape[-1]
            y_2d = y.reshape(-1, num_classes).contiguous()
            grad_y_2d = grad_y.reshape(-1, num_classes).contiguous()
            grad_x_2d = torch.empty_like(y_2d)
            bwd_kernel(y_2d, grad_y_2d, grad_x_2d, num_classes)
            return grad_x_2d.reshape(y.shape)

    return PyPTOSoftmaxImpl


# ---------------------------------------------------------------------------
# exp：y = exp(x)，反向：grad_x = exp(x) ⊙ grad_y = y ⊙ grad_y
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def exp_fwd_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    b.move(pypto.exp(a))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def exp_bwd_kernel(
    b: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
    grad_a: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(128, 128)
    grad_a.move(pypto.mul(grad_b, b))


def make_pypto_exp(fwd_kernel, bwd_kernel):
    """创建 PyPTO exp 激活算子。"""

    class PyPTOExpImpl(torch.autograd.Function):
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

    return PyPTOExpImpl


# ---------------------------------------------------------------------------
# layer_norm：沿最后一维的层归一化（可学习 γ/β）
#    y = (x - mean) / sqrt(var + eps) * γ + β
#    反向（N 为最后一维长度）：
#      grad_γ = sum(grad_y ⊙ y_hat, axis=-1)
#      grad_β = sum(grad_y, axis=-1)
#      grad_x = inv_std ⊙ (grad_y - mean(grad_y) - y_hat ⊙ mean(grad_y ⊙ y_hat))
#    输入展平为 2D (M, N)，γ/β 形状为 (N,)，tiling (8, aligned) 支持逐行归约。
# ---------------------------------------------------------------------------

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def layer_norm_fwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    gamma: pypto.Tensor([], pypto.DT_FP32),
    beta: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    num_features: int,
):
    pypto.set_vec_tile_shapes(8, ((num_features + 7) // 8) * 8)
    mean = pypto.mul(pypto.sum(x, 1, True), 1.0 / num_features)
    x_centered = pypto.sub(x, mean)
    var = pypto.mul(pypto.sum(pypto.mul(x_centered, x_centered), 1, True),
                    1.0 / num_features)
    inv_std = pypto.rsqrt(pypto.add(var, 1e-5))
    y_hat = pypto.mul(x_centered, inv_std)
    y.move(pypto.add(pypto.mul(y_hat, gamma), beta))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def layer_norm_bwd_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    gamma: pypto.Tensor([], pypto.DT_FP32),
    grad_y: pypto.Tensor([], pypto.DT_FP32),
    grad_x: pypto.Tensor([], pypto.DT_FP32),
    grad_gamma: pypto.Tensor([], pypto.DT_FP32),
    grad_beta: pypto.Tensor([], pypto.DT_FP32),
    num_features: int,
):
    pypto.set_vec_tile_shapes(8, ((num_features + 7) // 8) * 8)
    mean = pypto.mul(pypto.sum(x, 1, True), 1.0 / num_features)
    x_centered = pypto.sub(x, mean)
    var = pypto.mul(pypto.sum(pypto.mul(x_centered, x_centered), 1, True),
                    1.0 / num_features)
    inv_std = pypto.rsqrt(pypto.add(var, 1e-5))
    y_hat = pypto.mul(x_centered, inv_std)
    # γ 参与 grad_x 的均值项（对 y_hat 的梯度为 grad_y ⊙ γ）
    gy_gamma = pypto.mul(grad_y, gamma)
    g_mean = pypto.mul(pypto.sum(gy_gamma, 1, True), 1.0 / num_features)
    gy_hat = pypto.mul(pypto.sum(pypto.mul(gy_gamma, y_hat), 1, True),
                       1.0 / num_features)
    grad_x.move(pypto.mul(inv_std,
                          pypto.sub(pypto.sub(gy_gamma, g_mean),
                                    pypto.mul(y_hat, gy_hat))))
    grad_gamma.move(pypto.sum(pypto.mul(grad_y, y_hat), 0))
    grad_beta.move(pypto.sum(grad_y, 0))


def make_pypto_layer_norm(fwd_kernel, bwd_kernel):
    """创建 PyPTO 层归一化算子（支持任意维度，内部展平为 2D）。"""

    class PyPTOLayerNormImpl(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, gamma, beta, eps=1e-5):
            orig_shape = x.shape
            num_features = x.shape[-1]
            x_2d = x.reshape(-1, num_features).contiguous()
            gamma_2d = gamma.reshape(1, -1).contiguous()
            beta_2d = beta.reshape(1, -1).contiguous()
            y_2d = torch.empty_like(x_2d)
            fwd_kernel(x_2d, gamma_2d, beta_2d, y_2d, num_features)
            y = y_2d.reshape(orig_shape)
            ctx.save_for_backward(x, gamma, beta)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            x, gamma, beta = ctx.saved_tensors
            num_features = x.shape[-1]
            x_2d = x.reshape(-1, num_features).contiguous()
            gamma_2d = gamma.reshape(1, -1).contiguous()
            grad_y_2d = grad_y.reshape(-1, num_features).contiguous()
            grad_x_2d = torch.empty_like(x_2d)
            grad_gamma_2d = torch.empty(num_features, device=x.device,
                                        dtype=x.dtype)
            grad_beta_2d = torch.empty(num_features, device=x.device,
                                       dtype=x.dtype)
            bwd_kernel(x_2d, gamma_2d, grad_y_2d, grad_x_2d,
                       grad_gamma_2d, grad_beta_2d, num_features)
            grad_gamma = grad_gamma_2d.reshape(gamma.shape)
            grad_beta = grad_beta_2d.reshape(beta.shape)
            return grad_x_2d.reshape(x.shape), grad_gamma, grad_beta, None

    return PyPTOLayerNormImpl


# =========================================================================
# 实例化第 10 章算子（工厂函数调用）
# =========================================================================

PyPTOBMM = make_pypto_bmm(bmm_fwd_kernel, bmm_bwd_kernel)
PyPTOFusedAttention = make_pypto_fused_attention(
    attention_fwd_kernel, attention_bwd_kernel)
PyPTOSoftmax = make_pypto_softmax(softmax_fwd_kernel, softmax_bwd_kernel)
PyPTOExp = make_pypto_exp(exp_fwd_kernel, exp_bwd_kernel)
PyPTOLayerNorm = make_pypto_layer_norm(layer_norm_fwd_kernel,
                                       layer_norm_bwd_kernel)


class PyPTOLayerNormModule(nn.Module):
    """用 PyPTO kernel 实现的层归一化（支持多维 normalized_shape）。

    与 nn.LayerNorm 接口兼容（weight/bias 可学习参数，名字同为
    weight/bias）。首次出现在 10.7 节（notebook 内联完整实现），
    此处收录供 10.8 节及后续章节复用。
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, X):
        orig_shape = X.shape
        if isinstance(self.normalized_shape, int):
            num_dims = 1
        else:
            num_dims = len(self.normalized_shape)
        # 将最后 num_dims 维展平为单维，调用 PyPTO LayerNorm kernel
        x2 = X.reshape(orig_shape[:-num_dims] + (-1,)).contiguous()
        y2 = PyPTOLayerNorm.apply(x2, self.weight.reshape(-1),
                                  self.bias.reshape(-1))
        return y2.reshape(orig_shape)


# =========================================================================
# conv2d：前向 pypto.conv（L1/L0/vec tile 显式配置），
#   反向用 unfold/matmul/fold 等价公式（pypto 暂无 conv 反向 kernel）
# =========================================================================


def _make_conv2d_kernel(in_shape, weight_shape, out_shape, dtype, strides,
                        paddings, dilations):
    """创建 2D 卷积前向 kernel（strides/paddings/dilations/tile 编译期常量）。"""

    B, Cin, H, W = in_shape
    Cout, _, kh, kw = weight_shape
    _, _, out_h, out_w = out_shape
    C0 = 8 if dtype == pypto.DT_FP32 else 16
    dtsize = 4 if dtype == pypto.DT_FP32 else 2
    k0 = 32 // dtsize
    # L1 tile：非 16 对齐的 Wout 强制 tileHout=1、tileWout=16
    tile_wout = out_w if out_w % 16 == 0 else 16
    tile_hout = 1 if out_w % 16 != 0 else min(out_h, 16)
    hi_al1 = min((tile_hout - 1) * strides[0] + (kh - 1) * dilations[0] + 1, H)
    wi_al1 = min((tile_wout - 1) * strides[1] + (kw - 1) * dilations[1] + 1, W)
    # L0 tile：L0A 空间约束 tileH*tileW*tileK*dtsize <= 64KB（H=1, W=16）
    k_al1 = ((Cin * kh * kw + k0 - 1) // k0) * k0
    tile_k = 65536 // (16 * 16 * dtsize)
    tile_k = (tile_k // k0) * k0
    while k_al1 % tile_k != 0:
        tile_k //= 2
    # vec tile：N 为 16 的倍数、末维 = C0；transdata 的 UB workspace 随
    # batch × vec_h 线性增长，H 维取 1 以适配大 batch（实测 B=128 时
    # vec_h=2 溢出 UB，vec_h=1 通过）
    vec_tile = (16, min(Cout, 512), 1, C0)

    l1_info = pypto_impl.TileL1Info(
        tileHin=hi_al1, tileHout=tile_hout, tileWin=wi_al1, tileWout=tile_wout,
        tileCinFmap=Cin, tileCinWeight=Cin, tileN=16, tileBatch=1)
    l0_info = pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=tile_k, tileN=16)

    @pypto.frontend.jit()
    def conv2d_fwd_kernel(
        fmap: pypto.Tensor([B, Cin, H, W], dtype),
        weight: pypto.Tensor(weight_shape, dtype),
        bias: pypto.Tensor([Cout], dtype),
        out: pypto.Tensor([B, Cout, out_h, out_w], dtype),
    ):
        pypto.set_conv_tile_shapes(l1_info, l0_info)
        pypto.set_vec_tile_shapes(*vec_tile)
        output = pypto.conv(fmap, weight, dtype, strides, paddings, dilations,
                            extend_params={"bias_tensor": bias}, groups=1)
        out.move(output)

    return conv2d_fwd_kernel


class PyPTOConv2dImpl(torch.autograd.Function):
    """2D 卷积 autograd 包装：前向 pypto.conv，反向 unfold/matmul/fold。

    输入 x/weight 已由 PyPTOConv2d 补零到 C0 对齐通道数（补零通道
    梯度恒为 0，F.pad 的自动微分会将其裁回原始通道）。
    """

    @staticmethod
    def forward(ctx, x, weight, bias, kernel, out_shape):
        ctx.save_for_backward(x, weight)
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        b = bias if bias is not None else torch.zeros(
            out_shape[1], device=x.device, dtype=x.dtype)
        kernel(x.contiguous(), weight.contiguous(), b, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        kh, kw = weight.shape[2], weight.shape[3]
        N, Cin, H, W = x.shape
        Cout = weight.shape[0]
        grad_x = grad_w = None
        if need_x or need_w:
            # 卷积的梯度等价公式（与 PyTorch im2col 版一致）：
            #   grad_w = unfold(x)^T @ grad_out
            #   grad_x = fold(weight^T @ grad_out)
            patches = F.unfold(x, (kh, kw), stride=(kh, kw))  # (N, Cin*kh*kw, P)
            g_flat = grad_out.contiguous().reshape(N, Cout, -1)  # (N, Cout, P)
            pp = patches.transpose(1, 2).reshape(-1, Cin * kh * kw)
            gg = g_flat.transpose(1, 2).reshape(-1, Cout)
            if need_w:
                grad_w = torch.matmul(pp.transpose(0, 1).contiguous(), gg)
                grad_w = grad_w.transpose(0, 1).reshape(weight.shape)
            if need_x:
                w2 = weight.reshape(Cout, -1).transpose(0, 1)  # (Cpp, Cout)
                tmp = torch.matmul(w2.unsqueeze(0), g_flat)  # (N, Cpp, P)
                grad_x = F.fold(tmp, output_size=(H, W),
                                kernel_size=(kh, kw), stride=(kh, kw))
        grad_b = grad_out.sum(dim=(0, 2, 3)) if need_b else None
        return grad_x, grad_w, grad_b, None, None


# conv 单次调用的固定 batch 切块（编译期常量）。
# 理由（实测，pypto 0.2.1 / 950D）：
#   - conv 的 batch 维必须静态（不支持动态轴），不同 batch 各编译一次；
#   - transdata 的 UB workspace 随 batch 线性增长，B<=4 时才不超 192KB；
#   - 编译耗时随 batch 线性增长（B=4 约 80s，B=128 约 40 分钟不可用）。
# 因此 kernel 固定 B=4，更大 batch 由 PyPTOConv2d 在 Python 层循环切块调用。
_CONV_BATCH_CHUNK = 4

# 同一进程内共享已编译 kernel（jit 缓存按闭包对象身份失效，需按 shape 自建缓存）
_CONV_KERNEL_CACHE = {}


def _get_conv2d_kernel(in_shape, weight_shape, out_shape, dtype, strides,
                       paddings, dilations):
    key = (in_shape, weight_shape, out_shape, dtype,
           tuple(strides), tuple(paddings), tuple(dilations))
    if key not in _CONV_KERNEL_CACHE:
        _CONV_KERNEL_CACHE[key] = _make_conv2d_kernel(
            in_shape, weight_shape, out_shape, dtype, strides, paddings,
            dilations)
    return _CONV_KERNEL_CACHE[key]


class PyPTOConv2d(nn.Module):
    """用 PyPTO conv 算子实现的 2D 卷积层（与 nn.Conv2d 接口兼容子集）。

    当前仅支持 stride == kernel_size、padding == 0（覆盖 10.8 节
    图像块嵌入的切块卷积场景；其余参数下 tile 推导未经验证）。

    前向：通道数不足 C0 对齐（FP32 为 8）时自动补零通道后调用
    `pypto.conv`；反向：pypto 暂无 conv 反向 kernel，使用
    unfold/matmul/fold 等价公式（torch 实现，结果与卷积梯度一致）。

    实现说明：kernel 以固定 batch=4 编译（见 `_CONV_BATCH_CHUNK`），
    更大的 batch 在 Python 层按 4 切块循环调用并拼接，因此任何 batch
    都只编译一次；尾块不足 4 时补零并裁回。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if not isinstance(
            kernel_size, (list, tuple)) else tuple(kernel_size)
        sh, sw = (stride, stride) if not isinstance(
            stride, (list, tuple)) else tuple(stride)
        assert sh == kh and sw == kw, \
            "PyPTOConv2d 当前仅支持 stride == kernel_size"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kh, kw) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self._impl = None  # (kernel, cin_pad, chunk_out_shape)，首次 forward 构建

    def forward(self, X):
        if self._impl is None:
            self._impl = self._build_impl(X)
        kernel, cin_pad, chunk_out_shape = self._impl
        B = X.shape[0]
        kh, kw = self.kernel_size
        pad_c = cin_pad - X.shape[1]
        w = self.weight
        if pad_c > 0:
            w = F.pad(w, (0, 0, 0, 0, 0, pad_c, 0, 0))
        outputs = []
        for start in range(0, B, _CONV_BATCH_CHUNK):
            xc = X[start:start + _CONV_BATCH_CHUNK]
            tail = _CONV_BATCH_CHUNK - xc.shape[0]
            if tail > 0:
                xc = F.pad(xc, (0, 0, 0, 0, 0, 0, 0, tail))
            if pad_c > 0:
                xc = F.pad(xc, (0, 0, 0, 0, 0, pad_c, 0, 0))
            y = PyPTOConv2dImpl.apply(xc, w, self.bias, kernel,
                                      chunk_out_shape)
            outputs.append(y[:xc.shape[0] - tail] if tail else y)
        return torch.cat(outputs, dim=0)

    def _build_impl(self, X):
        C0 = 8 if X.dtype == torch.float32 else 16
        cin_pad = ((self.in_channels + C0 - 1) // C0) * C0
        B, _, H, W = X.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dtype = pypto.DT_FP32 if X.dtype == torch.float32 else pypto.DT_FP16
        chunk = _CONV_BATCH_CHUNK
        in_shape = (chunk, cin_pad, H, W)
        w_shape = (self.out_channels, cin_pad, kh, kw)
        chunk_out_shape = (chunk, self.out_channels, H // kh, W // kw)
        kernel = _get_conv2d_kernel(in_shape, w_shape, chunk_out_shape,
                                    dtype, [sh, sw], [0, 0, 0, 0], [1, 1])
        return kernel, cin_pad, chunk_out_shape


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
    "PyPTOBMM", "PyPTOFusedAttention", "PyPTOSoftmax", "PyPTOExp", "PyPTOLayerNorm",
    "PyPTOLayerNormModule",
    "PyPTOLinear", "PyPTOReLU",
    "PyPTOConv2d",
    "PyPTORNN", "PyPTOGRU", "PyPTOLSTM",
    "loss_fn",
]

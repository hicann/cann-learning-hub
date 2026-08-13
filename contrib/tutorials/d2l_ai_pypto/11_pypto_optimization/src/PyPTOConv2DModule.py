"""PyPTO Conv2d 模块 (AIC 优化版 v3, 基于 custom_module_conv2 重构)

fwd / dK / dX 全部走 PyPTO ``matmul`` (AIC/Cube, FP32), 与
custom_module_conv2/PyPTOConv2DModule_copy.py 同路线。

vs v2 的改进:
- **消除 3 处转置拷贝** (v2 每轮 fwd/dK/dX 各做一次 .t().contiguous()):
  - fwd:  C = matmul(cols, K2d, b_trans=True)   K2d = K.reshape(Cout, K) 零拷贝 view
  - dK:   C = matmul(cols, dY2d, a_trans=True)  免 cols^T 拷贝
  - dX:   C = matmul(cols, K_ic,  b_trans=True) 免 K_ic^T 拷贝
- kernel 从 3 个精简为 2 个: b_trans kernel 供 fwd/dX 共用, a_trans kernel 供 dK
  (trans 标志必须编译期硬编码, 不能作为运行时参数)
- 移除 transposed 分支与 2D 函数式 API (均未验证, 以简洁的报错替代)
- 移除 run_mode argv 解析等样板代码

限制: stride 任意, dilation=1, transposed=False (与 v2 实测路径一致)。
"""

import torch
import torch.nn.functional as F
from torch import nn

import pypto

# 统一 Cube tile shape (经 bench 验证对 LeNet 各 shape 最优)
_CUBE_TILE = [128, 128], [64, 256], [256, 256]


@pypto.frontend.jit(runtime_options={})
def _matmul_bt_kernel(
    A:     pypto.Tensor([], pypto.DT_FP32),   # (M, K)
    B:     pypto.Tensor([], pypto.DT_FP32),   # (N, K), 计算时取 B^T
    out:   pypto.Tensor([], pypto.DT_FP32),   # (M, N)
):
    """C = A @ B^T (免 B 转置拷贝; fwd/dX 共用)."""
    pypto.set_cube_tile_shapes(*_CUBE_TILE)
    out[:] = pypto.matmul(A, B, pypto.DT_FP32, b_trans=True)


@pypto.frontend.jit(runtime_options={})
def _matmul_at_kernel(
    A:     pypto.Tensor([], pypto.DT_FP32),   # (M, K), 计算时取 A^T
    B:     pypto.Tensor([], pypto.DT_FP32),   # (M, N)
    out:   pypto.Tensor([], pypto.DT_FP32),   # (K, N)
):
    """C = A^T @ B (免 A 转置拷贝; dK 用)."""
    pypto.set_cube_tile_shapes(*_CUBE_TILE)
    out[:] = pypto.matmul(A, B, pypto.DT_FP32, a_trans=True)


# ── 辅助 ──


def _to_pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _im2col_strided(X_padded, kH, kW, stride=(1, 1)):
    """X_padded (N,Cin,Hp,Wp) → cols (M, Cin*kH*kW), M=N*H_out*W_out.

    torch.as_strided 零拷贝窗口 view + 一次 permute/reshape/contiguous.
    (torch_npu 上 F.unfold 慢 ~10-18x, 故不用)
    """
    s_h, s_w = _to_pair(stride)
    N, Cin, Hp, Wp = X_padded.shape
    Ho = (Hp - kH) // s_h + 1
    Wo = (Wp - kW) // s_w + 1
    view = X_padded.as_strided(
        (N, Cin, Ho, Wo, kH, kW),
        (Cin * Hp * Wp, Hp * Wp, s_h * Wp, s_w, Wp, 1),
    )
    return view.permute(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, Cin * kH * kW).contiguous()


def _pad4d(X, p_h, p_w):
    """对称 pad; F.pad 实测比 zeros+copy 快 ~4.5x (torch_npu)."""
    if p_h == 0 and p_w == 0:
        return X
    return F.pad(X, [p_w, p_w, p_h, p_h])


def _conv_geom(X_shape, kH, kW, stride, padding):
    """输出尺寸 (H_out, W_out)."""
    s_h, s_w = _to_pair(stride)
    p_h, p_w = _to_pair(padding)
    H, W = X_shape[2], X_shape[3]
    return (H + 2 * p_h - kH) // s_h + 1, (W + 2 * p_w - kW) // s_w + 1


# ── forward: Y = im2col(X_pad) @ K^T ──


def _conv_forward_via_matmul(X, K, bias, stride, padding):
    N, Cin, H, W = X.shape
    Cout, _, kH, kW = K.shape
    p_h, p_w = _to_pair(padding)
    H_out, W_out = _conv_geom(X.shape, kH, kW, stride, padding)

    X_pad = _pad4d(X, p_h, p_w)
    cols = _im2col_strided(X_pad, kH, kW, stride)          # (M, K_dim)
    K_2d = K.reshape(Cout, Cin * kH * kW)                  # (Cout, K_dim), 零拷贝 view
    out = torch.zeros(N * H_out * W_out, Cout, device=X.device, dtype=X.dtype)
    _matmul_bt_kernel(cols, K_2d, out)                     # C = cols @ K_2d^T
    if bias is not None:
        out = out + bias.view(1, Cout)                     # bias 不融合: 多核切 K 时 extend_params 结果错误
    return out.reshape(N, H_out, W_out, Cout).permute(0, 3, 1, 2).contiguous()


# ── dK: dK = im2col(X_pad)^T @ dY_2d ──


def _conv_dK_via_matmul(X, grad_Y, K_4d, stride, padding):
    N, Cin, H, W = X.shape
    Cout = grad_Y.shape[1]
    kH, kW = K_4d.shape[2], K_4d.shape[3]
    p_h, p_w = _to_pair(padding)
    H_out, W_out = _conv_geom(X.shape, kH, kW, stride, padding)

    X_pad = _pad4d(X, p_h, p_w)
    cols = _im2col_strided(X_pad, kH, kW, stride)          # (M, K_dim)
    dY_2d = grad_Y.permute(0, 2, 3, 1).reshape(N * H_out * W_out, Cout).contiguous()
    dK_unfold = torch.zeros(Cin * kH * kW, Cout, device=X.device, dtype=X.dtype)
    _matmul_at_kernel(cols, dY_2d, dK_unfold)              # C = cols^T @ dY_2d
    return dK_unfold.reshape(Cin, kH, kW, Cout).permute(3, 0, 1, 2).contiguous()


# ── dX: dX = im2col(dY_pad) @ K_ic^T (旋转核, 无 col2im 折叠) ──


def _conv2d_compute_dX(grad_Y, K_4d, X_shape, stride, padding):
    """dX[n,ic,h,w] = Σ K_rot[oc,ic,kh,kw] * dY[oc, h+kh-p, w+kw-p].

    stride>1: 由恒等式 dX = conv2d(upsample(dY, stride), K_rot, pad=kH-1-p, stride=1)
    (strided conv 的 backward 等价于把 dY 插零放大后再做 stride=1 的卷积)
    dY 对称 pad (kH-1-p, kW-1-p) 后 im2col, matmul 输出直接是 dX (无偏移),
    matmul 结果再按 X_shape 裁剪 (p > kH-1 时补零)。
    """
    N, Cin, H, W = X_shape
    Cout = grad_Y.shape[1]
    kH, kW = K_4d.shape[2], K_4d.shape[3]
    p_h, p_w = _to_pair(padding)
    s_h, s_w = _to_pair(stride)
    gH, gW = grad_Y.shape[2], grad_Y.shape[3]
    pad_h, pad_w = max(0, kH - 1 - p_h), max(0, kW - 1 - p_w)

    if s_h != 1 or s_w != 1:
        # 插零放大 grad_Y: dX[h] = Σ dY_up[h+p-j]·K[j], 需覆盖下标到 H-1+p,
        # 故再补零到 max((gH-1)*s+1, H+p) (多出的行全零不影响结果)
        gy_up = torch.zeros(N, Cout, (gH - 1) * s_h + 1, (gW - 1) * s_w + 1,
                            device=grad_Y.device, dtype=grad_Y.dtype)
        gy_up[:, :, ::s_h, ::s_w] = grad_Y
        l_up_h = max((gH - 1) * s_h + 1, H + p_h)
        l_up_w = max((gW - 1) * s_w + 1, W + p_w)
        if l_up_h > gy_up.shape[2] or l_up_w > gy_up.shape[3]:
            gy_up = F.pad(gy_up, [0, l_up_w - gy_up.shape[3],
                                  0, l_up_h - gy_up.shape[2]])
        grad_Y = gy_up
        gH, gW = grad_Y.shape[2], grad_Y.shape[3]

    K_rot = torch.flip(K_4d, [2, 3])                       # rot180
    K_ic = K_rot.permute(1, 0, 2, 3).reshape(Cin, Cout * kH * kW).contiguous()
    grad_Y_pad = _pad4d(grad_Y, pad_h, pad_w)
    cols = _im2col_strided(grad_Y_pad, kH, kW, (1, 1))     # (M, Cout*kH*kW)
    out = torch.zeros(cols.shape[0], Cin, device=grad_Y.device, dtype=grad_Y.dtype)
    _matmul_bt_kernel(cols, K_ic, out)                     # C = cols @ K_ic^T

    L_h, L_w = gH + 2 * pad_h - kH + 1, gW + 2 * pad_w - kW + 1
    dX_inter = out.reshape(N, L_h, L_w, Cin).permute(0, 3, 1, 2).contiguous()
    if L_h >= H and L_w >= W:
        return dX_inter[:, :, :H, :W].contiguous()
    dX = torch.zeros(N, Cin, H, W, device=grad_Y.device, dtype=grad_Y.dtype)
    dX[:, :, :L_h, :L_w] = dX_inter
    return dX


# ── autograd Function + nn.Module ──


class PyPTOConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, K, stride, padding, dilation, bias):
        s_h, s_w = _to_pair(stride)
        if _to_pair(dilation) != (1, 1):
            raise NotImplementedError("AIC matmul 路线不支持 dilation")
        X_c, K_c = X.contiguous(), K.contiguous()
        Y = _conv_forward_via_matmul(X_c, K_c, bias, stride, padding)
        ctx.save_for_backward(X_c, K_c, bias)
        ctx.params = (s_h, s_w, padding)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, K, bias = ctx.saved_tensors
        s_h, s_w, padding = ctx.params
        grad_Y = grad_output.contiguous()

        dK = _conv_dK_via_matmul(X, grad_Y, K, (s_h, s_w), padding)
        dX = _conv2d_compute_dX(grad_Y, K, X.shape, (s_h, s_w), padding)
        db = grad_Y.sum(dim=(0, 2, 3)).contiguous() if bias is not None else None
        return dX, dK, None, None, None, db


class PyPTOConv2d(nn.Module):
    """PyPTO Conv2d (AIC matmul, FP32): fwd/dK/dX 全 PyPTO.

    stride 任意, dilation=1, transposed=False (接口参数保留, 不支持时抛错)。
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3),
                 stride=1, padding=0, dilation=1, transposed=False,
                 bias=False, device="cpu", dtype=torch.float32):
        super().__init__()
        if transposed:
            raise NotImplementedError("transposed 未实现, 请用 torch.nn.ConvTranspose2d")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size,
                                              device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.bias = (nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
                     if bias else None)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -fan_in ** -0.5, fan_in ** -0.5)

    def forward(self, x):
        return PyPTOConv2dFunction.apply(
            x, self.weight, self.stride, self.padding, self.dilation, self.bias)

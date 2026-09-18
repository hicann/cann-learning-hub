"""PyPTO ConvTranspose2d 模块 (相位分解 + matmul 路线, 与 PyPTOConv2DModule 同风格)

fwd / dK / dX 全部走 PyPTO ``matmul`` (AIC/Cube, FP32)。

转置卷积 (transposed conv / deconv):
    Y[oc, h, w] = Σ_ic Σ_kh Σ_kw X[ic, h_src, w_src] * K[oc, ic, kh, kw]
    h = h_src * s - p + kh,  w = w_src * s - p + kw

对大 stride (s) 采用**相位分解** fwd, 避免 64x64 大核下 im2col 内存爆炸:
    记 r = ceil(K / s)。输出像素 (h, w) 只依赖 r×r 个输入像素:
        h = h'*s + a - p,  a = (h + p) mod s,  h' = (h + p - a) / s
        kh = a + s*m,  m ∈ [0, r)
    ⇒ Y[oc, h'*s+a-p, w'*s+b-p] = Σ_ic Σ_{m,n} X[ic, h'-m, w'-n] * K[oc, ic, a+s*m, b+s*n]

- fwd:   cols = im2col_{r×r}(pad(X, r-1))              (M, Cin*r²)
         out_tmp = cols @ W_full^T                     (M, Cout*s²)  [单次大 matmul]
         Y = pixel_shuffle(out_tmp) 后裁剪有效区
         (W_full 展平为 2D: (Cout*s², Cin*r²), 满足 matmul 同维约束)
- dX:    dX = conv(dY, K^swap, stride=s, padding=p)   (转置卷积 backward 恒等式)
         复用 PyPTOConv2DModule._conv_forward_via_matmul 单次完成
- dK:    dK^swap = conv_wgrad(dY_pad, X)               复用 _conv_dK_via_matmul 单次完成

限制: groups=1, dilation=1, output_padding=0。
"""

import torch
import torch.nn.functional as F
from torch import nn

import pypto

try:
    from .PyPTOConv2DModule import (
        _matmul_bt_kernel, _matmul_at_kernel, _matmul_bt_capped,
        _matmul_at_capped, _im2col_strided, _to_pair, _pad4d, _conv_geom)
except ImportError:
    from PyPTOConv2DModule import (
        _matmul_bt_kernel, _matmul_at_kernel, _matmul_bt_capped,
        _matmul_at_capped, _im2col_strided, _to_pair, _pad4d, _conv_geom)

# pypto.matmul ND 排布内轴上限 65535, 留余量; 大核 (如 FCN 21ch*64*64=86016) 需分块
_INNER_AXIS_MAX = 65000


def _build_W_full(K_conv, r_h, r_w, s_h, s_w):
    """K_conv (Cout, Cin, Kh, Kw) → W_full (Cout*s_h*s_w, Cin*r_h*r_w).

    核零 pad 到 r*s 再切成 (m, a; n, b) 相位块并翻转 m/n 序后展平,
    使单次 matmul 输出列即为 (Cout, a, b) pixel_shuffle 排布。
    """
    Cout, Cin, Kh, Kw = K_conv.shape
    Kpad = F.pad(K_conv, [0, r_w * s_w - Kw, 0, r_h * s_h - Kh])
    Kph = Kpad.reshape(Cout, Cin, r_h, s_h, r_w, s_w).permute(0, 1, 2, 4, 3, 5)
    # (Cout, Cin, m, n, a, b) --flip(m,n)--> 重排 (Cout, a, b, Cin, m, n) 后展平,
    # 行序 = out_tmp 列的 pixel_shuffle 分解 (Cout, a, b), 列序 = im2col 的 (Cin, m, n)
    return Kph.flip(2, 3).permute(0, 4, 5, 1, 2, 3).reshape(
        Cout * s_h * s_w, Cin * r_h * r_w).contiguous()


def _conv_transpose_forward(X, K_conv, bias, stride, padding):
    N, Cin, Hin, Win = X.shape
    Cout = K_conv.shape[0]
    Kh, Kw = K_conv.shape[2], K_conv.shape[3]
    s_h, s_w = _to_pair(stride)
    p_h, p_w = _to_pair(padding)
    r_h, r_w = -(-Kh // s_h), -(-Kw // s_w)
    Hout = (Hin - 1) * s_h - 2 * p_h + Kh
    Wout = (Win - 1) * s_w - 2 * p_w + Kw

    Xg = _pad4d(X, r_h - 1, r_w - 1)
    cols = _im2col_strided(Xg, r_h, r_w, (1, 1))
    W_full = _build_W_full(K_conv, r_h, r_w, s_h, s_w)
    out_tmp = torch.empty(cols.shape[0], Cout * s_h * s_w,
                          device=X.device, dtype=X.dtype)
    _matmul_bt_capped(cols, W_full, out_tmp)

    # (M, Cout*s²) → (N, Hg, Wg, Cout, s_h, s_w) → (N, Cout, Hg*s_h, Wg*s_w)
    Hg, Wg = Xg.shape[2] - r_h + 1, Xg.shape[3] - r_w + 1
    raw = out_tmp.reshape(N, Hg, Wg, Cout, s_h, s_w
                          ).permute(0, 3, 1, 4, 2, 5
                          ).reshape(N, Cout, Hg * s_h, Wg * s_w)
    Y = raw[:, :, p_h:p_h + Hout, p_w:p_w + Wout].contiguous()
    if bias is not None:
        Y = Y + bias.view(1, Cout, 1, 1)
    return Y


def _conv2d_fwd_matmul_chunked(X, K, stride, padding):
    """标准卷积前向 (im2col + 分块 matmul): Y[m, co] = Σ_g cols[:, Kg] @ K[:, Kg]^T.

    与 _conv_forward_via_matmul 相同, 但按输入通道分块使内轴 ≤_INNER_AXIS_MAX,
    各通道块贡献不相交求和。返回 (N*H_out*W_out, Cout)。
    """
    N, Cin, H, W = X.shape
    Cout = K.shape[0]
    kH, kW = K.shape[2], K.shape[3]
    p_h, p_w = _to_pair(padding)
    H_out, W_out = _conv_geom(X.shape, kH, kW, stride, padding)

    X_pad = _pad4d(X.contiguous(), p_h, p_w)
    cols = _im2col_strided(X_pad, kH, kW, stride)          # (M, Cin*kH*kW)
    K_2d = K.reshape(Cout, Cin * kH * kW)
    out = torch.zeros(cols.shape[0], Cout, device=X.device, dtype=X.dtype)
    K2 = kH * kW
    step = max(1, _INNER_AXIS_MAX // K2)
    for c0 in range(0, Cin, step):
        c1 = min(c0 + step, Cin)
        part = torch.empty(cols.shape[0], Cout,
                           device=X.device, dtype=X.dtype)
        _matmul_bt_capped(cols[:, c0 * K2:c1 * K2].contiguous(),
                          K_2d[:, c0 * K2:c1 * K2].contiguous(), part)
        out += part
    return out


def _conv_transpose_dX(grad_Y, K_conv, X_shape, stride, padding):
    """dX = conv(dY, swap(K), stride=s, padding=p) 恒等式 (核通道互换, 不翻转).

    该卷积的输出网格恰为 X 的空间尺寸。
    """
    K_swap = K_conv.permute(1, 0, 2, 3).contiguous()
    N, Cin, H_in, W_in = X_shape
    dX_cols = _conv2d_fwd_matmul_chunked(grad_Y.contiguous(), K_swap,
                                         stride, padding)
    return dX_cols.reshape(N, H_in, W_in, Cin).permute(0, 3, 1, 2).contiguous()


def _conv_transpose_dK(X, grad_Y, K_conv, stride, padding):
    """dW: 以 dY 为卷积输入、X 为输出梯度求权重梯度 (swap 视角).

    恒等式: dW[cin, cout, kh, kw] = Σ gY_pad[cout, i*s+kh, j*s+kw] * X[cin, i, j].
    按输出通道分块防内轴超限。
    """
    N = grad_Y.shape[0]
    Cout_t, Cin_t = grad_Y.shape[1], X.shape[1]
    H_in, W_in = X.shape[2], X.shape[3]
    Kh, Kw = K_conv.shape[2], K_conv.shape[3]
    p_h, p_w = _to_pair(padding)

    # 恒等式: gY_pad 上以 (Kh,Kw) 步长 s 滑窗的窗口数恰为 X 的空间网格数
    gY_pad = _pad4d(grad_Y.contiguous(), p_h, p_w)
    cols = _im2col_strided(gY_pad, Kh, Kw, stride)         # (M, Cout*K²)
    X_2d = X.permute(0, 2, 3, 1).reshape(
        N * H_in * W_in, Cin_t).contiguous()
    K2 = Kh * Kw
    step = max(1, _INNER_AXIS_MAX // K2)
    flat = torch.empty(Cout_t * K2, Cin_t,
                       device=grad_Y.device, dtype=grad_Y.dtype)
    for c0 in range(0, Cout_t, step):
        c1 = min(c0 + step, Cout_t)
        # flat 行序 (cout, kh, kw): cols 列与 flat 行同序, 直接切片
        part = torch.empty((c1 - c0) * K2, Cin_t,
                           device=grad_Y.device, dtype=grad_Y.dtype)
        _matmul_at_capped(cols[:, c0 * K2:c1 * K2].contiguous(),
                          X_2d, part)
        flat[c0 * K2:c1 * K2] = part
    # (Cout, kH, kW, Cin) → (Cin, Cout, kH, kW) = weight 布局
    return flat.reshape(Cout_t, Kh, Kw, Cin_t).permute(3, 0, 1, 2).contiguous()


class PyPTOConvTranspose2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, stride, padding, output_padding, dilation, bias):
        if _to_pair(dilation) != (1, 1):
            raise NotImplementedError("AIC matmul 路线不支持 dilation")
        if _to_pair(output_padding) != (0, 0):
            raise NotImplementedError("AIC matmul 路线不支持 output_padding")
        s_h, s_w = _to_pair(stride)
        X_c, W_c = X.contiguous(), W.contiguous()
        # ConvT weight (Cin, Cout, kH, kW) → conv 视角核 (Cout, Cin, kH, kW)
        K_conv = W_c.permute(1, 0, 2, 3).contiguous()
        Y = _conv_transpose_forward(X_c, K_conv, bias, stride, padding)
        ctx.save_for_backward(X_c, K_conv)
        ctx.params = (s_h, s_w, padding)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, K_conv = ctx.saved_tensors
        s_h, s_w, padding = ctx.params
        grad_Y = grad_output.contiguous()

        dW = (_conv_transpose_dK(X, grad_Y, K_conv, (s_h, s_w), padding)
              if ctx.needs_input_grad[1] else None)
        dX = (_conv_transpose_dX(grad_Y, K_conv, X.shape, (s_h, s_w), padding)
              if ctx.needs_input_grad[0] else None)
        db = (grad_Y.sum(dim=(0, 2, 3)).contiguous()
              if ctx.needs_input_grad[6] else None)
        return dX, dW, None, None, None, None, db


class PyPTOConvTranspose2d(nn.Module):
    """PyPTO 二维转置卷积 (相位分解 + AIC matmul, FP32): fwd/dK/dX 全 PyPTO."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, device="cpu", dtype=torch.float32):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("AIC matmul 路线不支持 groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(
            tuple([in_channels, out_channels]) + self.kernel_size,
            device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=2.23606797749979)  # sqrt(5)
        self.bias = (nn.Parameter(torch.zeros(out_channels, device=device,
                                              dtype=dtype))
                     if bias else None)

    def forward(self, x):
        return PyPTOConvTranspose2dFunction.apply(
            x, self.weight, self.stride, self.padding,
            self.output_padding, self.dilation, self.bias)

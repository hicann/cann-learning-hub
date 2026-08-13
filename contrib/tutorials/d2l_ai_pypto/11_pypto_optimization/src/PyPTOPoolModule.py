"""PyPTO Pool2d v2 — AvgPool2d / MaxPool2d (fwd + bwd)

基于 custom_module_pool / PyPTOPoolModule 重构, 参考 conv v3 风格:
- **快路径 (stride==kernel 且无 padding 且整除)**: reshape + sum/amax 归约,
  合并 N*C 维 (替代旧版 per-n host 展开, 减少 N 倍 dispatch):
    avg fwd: reshape (NC,H,W)→(NC,H/kh,kh,W)→sum(dim=2)→(NC,H/kh,W/kw,kw)→sum(dim=3)
    avg bwd: expand_clone 上采样 kh×kw 后除面积
    max fwd: 同 avg fwd 但 amax
  整个 kernel 无循环, 固定 ~7 个 op, 性能与 torch 相当。
- **通用路径 (重叠/padding)**: host im2col (as_strided, 与 conv v3 同款) +
  torch 归约。PyPTO vector 2D 归约 (sum/amax) 在 M 大时框架级退化
  (实测 (519168, 9) 归约 285ms vs torch 0.4ms), 故归约在 host 完成。
  旧版 4 层逐点嵌套循环 ~100x 慢, 且其 _pad4d_fast 在新 pypto 下编译失败。
- **bwd 通用路径**: host col2im 散射累加 (index_add_; as_strided view +=
  在 torch_npu 上会被 materialize, 重叠累加错误), 支持任意重叠窗口。
  max bwd 用 host argmax 定位 (fwd 无额外开销)。

性能实测 (bs=32, vs torch.nn.AvgPool2d):
  快路径 fwd 2.3x / bwd 2.4x; 通用路径 fwd ~2.8x (kernel launch 固定开销)。
"""

import torch
import torch.nn.functional as F
from torch import nn

import pypto


# ── PyPTO kernels ──


@pypto.frontend.jit()
def _avgpool_fwd_fast_kernel(
    X:   pypto.Tensor([], pypto.DT_FP32),   # (N, C, H, W)
    out: pypto.Tensor([], pypto.DT_FP32),   # (N, C, H//kh, W//kw)
    p_h: int,
    p_w: int,
    area: float,
):
    """非重叠 avg pool fwd: reshape + 两次 sum + 除面积 (合并 N*C)."""
    N, C, H, W = X.shape
    NC = N * C
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    v = pypto.reshape(X, [NC, H, W])
    r1 = pypto.reshape(v, [NC, H // p_h, p_h, W])
    s1 = pypto.sum(r1, dim=2, keepdim=True)
    r2 = pypto.reshape(s1, [NC, H // p_h, W // p_w, p_w])
    s2 = pypto.sum(r2, dim=3, keepdim=True)
    r3 = pypto.reshape(s2, [NC, H // p_h, W // p_w])
    g = pypto.reshape(r3, [N, C, H // p_h, W // p_w])
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    out.move(pypto.div(g, area))


@pypto.frontend.jit()
def _avgpool_bwd_fast_kernel(
    grad_Y: pypto.Tensor([], pypto.DT_FP32),   # (N, C, H_out, W_out)
    out:    pypto.Tensor([], pypto.DT_FP32),   # (N, C, H_out*kh, W_out*kw)
    p_h: int,
    p_w: int,
    area: float,
):
    """非重叠 avg pool bwd: grad_Y 上采样 kh×kw 后除面积 (合并 N*C)."""
    N, C, H_out, W_out = grad_Y.shape
    NC = N * C
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    v = pypto.reshape(grad_Y, [NC, H_out, W_out])
    r1 = pypto.reshape(v, [NC, H_out, W_out, 1])
    e1 = pypto.expand_clone(r1, [NC, H_out, W_out, p_w])
    r2 = pypto.reshape(e1, [NC, H_out, W_out * p_w])
    e2 = pypto.expand_clone(
        pypto.reshape(r2, [NC, H_out, 1, W_out * p_w]),
        [NC, H_out, p_h, W_out * p_w])
    r3 = pypto.reshape(e2, [NC, H_out * p_h, W_out * p_w])
    g_up = pypto.reshape(r3, [N, C, H_out * p_h, W_out * p_w])
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    out.move(pypto.div(g_up, area))


@pypto.frontend.jit()
def _maxpool_fwd_fast_kernel(
    X:   pypto.Tensor([], pypto.DT_FP32),   # (N, C, H, W)
    out: pypto.Tensor([], pypto.DT_FP32),   # (N, C, H//kh, W//kw)
    p_h: int,
    p_w: int,
):
    """非重叠 max pool fwd: reshape + 两次 amax (合并 N*C)."""
    N, C, H, W = X.shape
    NC = N * C
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    v = pypto.reshape(X, [NC, H, W])
    r1 = pypto.reshape(v, [NC, H // p_h, p_h, W])
    s1 = pypto.amax(r1, dim=2, keepdim=True)
    r2 = pypto.reshape(s1, [NC, H // p_h, W // p_w, p_w])
    s2 = pypto.amax(r2, dim=3, keepdim=True)
    r3 = pypto.reshape(s2, [NC, H // p_h, W // p_w])
    g = pypto.reshape(r3, [N, C, H // p_h, W // p_w])
    pypto.set_vec_tile_shapes(8, 8, 8, 8)
    out.move(g)


# ── host 辅助: im2col / col2im 散射累加 ──


def _im2col(X_pad, kH, kW, s_h, s_w):
    """X_pad (N,C,Hp,Wp) → (M, kH*kW), M=N*C*H_out*W_out (as_strided 零拷贝 + 一次 contiguous)."""
    N, C, Hp, Wp = X_pad.shape
    Ho = (Hp - kH) // s_h + 1
    Wo = (Wp - kW) // s_w + 1
    view = X_pad.as_strided(
        (N, C, Ho, Wo, kH, kW),
        (C * Hp * Wp, Hp * Wp, s_h * Wp, s_w, Wp, 1),
    )
    return view.reshape(N * C * Ho * Wo, kH * kW).contiguous()


def _col2im_add(grad_cols, N, C, Hp, Wp, kH, kW, s_h, s_w):
    """grad_cols (M, kH*kW) 散射累加到 (N, C, Hp, Wp).

    用 index_add_ 而非 as_strided view +=: 实测 torch_npu 的 as_strided 写视图
    会被 materialize (重叠窗口只加最后一次, 累加错误)。
    """
    Ho = (Hp - kH) // s_h + 1
    Wo = (Wp - kW) // s_w + 1
    M = Ho * Wo
    K = kH * kW
    dev = grad_cols.device
    # 每窗口内 K 个位置的平面内偏移: idx[k, m]
    kh_off = (torch.arange(kH, device=dev) * Wp).view(kH, 1, 1, 1)
    kw_off = torch.arange(kW, device=dev).view(1, kW, 1, 1)
    ho_off = (torch.arange(Ho, device=dev) * s_h * Wp).view(1, 1, Ho, 1)
    wo_off = (torch.arange(Wo, device=dev) * s_w).view(1, 1, 1, Wo)
    idx = (kh_off + kw_off + ho_off + wo_off).reshape(K, M)   # (K, M)
    idx_flat = idx.t().reshape(-1).unsqueeze(0)               # (1, M*K), M-major
    plane_off = torch.arange(N * C, device=dev).view(N * C, 1) * (Hp * Wp)
    idx_full = (idx_flat + plane_off).reshape(-1)             # (N*C*M*K,)
    out = torch.zeros(N * C * Hp * Wp, device=dev, dtype=grad_cols.dtype)
    out.index_add_(0, idx_full, grad_cols.reshape(-1))
    return out.reshape(N, C, Hp, Wp)


def _pool_geom(H, W, kH, kW, s_h, s_w, pad):
    return (H + 2 * pad - kH) // s_h + 1, (W + 2 * pad - kW) // s_w + 1


def _is_fast(kH, kW, s_h, s_w, pad, H, W):
    return s_h == kH and s_w == kW and pad == 0 and H % kH == 0 and W % kW == 0


def _to_pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


# ── avg pool ──


def _avg_pool_forward(X, kH, kW, s_h, s_w, pad):
    N, C, H, W = X.shape
    H_out, W_out = _pool_geom(H, W, kH, kW, s_h, s_w, pad)
    Y = torch.zeros(N, C, H_out, W_out, device=X.device, dtype=X.dtype)
    area = float(kH * kW)
    Xc = X.contiguous()
    if _is_fast(kH, kW, s_h, s_w, pad, H, W):
        _avgpool_fwd_fast_kernel(Xc, Y, kH, kW, area)
    else:
        X_pad = F.pad(Xc, [pad, pad, pad, pad])
        cols = _im2col(X_pad, kH, kW, s_h, s_w)          # (M, kH*kW)
        Y = (cols.sum(dim=1, keepdim=True) / area).reshape(N, C, H_out, W_out)
    return Y


def _avg_pool_backward(grad_Y, input_shape, kH, kW, s_h, s_w, pad):
    N, C, H, W = input_shape
    H_out, W_out = grad_Y.shape[2], grad_Y.shape[3]
    area = float(kH * kW)
    grad_X = torch.zeros(N, C, H + 2 * pad, W + 2 * pad,
                         device=grad_Y.device, dtype=grad_Y.dtype)
    gY = grad_Y.contiguous()
    if _is_fast(kH, kW, s_h, s_w, pad, H, W):
        _avgpool_bwd_fast_kernel(gY, grad_X, kH, kW, area)
    else:
        # 通用: 每个窗口内梯度均分 → col2im 散射累加
        grad_cols = (gY / area).reshape(N, C, H_out, W_out, 1).expand(
            N, C, H_out, W_out, kH * kW).reshape(N * C * H_out * W_out, kH * kW).contiguous()
        grad_X = _col2im_add(grad_cols, N, C, H + 2 * pad, W + 2 * pad,
                             kH, kW, s_h, s_w)
    if pad > 0:
        grad_X = grad_X[:, :, pad:pad + H, pad:pad + W]
    return grad_X


# ── max pool ──


def _max_pool_forward(X, kH, kW, s_h, s_w, pad):
    N, C, H, W = X.shape
    H_out, W_out = _pool_geom(H, W, kH, kW, s_h, s_w, pad)
    Xc = X.contiguous()
    if _is_fast(kH, kW, s_h, s_w, pad, H, W):
        Y = torch.zeros(N, C, H_out, W_out, device=X.device, dtype=X.dtype)
        _maxpool_fwd_fast_kernel(Xc, Y, kH, kW)
    else:
        X_pad = F.pad(Xc, [pad, pad, pad, pad], value=float("-inf"))
        cols = _im2col(X_pad, kH, kW, s_h, s_w)
        Y = cols.amax(dim=1, keepdim=True).reshape(N, C, H_out, W_out)
    return Y


def _max_pool_backward(grad_Y, X, kH, kW, s_h, s_w, pad):
    """host 实现: 重算 im2col, argmax 定位 → one-hot 掩码 → col2im 散射."""
    N, C, H, W = X.shape
    H_out, W_out = grad_Y.shape[2], grad_Y.shape[3]
    X_pad = F.pad(X.contiguous(), [pad, pad, pad, pad], value=float("-inf"))
    cols = _im2col(X_pad, kH, kW, s_h, s_w)              # (M, kH*kW)
    idx = cols.argmax(dim=1)                             # 每窗口最大值位置
    grad_cols = F.one_hot(idx, kH * kW).to(grad_Y.dtype) * \
        grad_Y.contiguous().reshape(-1, 1)
    grad_X = _col2im_add(grad_cols, N, C, H + 2 * pad, W + 2 * pad,
                         kH, kW, s_h, s_w)
    if pad > 0:
        grad_X = grad_X[:, :, pad:pad + H, pad:pad + W]
    return grad_X


# ── autograd Function + nn.Module ──


class PyPTOAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, kH, kW, s_h, s_w, pad):
        ctx.input_shape = X.shape
        ctx.params = (kH, kW, s_h, s_w, pad)
        return _avg_pool_forward(X, kH, kW, s_h, s_w, pad)

    @staticmethod
    def backward(ctx, grad_output):
        return (_avg_pool_backward(grad_output, ctx.input_shape, *ctx.params),
                None, None, None, None, None)


class PyPTOMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, kH, kW, s_h, s_w, pad):
        ctx.save_for_backward(X)
        ctx.params = (kH, kW, s_h, s_w, pad)
        return _max_pool_forward(X, kH, kW, s_h, s_w, pad)

    @staticmethod
    def backward(ctx, grad_output):
        X, = ctx.saved_tensors
        return (_max_pool_backward(grad_output, X, *ctx.params),
                None, None, None, None, None)


class PyPTOAvgPool2d(nn.Module):
    """PyPTO 二维平均汇聚 (stride==kernel 走 PyPTO 快路径, 其余走 im2col 通用路径)."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_pair(kernel_size)
        self.stride = _to_pair(stride) if stride is not None else self.kernel_size
        self.padding = _to_pair(padding)

    def forward(self, x):
        kH, kW = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        if p_h != p_w:
            raise NotImplementedError("对称 padding 仅支持单值")
        return PyPTOAvgPool2dFunction.apply(x, kH, kW, s_h, s_w, p_h)


class PyPTOMaxPool2d(nn.Module):
    """PyPTO 二维最大汇聚 (fwd 快路径 PyPTO, bwd host argmax 散射)."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_pair(kernel_size)
        self.stride = _to_pair(stride) if stride is not None else self.kernel_size
        self.padding = _to_pair(padding)

    def forward(self, x):
        kH, kW = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        if p_h != p_w:
            raise NotImplementedError("对称 padding 仅支持单值")
        return PyPTOMaxPool2dFunction.apply(x, kH, kW, s_h, s_w, p_h)

"""PyPTO AIV Conv2d (LeNet) — 纯向量实现, 不使用 AIC (Cube)

支持 forward / dK / dX / dbias 全部用 PyPTO 向量算子实现。

设计 (window-stack, 详见 README.md):
1. 窗口 view 覆盖 Cin 轴: view(padded, [1, Cin, ROWS, WW], [n, 0, h, kw])
   → 4D view 的 dim1 映射到 parent 的 dim1 (通道轴) ✓
2. concat (kH*kW 个) → (kH*kW, Cin, ROWS, WW) → reshape (T, 1, ROWS, WW),
   T = Cin*kH*kW (tap-major, 顺序 (kh,kw,ic), 与 k_host 一致)
3. 单轴广播 mul (ws 扩 dim1, k 扩 dim2) + sum(dim=0) 归约 tap 轴
4. kernel 权重 host 预展开: K → (T, Cout, 1, WW)
5. 32B 对齐窗口 (WW=32), 统一对齐 tile
6. 输出 2D 写回 (N*Cout*H_out, W_out), 每 (n,oc,chunk) 一次 2D assemble
7. 大 Cout/T 场景: oc 分组循环 (mul 输出 (T,G,ROWS,WW) 需满足 UB 限制)

已知框架限制 (实测, 详见 README):
- 隐式广播/expand_clone 只允许单轴扩展
- 4D 多通道/多行 assemble 只写第一个 slice → 必须 2D 写回 + 逐 oc
- concat/2D assemble 的 tile 尾轴需 32B 对齐
- 不能把"重新赋值的变量"传给嵌套 frontend 函数 (base 为空报 Empty tensor)
- move 到 strided view 写全零

限制: stride=1; W_out <= WW; H_out/H 需能被 ROWS 整除。
"""

import os
import sys

import torch
from torch import nn

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import pypto

# ── 可调参数 (环境变量) ──
WW = int(os.environ.get("PTO_WW", "32"))                # 对齐窗口宽度
ROWS_FWD = int(os.environ.get("PTO_ROWS_FWD", "28"))    # forward 每 chunk 行数
ROWS_DK = int(os.environ.get("PTO_ROWS_DK", "28"))      # dK 每 chunk 行数
ROWS_DX = int(os.environ.get("PTO_ROWS_DX", "28"))      # dX 每 chunk 行数 (多 chunk 结果错误, 恒取 H)
CT2 = int(os.environ.get("PTO_CT2", "2"))               # 计算 tile 的 ROWS 切分上限
OC_GROUPS = int(os.environ.get("PTO_OC_GROUPS", "0"))   # 0=自动

_UB_BYTES = 192 * 1024


def _debug_options():
    return {"runtime_debug_mode": 1} if os.environ.get("PTO_DEBUG", "") == "1" else {}


def _pick_rows(h_out, rows_env):
    rows = rows_env if h_out % rows_env == 0 else h_out
    return rows


# ── 快速 pad: concat 方案 + 大 tile ──


@pypto.frontend.function
def _pad4d_fast(
    X:   pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
):
    Np = X.shape[0]
    Cp = X.shape[1]
    Hp = X.shape[2]
    Wp = X.shape[3]
    pypto.set_vec_tile_shapes(8, 1, 8, 32)
    cur = X.clone()
    cur_h = Hp
    cur_w = Wp
    if pad_top > 0:
        zero_t = pypto.zeros(Np, Cp, pad_top, cur_w, dtype=pypto.DT_FP32)
        cur = pypto.concat([zero_t, cur], dim=2)
        cur_h = cur_h + pad_top
    if pad_bottom > 0:
        zero_b = pypto.zeros(Np, Cp, pad_bottom, cur_w, dtype=pypto.DT_FP32)
        cur = pypto.concat([cur, zero_b], dim=2)
        cur_h = cur_h + pad_bottom
    if pad_left > 0:
        zero_l = pypto.zeros(Np, Cp, cur_h, pad_left, dtype=pypto.DT_FP32)
        cur = pypto.concat([zero_l, cur], dim=3)
        cur_w = cur_w + pad_left
    if pad_right > 0:
        zero_r = pypto.zeros(Np, Cp, cur_h, pad_right, dtype=pypto.DT_FP32)
        cur = pypto.concat([cur, zero_r], dim=3)
        cur_w = cur_w + pad_right
    out.move(cur)


def _build_windows(input, Cin, ROWS, WW_, kH, kW, n_idx, ohc):
    """窗口堆叠: kH*kW 个 (1,Cin,ROWS,WW) view → concat → reshape (T,1,ROWS,WW)"""
    # concat 阶段 tile: dim0=kH*kW, dim1=Cin, dim2 自适应避免 UB 溢出
    per_row = kH * kW * Cin * WW_ * 4
    ct2c = max(1, min(ROWS, (_UB_BYTES - 32 * 1024) // per_row))
    pypto.set_vec_tile_shapes(kH * kW, Cin, ct2c, 32)
    wins = []
    for kh in range(kH):
        for kw in range(kW):
            wins.append(pypto.view(
                input, [1, Cin, ROWS, WW_],
                [n_idx, 0, ohc * ROWS + kh, kw],
            ))
    ws4 = pypto.concat(wins, 0)  # (kH*kW, Cin, ROWS, WW)
    T = Cin * kH * kW
    ws = pypto.reshape(ws4, [T, 1, ROWS, WW_])  # (T, 1, ROWS, WW), 顺序 (kh,kw,ic)
    return ws


# ── forward ──


@pypto.frontend.function
def _conv2d_fwd(
    input:  pypto.Tensor([], pypto.DT_FP32),    # padded (N, Cin, Hp, Wp)
    k_host: pypto.Tensor([], pypto.DT_FP32),    # (T, Cout, 1, WW)
    output: pypto.Tensor([], pypto.DT_FP32),    # (N*Cout*H_out, W_out) 2D
    kH: int,
    kW: int,
    W_out: int,
    oc_groups: int,
    ct2: int,
):
    N = input.shape[0]
    Cin = input.shape[1]
    Cout = k_host.shape[1]
    T = k_host.shape[0]
    WW_ = k_host.shape[3]
    H_out = (input.shape[2] - kH) // 1 + 1
    G = Cout // oc_groups
    ROWS = _pick_rows(H_out, ROWS_FWD)

    for n_idx in range(N):
        for ohc in pypto.loop(H_out // ROWS, name="LOOP_chunk", idx_name="ohc"):
            pypto.set_vec_tile_shapes(kH * kW, Cin, ROWS, 32)
            ws = _build_windows(input, Cin, ROWS, WW_, kH, kW, n_idx, ohc)
            pypto.set_vec_tile_shapes(T, G, ct2, 32)
            for g in range(oc_groups):
                k_g = pypto.view(k_host, [T, G, 1, WW_], [0, g * G, 0, 0])
                m = pypto.mul(ws, k_g)  # ws 扩 dim1 (1->G), k_g 扩 dim2 (1->ROWS)
                s = pypto.sum(m, 0, keepdim=False)  # (G, ROWS, WW)
                s2 = pypto.reshape(s, [G * ROWS, WW_])
                pypto.set_vec_tile_shapes(ROWS, 32)
                for oc in range(G):
                    acc_oc = pypto.view(s2, [ROWS, W_out], [oc * ROWS, 0])
                    pypto.assemble(
                        acc_oc,
                        [n_idx * Cout * H_out + (g * G + oc) * H_out + ohc * ROWS, 0],
                        output,
                    )
                pypto.set_vec_tile_shapes(T, G, ct2, 32)


@pypto.frontend.jit(
    debug_options=_debug_options(),
    runtime_options={},
)
def conv2d_fwd_kernel(
    input:  pypto.Tensor([], pypto.DT_FP32),
    k_host: pypto.Tensor([], pypto.DT_FP32),
    output: pypto.Tensor([], pypto.DT_FP32),
    kH: int,
    kW: int,
    pad: int,
    oc_groups: int,
    ct2: int,
):
    N = input.shape[0]
    Cin = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    pypto.set_vec_tile_shapes(8, 1, 8, 32)
    W_out = W + 2 * pad - kW + 1

    src = input
    if pad > 0:
        p_wr = max(pad, kW + WW - 1 - W - pad)
        padded = pypto.zeros(N, Cin, H + 2 * pad,
                             W + pad + p_wr, dtype=pypto.DT_FP32)
        _pad4d_fast(src, padded, pad, pad, pad, p_wr)
        _conv2d_fwd(padded, k_host, output, kH, kW, W_out, oc_groups, ct2)
    else:
        p_wr = max(0, kW + WW - 1 - W)
        padded = pypto.zeros(N, Cin, H, W + p_wr, dtype=pypto.DT_FP32)
        _pad4d_fast(src, padded, 0, 0, 0, p_wr)
        _conv2d_fwd(padded, k_host, output, kH, kW, W_out, oc_groups, ct2)


# ── dK ──


@pypto.frontend.function
def _conv2d_dk(
    input:  pypto.Tensor([], pypto.DT_FP32),    # padded X
    dy:     pypto.Tensor([], pypto.DT_FP32),    # padded dY (N, Cout, H_out, Wp)
    dk2:    pypto.Tensor([], pypto.DT_FP32),    # (Cout, T) 输出
    kH: int,
    kW: int,
    oc_groups: int,
):
    N = input.shape[0]
    Cin = input.shape[1]
    Cout = dy.shape[1]
    T = Cin * kH * kW
    WW_ = dy.shape[3]
    H_out = (input.shape[2] - kH) // 1 + 1
    G = Cout // oc_groups
    # 注意: pypto.loop 不支持跨迭代携带累加 tensor, chunk 循环用 Python range 展开。
    # tile (T,G,1,32): ROWS 被切分, sum(dim2) 为跨 tile 归约 (正确但偏慢, 已知限制:
    # tile (T,1,ROWS,32) 会触发 TExpand codegen bug, 无法使用)。
    ROWS = _pick_rows(H_out, ROWS_DK)

    pypto.set_vec_tile_shapes(T, G, 1, 32)
    dk_accs = []
    for g in range(oc_groups):
        dk_accs.append(pypto.zeros(T, G, 1, 1, dtype=pypto.DT_FP32))
    for n_idx in range(N):
        for ohc in range(H_out // ROWS):
            pypto.set_vec_tile_shapes(kH * kW, Cin, ROWS, 32)
            ws = _build_windows(input, Cin, ROWS, WW_, kH, kW, n_idx, ohc)
            pypto.set_vec_tile_shapes(T, G, 1, 32)
            for g in range(oc_groups):
                dy_v = pypto.view(dy, [1, G, ROWS, WW_], [n_idx, g * G, ohc * ROWS, 0])
                m = pypto.mul(ws, dy_v)  # ws 扩 dim1, dy_v 扩 dim0, 各单轴 ✓
                s1 = pypto.sum(m, 2, keepdim=True)  # (T, G, 1, WW)
                s2 = pypto.sum(s1, 3, keepdim=True)  # (T, G, 1, 1)
                dk_accs[g] = pypto.add(dk_accs[g], s2)
    pypto.set_vec_tile_shapes(T, Cout, 1, 32)
    if oc_groups == 1:
        dk_acc = dk_accs[0]  # 单组时 concat 单元素精度不保证
    else:
        dk_acc = pypto.concat(dk_accs, 1)  # (T, Cout, 1, 1)
    pypto.set_vec_tile_shapes(8, 8)
    dk_acc2 = pypto.reshape(dk_acc, [T, Cout])
    dk_t = pypto.transpose(dk_acc2, 0, 1)  # (Cout, T)
    pypto.set_vec_tile_shapes(ROWS, 32)
    dk2.move(dk_t)


@pypto.frontend.jit(
    debug_options=_debug_options(),
    runtime_options={},
)
def conv2d_dk_kernel(
    input:  pypto.Tensor([], pypto.DT_FP32),
    dy:     pypto.Tensor([], pypto.DT_FP32),
    dk2:    pypto.Tensor([], pypto.DT_FP32),
    kH: int,
    kW: int,
    pad: int,
    oc_groups: int,
):
    N = input.shape[0]
    Cin = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    Cout = dy.shape[1]
    Hg = dy.shape[2]
    Wg = dy.shape[3]
    pypto.set_vec_tile_shapes(8, 1, 8, 32)

    p_wr = max(0, kW + WW - 1 - W - pad)  # 窗口对齐所需右 pad (W < kW+WW-1 时)
    padded = pypto.zeros(N, Cin, H + 2 * pad,
                         W + pad + p_wr, dtype=pypto.DT_FP32)
    _pad4d_fast(input, padded, pad, pad, pad, p_wr)

    dy_pad_w = max(WW + kW - 1, Wg)
    dy_pad = pypto.zeros(N, Cout, Hg, dy_pad_w, dtype=pypto.DT_FP32)
    _pad4d_fast(dy, dy_pad, 0, 0, 0, dy_pad_w - Wg)

    _conv2d_dk(padded, dy_pad, dk2, kH, kW, oc_groups)


# ── dX ──


@pypto.frontend.function
def _conv2d_dx(
    dy_pad: pypto.Tensor([], pypto.DT_FP32),    # padded dY (N, Cout, Hp, Wp)
    k_dx:   pypto.Tensor([], pypto.DT_FP32),    # (T2, Cin, 1, WW), T2=kH*kW*Cout
    dx2:    pypto.Tensor([], pypto.DT_FP32),    # (N*Cin*H, W) 输出
    kH: int,
    kW: int,
    Hin: int,
    Win: int,
):
    N = dy_pad.shape[0]
    Cout = dy_pad.shape[1]
    Cin = k_dx.shape[1]
    T2 = k_dx.shape[0]
    WW_ = k_dx.shape[3]
    ROWS = _pick_rows(Hin, ROWS_DX)
    ROWS = Hin  # 多 chunk 的 sum 部分和合并结果错误 (框架限制), 恒单 chunk

    for n_idx in range(N):
        for ohc in range(H // ROWS):
            pypto.set_vec_tile_shapes(kH * kW, Cout, ROWS, 32)
            ws = _build_windows(dy_pad, Cout, ROWS, WW_, kH, kW, n_idx, ohc)
            pypto.set_vec_tile_shapes(T2, 1, 1, 32)
            for ic in range(Cin):
                k_ic = pypto.view(k_dx, [T2, 1, 1, WW_], [0, ic, 0, 0])
                m = pypto.mul(ws, k_ic)  # k_ic 扩展 dim2 (1->ROWS) ✓
                s = pypto.sum(m, 0, keepdim=False)  # (1, ROWS, WW)
                s2 = pypto.reshape(s, [ROWS, WW_])
                pypto.set_vec_tile_shapes(ROWS, 32)
                acc_ic = pypto.view(s2, [ROWS, Win], [0, 0])
                pypto.assemble(
                    acc_ic,
                    [(n_idx * Cin + ic) * Hin + ohc * ROWS, 0],
                    dx2,
                )
                pypto.set_vec_tile_shapes(T2, 1, 1, 32)


@pypto.frontend.jit(
    debug_options=_debug_options(),
    runtime_options={},
)
def conv2d_dx_kernel(
    dy:     pypto.Tensor([], pypto.DT_FP32),
    k_dx:   pypto.Tensor([], pypto.DT_FP32),
    dx2:    pypto.Tensor([], pypto.DT_FP32),
    kH: int,
    kW: int,
    pad: int,
):
    N = dy.shape[0]
    Cout = dy.shape[1]
    Hg = dy.shape[2]
    Wg = dy.shape[3]
    Cin = k_dx.shape[1]
    # dX 尺寸由 dY 推导: H = Hg + 2*p_h - kH + 1 (前端对 dx2.shape 表达式求值有误)
    p_h = kH - 1 - pad
    p_w = kW - 1 - pad
    H = Hg + 2 * p_h - kH + 1
    W = Wg + 2 * p_w - kW + 1
    pypto.set_vec_tile_shapes(8, 1, 8, 32)

    dy_pad_h = Hg + 2 * p_h
    dy_pad_w = max(Wg + 2 * p_w, WW + kW - 1)
    dy_pad = pypto.zeros(N, Cout, dy_pad_h, dy_pad_w, dtype=pypto.DT_FP32)
    _pad4d_fast(dy, dy_pad, p_h, p_h, p_w, dy_pad_w - Wg - p_w)

    _conv2d_dx(dy_pad, k_dx, dx2, kH, kW, H, W)


# ── dbias ──


@pypto.frontend.jit(
    debug_options=_debug_options(),
    runtime_options={},
)
def conv2d_dbias_kernel(
    dy:  pypto.Tensor([], pypto.DT_FP32),
    db:  pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 1, 8, 32)
    Cout = dy.shape[1]
    s0 = pypto.sum(dy, 0, keepdim=True)    # (1, Cout, Hg, Wg)
    s1 = pypto.sum(s0, 2, keepdim=True)    # (1, Cout, 1, Wg)
    s2 = pypto.sum(s1, 3, keepdim=True)    # (1, Cout, 1, 1)
    s3 = pypto.reshape(s2, [Cout])
    db.move(s3)


# ── host 侧预处理 ──


def make_k_host(K, kH, kW, wwin=WW):
    """K (Cout, Cin, kH, kW) → (T, Cout, 1, WW), T=Cin*kH*kW, 顺序 (kh,kw,ic)"""
    Cout, Cin, _, _ = K.shape
    K_t = K.permute(2, 3, 1, 0).contiguous()   # (kH, kW, Cin, Cout)
    T = Cin * kH * kW
    K_flat = K_t.reshape(T, Cout).contiguous()
    return K_flat[:, :, None, None].expand(T, Cout, 1, wwin).contiguous()


def make_k_dx(K, kH, kW, wwin=WW):
    """K (Cout, Cin, kH, kW) → (T2, Cin, 1, WW), T2=kH*kW*Cout, 顺序 (kh,kw,oc)
    dX 使用旋转核: dX = full_conv(dY, K_rot)"""
    Cout, Cin, _, _ = K.shape
    K_rot = torch.flip(K, [2, 3]).contiguous()   # rot180
    K_t = K_rot.permute(2, 3, 0, 1).contiguous()  # (kH, kW, Cout, Cin)
    T2 = kH * kW * Cout
    K_flat = K_t.reshape(T2, Cin).contiguous()
    return K_flat[:, :, None, None].expand(T2, Cin, 1, wwin).contiguous()


def pick_oc_groups(T, Cout, wwin=WW):
    """选择 oc 分组使 mul 的 tile (T, G, 1, WW) 满足 UB (保守: 预留 ws/k/acc 空间)"""
    if OC_GROUPS > 0:
        return OC_GROUPS
    per_tile = T * wwin * 4
    max_g = max(1, (_UB_BYTES - 128 * 1024) // per_tile)
    for g in (1, 2, 4, 8, 16, 32, 64):
        if Cout % g == 0 and Cout // g <= max_g:
            return g
    return max(1, Cout // max_g)


def pick_ct2(T, G, wwin=WW):
    """选择计算 tile 的 ROWS 切分使 (T, G, ct2, WW) 满足 UB (保守)"""
    per_row = T * G * wwin * 4
    max_ct2 = max(1, (_UB_BYTES - 96 * 1024) // per_row)
    return max(1, min(CT2, max_ct2))


# ── nn.Module (autograd Function, forward/backward 全 PyPTO) ──


def _pad_wr(W, k, p, wwin=WW):
    """窗口 32B 对齐所需的右 pad"""
    return max(p, k + wwin - 1 - W - p)


class PyPTOConv2dAIVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, K, bias, kH, kW, pad):
        N, Cin, H, W = X.shape
        Cout = K.shape[0]
        H_out = H + 2 * pad - kH + 1
        W_out = W + 2 * pad - kW + 1
        T = Cin * kH * kW
        og = pick_oc_groups(T, Cout)
        ct2 = pick_ct2(T, Cout // og)

        K_host = make_k_host(K, kH, kW)
        Y = torch.zeros(N, Cout, H_out, W_out, device=X.device, dtype=X.dtype)
        Y2 = Y.view(N * Cout * H_out, W_out)
        conv2d_fwd_kernel(X, K_host, Y2, kH, kW, pad, og, ct2)
        if bias is not None:
            Y = Y + bias.view(1, -1, 1, 1)

        ctx.save_for_backward(X, K, bias)
        ctx.params = (kH, kW, pad)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        X, K, bias = ctx.saved_tensors
        kH, kW, pad = ctx.params
        N, Cin, H, W = X.shape
        Cout = K.shape[0]
        T = Cin * kH * kW
        og = pick_oc_groups(T, Cout)

        dY = grad_Y.contiguous()
        dX = torch.zeros_like(X)
        dX2 = dX.view(N * Cin * H, W)
        K_dx = make_k_dx(K, kH, kW)
        conv2d_dx_kernel(dY, K_dx, dX2, kH, kW, pad)

        dK = torch.zeros_like(K)
        dK2 = torch.zeros(Cout, T, device=X.device, dtype=X.dtype)
        conv2d_dk_kernel(X, dY, dK2, kH, kW, pad, og)
        dK.copy_(dK2.reshape(Cout, kH, kW, Cin).permute(0, 3, 1, 2))

        db = None
        if bias is not None:
            db = torch.zeros_like(bias)
            conv2d_dbias_kernel(dY, db)
        return dX, dK, db, None, None, None


class PyPTOConv2d(nn.Module):
    """PyPTO AIV Conv2d (stride=1, 对称 padding), forward/backward 全 PyPTO 向量算子"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 bias=True, device="npu:0", dtype=torch.float32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.padding = padding
        kH, kW = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kH, kW,
                                               device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_channels * kH * kW
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        kH, kW = self.kernel_size
        return PyPTOConv2dAIVFunction.apply(x, self.weight, self.bias, kH, kW, self.padding)

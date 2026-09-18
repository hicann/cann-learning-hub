"""PyPTO Conv2d 模块 v4 (混合路线: 原生 conv 引擎 + im2col/matmul) — 7.6/7.7 节 ResNet/DenseNet 专用优化

问题背景
--------
v3 (im2col + pypto matmul) 在 ResNet/DenseNet 形状 (batch=4, 96x96 输入) 下
训练过慢: 7x7 s2 8.1ms/步、3x3 s1 24x24 1.56ms/步、3x3 512ch 4.15ms/步,
10 epochs x 15 万步 实测 50+ 分钟跑不完。

瓶颈分析 (NPU 实测分解, 见 /tmp/opencode/bench_conv/REPORT.md):
1. host im2col 是 gather 拷贝, 带宽仅 ~20-50GB/s:
   - 3x3 s1 96x96: 85MB 拷贝 4.3ms (整层 fwd+bwd 才 13.3ms)
   - 7x7 dX: 453MB 拷贝 4.9ms
2. dK 的 a_trans matmul (K=36864) 编译期循环展开爆炸 (>5min 编译)
3. matmul tile [128,128]/[64,256]/[256,256] 对 FP32 小 N 形状不匹配
   (nL1=256 > Cout, 只能靠框架 clamp); 实测 pypto matmul 各 tile 配置
   全部 ~0.15ms (kernel 启动开销主导), 调 tile 收益 <0.02ms
4. torch_npu 的 conv/conv_transpose2d 是原生 kernel:
   fwd+bwd 0.2-0.7ms; torch convT (dX) 0.06-0.31ms

方案 (两路线对比后混合, 见 REPORT.md 对比表)
-----------------------------------------------
- fwd 路线 b (原生 pypto.conv, 隐式 GEMM, 无 im2col 膨胀):
  大 spatial (Ho*Wo>=2304, 即 7x7 conv1 的 48x48 输出与 96x96 输入) 与
  (Ho*Wo==576 且 Cout<=128) 的形状。L1/L0/vec tile 预算式推导
  (L1 192KB: 输入 tile + 权重 tile; L0A 64KB/L0C 128KB; vec 双缓冲
  <=192KB UB; Wout%16!=0 时 tileHout 必须为 1; tileK 必须整除
  tileCin*k*k)。batch 维 pypto.DYNAMIC + 核内 _BATCH_TILE 循环
  (view + assemble, 参考 tests/st/test_conv.py 的
  conv2d_dynamic_batch_stride_kernel): 一次调用处理整个 batch,
  缓存 key 不含 batch, 各 batch 只编译一次 (transdata UB 随 batch
  线性增长, 单次 conv 调用 batch<=4; 尾块不足时 Python 补零裁回)。
- fwd 路线 a (im2col + pypto matmul, 免 b_trans 转置):
  小 spatial (1x1 与 Ho*Wo<=576) 形状。1x1 的 im2col 是一次
  permute+reshape 拷贝 (M*Cin); 3x3 在 24x24 以下拷贝仅 0.1-0.3ms。
  tile 按 M/K/N 分档 (M<=144 或 N<=64 用小 tile, 其余 128-256)。
- bwd (同第 10 章 PyPTOConv2d 的 bwd 用 torch 的先例):
  - dX = F.conv_transpose2d(grad_Y, K) (原生 NPU kernel, 0.06-0.31ms;
    pypto 无 transposed conv, 且自组 conv 引擎 dX 在 7x7 时 95 个
    H-tile 编译爆炸)
  - dW = im2col(X)^T @ dY_2d: 大拷贝形状 (M*K_dim*4B >= 8MB 且 M <= 40000)
    用 pypto matmul(a_trans=True) 免 cols.t().contiguous() 二次拷贝
    (实测 b2 85MB: 0.31ms vs torch 1.2ms; b5 512ch: 0.47ms vs torch 18.2ms;
    小形状 torch 0.04-0.2ms 更快; M=147456 的 b1 7x7 dW aT 编译爆炸已排除);
    其余 torch matmul (精确 FP32)
  - db = grad_Y.sum(dim=(0,2,3))

实测 (NPU, fwd+bwd 每层一步, batch=4 见 REPORT.md 对比表; batch=64 端到端):
  ResNet18 全链 step (batch=64, 96x96, 含 BN/pool/linear/Adam):
  动态 batch 引擎 98-107ms/step ≈ 600-650 ex/s, 较 batch=4 (34-41ms/步)
  吞吐 5.5x。07.06/07.07 notebook 已改为 batch=64, lr=0.001 并实测:
  07.06 loss 0.063 / acc 0.924 (~77 min 含一次性编译预热; 原 batch=4
  115 min / acc 0.917), 07.07 loss 0.121 / acc 0.930。
  精度: fwd vs 精确 FP32 参考 0~2.5e-5, dW (aT 路径) vs fp64 2.5e-3 max
  (相对 ~1e-6), dX 0; vs torch 的 ~2e-3 差值是 torch conv 自身非精确
  累积所致 (实测 torch fwd 与精确参考差 4e-4、dW 差 1.3e-2)。
  编译: conv 引擎 kernel 按 shape 首次调用时编译 17-70s+, 缓存复用;
  动态引擎循环界取 fmap.shape[0], 缓存 key 不含 batch, 各 batch 只编译
  一次 (实测 N=64 编译后 N=4/66 调用 0.1s 复用); ResNet 10 形状
  (~45-55min 一次性预热), DenseNet 更多 (~60-80min)。

限制: stride 任意, dilation=1, transposed=False, FP32 (同 v3)。
"""

import torch
import torch.nn.functional as F
from torch import nn

import pypto
from pypto import pypto_impl

_C0 = 8                    # FP32 通道对齐 (conv 引擎要求 Cout % 8 == 0)
_L1_SIZE = 192 * 1024      # L1 预算 (DAV_2201)
_BATCH_TILE = 4            # conv 引擎核内 batch 循环步长 (transdata UB 随 batch 线性增长,
                           # 单次 conv 调用 batch<=4 才不超 192KB; 整 batch 由核内循环处理)
_DYNAMIC_BATCH = True      # conv 引擎动态 batch 编译 (核内循环 + view/assemble, 一次调用处理整
                           # 个 batch; 缓存 key 不含 batch, 各 batch 只编译一次; 对应
                           # tests/st/test_conv.py 的 conv2d_dynamic_batch_stride_kernel 先例)
# dW a_trans 阈值: 仅当 cols 拷贝量 (M*K_dim*4B) >= 8MB 且归约维 M <= 40000 时用
# pypto matmul(a_trans=True) 免 cols.t().contiguous() 二次拷贝 (实测:
#   - M=36864/K_dim=576: pypto 0.31ms vs torch 1.2ms;  M=9216/K_dim=576: 0.24 vs 22
#   - M=2304/K_dim=4608: 0.47 vs 18.2;  小形状 torch 0.1-0.2ms 更快
#   - M=147456 (b1 7x7 dW) aT 编译 K=147456 展开爆炸, 硬杀进程, 必须排除)
# 编译失败会硬杀进程, 故只允许实测验证过的形状区间 (K=M 即归约维)。
_DW_ATRANS_MIN_COPY = 8 * 1024 * 1024
_DW_ATRANS_MAX_K = 40000
_CONV_KERNEL_CACHE = {}
_MM_KERNEL_CACHE = {}
_ATRANS_KERNEL_CACHE = {}


# ════════════════════════════════════════════════════════════════════
# 路线 b: 原生 pypto.conv (隐式 GEMM, 无 im2col 膨胀)
# ════════════════════════════════════════════════════════════════════


def _derive_conv_tiles(Cin, Cout, H, W, k, s, p):
    """L1/L0/vec tile 预算式推导 (约束实测于 pypto 0.2.1 / DAV_2201):
    - L1: 输入 tile (hin*win*tileCin*4B) + 权重 tile (tileN*tileCin*k*k*4B)
      <= 192KB; tileCin 从 Cin 向下减半做 K 轴切分
    - L0: L0A tileH*tileW*tileK*4 <= 64KB, L0C tileH*tileW*tileN*4 <= 128KB,
      tileN <= 16, tileW 需 16 元素对齐
    - tileK 必须整除 tileCin*k*k (kAL1/kBL1)
    - Wout 不是 16 的倍数时 tileHout 必须为 1
    - vec: batch 16 * Cout(<=64) * 1 * C0, 双缓冲 <= 192KB UB (实测
      Cout=512 时 vecN=128 与 fmap transdata 叠加超 UB)
    """
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    cin_pad = ((Cin + _C0 - 1) // _C0) * _C0
    if Wo % 16 != 0:
        tile_hout = 1
    else:
        tile_hout = None
    tile_wout = 16
    win = (tile_wout - 1) * s + k
    tileN = 16

    def fit(tc):
        return tc * k * k * tileN * 4 + win * k * tc * 4 <= _L1_SIZE - 2048
    tile_cin = cin_pad
    while not fit(tile_cin) and tile_cin > 8:
        tile_cin //= 2
    if tile_hout is None:
        w_bytes = tile_cin * k * k * tileN * 4
        in_budget = _L1_SIZE - w_bytes - 2048
        hin_max = in_budget // (win * tile_cin * 4)
        if s == 1:
            tile_hout = max(1, min(hin_max - k + 1, Ho))
        else:
            tile_hout = max(1, min((hin_max - k) // s + 1, Ho))
    hi_al1 = min((tile_hout - 1) * s + (k - 1) + 1, H)
    wi_al1 = min((tile_wout - 1) * s + (k - 1) + 1, W)
    k_fact = tile_cin * k * k
    tile_k = 64
    while k_fact % tile_k != 0 and tile_k > 8:
        tile_k //= 2
    l1 = pypto_impl.TileL1Info(tileHin=hi_al1, tileHout=tile_hout, tileWin=wi_al1,
                               tileWout=tile_wout, tileCinFmap=tile_cin,
                               tileCinWeight=tile_cin, tileN=tileN, tileBatch=1)
    l0 = pypto_impl.TileL0Info(tileH=min(tile_hout, 8), tileW=tile_wout,
                               tileK=tile_k, tileN=16)
    vec = (16, min(Cout, 64), 1, _C0)
    return l1, l0, vec, cin_pad


def _make_conv_kernel(Cin, Cout, H, W, k, s, p):
    """按 shape 生成 conv fwd kernel (闭包工厂; 编译后缓存).

    _DYNAMIC_BATCH=True: batch 维用 pypto.DYNAMIC, 核内按 _BATCH_TILE 循环
    (view + assemble), 一次调用处理整个 batch; 缓存 key 不含 batch。
    参考 tests/st/test_conv.py 的 conv2d_dynamic_batch_stride_kernel。
    _DYNAMIC_BATCH=False: 固定 batch=_BATCH_TILE 编译 (旧 v4, 同第 10 章),
    更大 batch Python 切块。
    """
    key = (Cin, Cout, H, W, k, s, p)
    if key in _CONV_KERNEL_CACHE:
        return _CONV_KERNEL_CACHE[key]
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    l1, l0, vec, cin_pad = _derive_conv_tiles(Cin, Cout, H, W, k, s, p)
    w_shape = (Cout, cin_pad, k, k)
    # 注意: pypto 0.2.1 的动态维需用 status-shape 列表 [pypto.DYNAMIC, ...] (tuple 会报
    # "Invalid value type", 见 Tensor.__init__ 的 _validate_status_shape)
    out_shape = [pypto.DYNAMIC, Cout, Ho, Wo] if _DYNAMIC_BATCH else (_BATCH_TILE, Cout, Ho, Wo)
    fmap_shape = [pypto.DYNAMIC, cin_pad, H, W] if _DYNAMIC_BATCH else (_BATCH_TILE, cin_pad, H, W)

    @pypto.frontend.jit()
    def conv_fwd_kernel(fmap: pypto.Tensor(fmap_shape, pypto.DT_FP32),
                        weight: pypto.Tensor(w_shape, pypto.DT_FP32),
                        bias: pypto.Tensor([Cout], pypto.DT_FP32),
                        out: pypto.Tensor(out_shape, pypto.DT_FP32)):
        pypto.set_conv_tile_shapes(l1, l0)
        pypto.set_vec_tile_shapes(*vec)
        if _DYNAMIC_BATCH:
            # 循环界取自 fmap 动态维 (运行时 batch), 不用 params 字典:
            # 编译缓存 key 不含 batch, 各 batch (含尾块/不同数据集) 只编译一次
            batch = fmap.shape[0]
            tile_batch = pypto.symbolic_scalar(_BATCH_TILE)
            batch_loop = (batch + tile_batch - 1) // tile_batch
            for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_batch"):
                batch_offset = batch_idx * tile_batch
                input_view = pypto.view(fmap, [tile_batch, cin_pad, H, W],
                                        [batch_offset, 0, 0, 0])
                output = pypto.conv(input_view, weight, pypto.DT_FP32, [s, s],
                                    [p, p, p, p], [1, 1],
                                    extend_params={"bias_tensor": bias}, groups=1)
                pypto.assemble(output, [batch_offset, 0, 0, 0], out)
        else:
            output = pypto.conv(fmap, weight, pypto.DT_FP32, [s, s],
                                [p, p, p, p], [1, 1],
                                extend_params={"bias_tensor": bias}, groups=1)
            out.move(output)

    _CONV_KERNEL_CACHE[key] = conv_fwd_kernel
    return conv_fwd_kernel


# ════════════════════════════════════════════════════════════════════
# 路线 a: im2col + pypto matmul (fwd)
# ════════════════════════════════════════════════════════════════════


def _tiles_for(M, K, N):
    """按 M/K/N 分档选 cube tile (对照 matmul_performance_guide 128-256 组合;
    FP32 下 L1 预算 192KB 限制 mL1*kL1+nL1*kL1 <= 48K 元素).
    实测 (tile_sweep) 小 shape 下各配置均在 ~0.15ms (启动开销主导),
    此分档在最优区间内。"""
    if M <= 144 or N <= 64:
        return [64, 64], [64, 128], [128, 128]
    return [128, 128], [64, 256], [128, 128]


def _make_matmul_kernel(M, K, N):
    """按 (M,K,N) 分档生成 b_trans matmul kernel (fwd 路线 a)."""
    key = (M, K, N)
    if key in _MM_KERNEL_CACHE:
        return _MM_KERNEL_CACHE[key]
    mt, kt, nt = _tiles_for(M, K, N)

    @pypto.frontend.jit()
    def matmul_kernel(A: pypto.Tensor([], pypto.DT_FP32),
                      B: pypto.Tensor([], pypto.DT_FP32),
                      out: pypto.Tensor([], pypto.DT_FP32)):
        """C = A @ B^T (免 B 转置拷贝)."""
        pypto.set_cube_tile_shapes(mt, kt, nt)
        out[:] = pypto.matmul(A, B, pypto.DT_FP32, b_trans=True)
    _MM_KERNEL_CACHE[key] = matmul_kernel
    return matmul_kernel


def _make_atrans_matmul_kernel(M, Kdim, Cout):
    """dW 专用 a_trans matmul: C = cols^T @ dY_2d = (Kdim, Cout).

    C = A^T @ B, A=cols (M, Kdim), B=dY_2d (M, Cout): 免 cols.t().contiguous()
    二次拷贝 (计划 B)。仅 _use_dw_atrans() 验证过的形状区间启用。
    """
    key = (M, Kdim, Cout)
    if key in _ATRANS_KERNEL_CACHE:
        return _ATRANS_KERNEL_CACHE[key]
    mt, kt, nt = _tiles_for(Kdim, M, Cout)   # 按输出 (Kdim, Cout) 分档

    @pypto.frontend.jit()
    def atrans_matmul_kernel(A: pypto.Tensor([], pypto.DT_FP32),
                             B: pypto.Tensor([], pypto.DT_FP32),
                             out: pypto.Tensor([], pypto.DT_FP32)):
        """C = A^T @ B."""
        pypto.set_cube_tile_shapes(mt, kt, nt)
        out[:] = pypto.matmul(A, B, pypto.DT_FP32, a_trans=True)
    _ATRANS_KERNEL_CACHE[key] = atrans_matmul_kernel
    return atrans_matmul_kernel


# ── 辅助 ──


def _to_pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _im2col_strided(X_padded, kH, kW, stride=(1, 1)):
    """X_padded (N,Cin,Hp,Wp) → cols (M, Cin*kH*kW), M=N*H_out*W_out.

    torch.as_strided 零拷贝窗口 view + 一次 permute/reshape/contiguous
    (torch_npu 上 F.unfold 慢 ~10-18x, 故不用)。
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


def _pad_c0(X, C):
    """通道维补零到 C0 对齐 (FP32: 8), 供 conv 引擎使用."""
    pad = ((C + _C0 - 1) // _C0) * _C0 - C
    if pad == 0:
        return X
    return F.pad(X, (0, 0, 0, 0, 0, pad, 0, 0))


def _conv_geom(X_shape, kH, kW, stride, padding):
    """输出尺寸 (H_out, W_out)."""
    s_h, s_w = _to_pair(stride)
    p_h, p_w = _to_pair(padding)
    H, W = X_shape[2], X_shape[3]
    return (H + 2 * p_h - kH) // s_h + 1, (W + 2 * p_w - kW) // s_w + 1


def _use_conv_engine(k, Ho, Wo, Cout):
    """fwd 路线选择 (实测分档, 见模块 docstring):
    conv 引擎赢大 spatial (48x48+ 与 24x24 少输出通道);
    小 spatial / 1x1 走 im2col+matmul (启动开销 ~0.15ms 更优)."""
    if k == 1:
        # 1x1 实测 im2col+matmul 更快 (96->96 @224 batch64 fwd+bwd 38ms vs conv 引擎 72ms,
        # 且 1x1 引擎编译 ~73min vs matmul 路线 ~20min), 不走引擎
        return False
    if Ho * Wo >= 2304:                     # 7x7 conv1 的 48x48 输出与更大
        return True
    if Ho * Wo == 576 and Cout <= 128:      # 24x24, 输出通道 <= 128 (ResNet b2, DenseNet 块)
        return True
    return False


# ── forward ──


def _fwd_conv(X, K, bias, stride, padding):
    """conv 引擎 fwd: 通道补零对齐 + 一次 kernel 调用处理整个 batch.

    _DYNAMIC_BATCH=True: batch 维 DYNAMIC, 核内循环, 尾块不足 _BATCH_TILE
    时 batch 维补零后整批一次调用 (输出多出 tail 行, 裁回即可);
    _DYNAMIC_BATCH=False: 固定 batch=_BATCH_TILE 编译, 大 batch Python 切块。
    """
    N, Cin, H, W = X.shape
    Cout, _, kH, kW = K.shape
    s_h, s_w = _to_pair(stride)
    p_h, p_w = _to_pair(padding)
    H_out, W_out = _conv_geom(X.shape, kH, kW, stride, padding)
    cout_pad = ((Cout + _C0 - 1) // _C0) * _C0
    kernel = _make_conv_kernel(Cin, cout_pad, H, W, kH, s_h, p_h)
    Xp = _pad_c0(X, Cin)
    Kp = _pad_c0(K, Cin)                                    # dim1 输入通道补零
    if cout_pad > Cout:
        Kp = F.pad(Kp, (0, 0, 0, 0, 0, 0, 0, cout_pad - Cout))  # dim0 输出通道补零
    b = bias
    if b is None:
        b = torch.zeros(cout_pad, device=X.device, dtype=X.dtype)
    elif cout_pad > Cout:
        b = F.pad(b, (0, cout_pad - Cout))   # kernel 输出通道按 C0 对齐, bias 同步补零
    if _DYNAMIC_BATCH:
        tail = (-N) % _BATCH_TILE
        if tail:
            Xp = F.pad(Xp, (0, 0, 0, 0, 0, 0, 0, tail))     # N 维补零到 tile 倍数
        out = torch.empty(Xp.shape[0], cout_pad, H_out, W_out,
                          device=X.device, dtype=X.dtype)
        kernel(Xp.contiguous(), Kp.contiguous(), b, out)
        out = out[:N]
    else:
        outs = []
        for start in range(0, N, _BATCH_TILE):
            xc = Xp[start:start + _BATCH_TILE].contiguous()
            tail = _BATCH_TILE - xc.shape[0]
            if tail > 0:
                xc = F.pad(xc, (0, 0, 0, 0, 0, 0, 0, tail))
            chunk = torch.empty(xc.shape[0], cout_pad, H_out, W_out,
                                device=X.device, dtype=X.dtype)
            kernel(xc, Kp.contiguous(), b, chunk)
            outs.append(chunk[:xc.shape[0] - tail] if tail else chunk)
        out = torch.cat(outs, dim=0)
    if cout_pad > Cout:
        out = out[:, :Cout]
    return out


def _fwd_matmul(X, K, bias, stride, padding):
    """im2col + pypto matmul fwd: C = cols @ K_2d^T (K_2d 零拷贝 view)."""
    N, Cin, H, W = X.shape
    Cout, _, kH, kW = K.shape
    s_h, s_w = _to_pair(stride)
    p_h, p_w = _to_pair(padding)
    H_out, W_out = _conv_geom(X.shape, kH, kW, stride, padding)

    X_pad = _pad4d(X, p_h, p_w)
    cols = _im2col_strided(X_pad, kH, kW, (s_h, s_w))       # (M, K_dim)
    K_2d = K.reshape(Cout, Cin * kH * kW)                   # (Cout, K_dim), 零拷贝
    out = torch.zeros(cols.shape[0], Cout, device=X.device, dtype=X.dtype)
    _make_matmul_kernel(cols.shape[0], cols.shape[1], Cout)(cols, K_2d, out)
    if bias is not None:
        out = out + bias.view(1, Cout)      # bias 不融合: 多核切 K 时 extend_params 结果错误
    return out.reshape(N, H_out, W_out, Cout).permute(0, 3, 1, 2).contiguous()


def _conv_forward(X, K, bias, stride, padding):
    """fwd 分派: 大 spatial → conv 引擎; 小 spatial / 1x1 → im2col+matmul."""
    N, Cin, H, W = X.shape
    Cout, _, kH, kW = K.shape
    H_out, W_out = _conv_geom(X.shape, kH, kW, stride, padding)
    if _use_conv_engine(kH, H_out, W_out, Cout):
        return _fwd_conv(X, K, bias, stride, padding)
    return _fwd_matmul(X, K, bias, stride, padding)


# ── backward ──
# dX = F.conv_transpose2d (原生 NPU kernel, 精确); dW = im2col + torch matmul
# (精确 FP32)。同第 10 章 PyPTOConv2d 的 bwd 用 torch 的先例。


def _use_dw_atrans(M, Kdim, Cout):
    """dW 是否走 pypto a_trans (免 cols.t() 二次拷贝): 实测分档, 见模块头
    _DW_ATRANS_MIN_COPY / _DW_ATRANS_MAX_K 注释 (小拷贝 torch 更快;
    M=147456 的 b1 7x7 dW 会编译爆炸, 已排除)."""
    return (M * Kdim * 4 >= _DW_ATRANS_MIN_COPY
            and M <= _DW_ATRANS_MAX_K)


def _conv_dW(X, grad_Y, stride, padding, kH, kW):
    """dW = im2col(X_pad)^T @ dY_2d.

    大拷贝形状用 pypto matmul(a_trans=True) (免 cols.t().contiguous() 二次
    拷贝, 实测 0.24-0.47ms vs torch 1.2-22ms); 其余用 torch matmul (精确 FP32,
    0.04-0.2ms)。"""
    N, Cin, H, W = X.shape
    Cout = grad_Y.shape[1]
    s_h, s_w = _to_pair(stride)
    p_h, p_w = _to_pair(padding)
    H_out, W_out = _conv_geom(X.shape, kH, kW, stride, padding)

    X_pad = _pad4d(X, p_h, p_w)
    cols = _im2col_strided(X_pad, kH, kW, (s_h, s_w))       # (M, K_dim)
    dY_2d = grad_Y.permute(0, 2, 3, 1).reshape(N * H_out * W_out, Cout).contiguous()
    M, K_dim = cols.shape[0], cols.shape[1]
    if _use_dw_atrans(M, K_dim, Cout):
        dK = torch.empty(K_dim, Cout, device=X.device, dtype=X.dtype)
        _make_atrans_matmul_kernel(M, K_dim, Cout)(cols, dY_2d, dK)
    else:
        dK = torch.matmul(cols.t().contiguous(), dY_2d)     # (K_dim, Cout)
    return dK.reshape(Cin, kH, kW, Cout).permute(3, 0, 1, 2).contiguous()


def _conv_dX(X_shape, grad_Y, K_4d, stride, padding):
    """dX = conv_transpose2d(grad_Y, K) (原生 kernel, 精确 FP32).

    当 (H+2p-k)%s != 0 时 (如 7x7 s2 p3: 95%2=1), convT 输出尺寸
    (gH-1)*s-2p+k = H-r 会少 r 行: 补零 gH+1 行后 convT 尺寸 >= H,
    裁回即精确伴随 (补的行贡献为零, 实测 7x7 s2 p3 误差 0)。
    """
    N, Cin, H, W = X_shape
    kH, kW = K_4d.shape[2], K_4d.shape[3]
    p_h, p_w = _to_pair(padding)
    s_h, s_w = _to_pair(stride)
    gY = grad_Y.contiguous()
    if (H + 2 * p_h - kH) % s_h != 0:
        gY = F.pad(gY, (0, 0, 0, 1))
    if (W + 2 * p_w - kW) % s_w != 0:
        gY = F.pad(gY, (0, 1))
    dX = F.conv_transpose2d(gY, K_4d, stride=(s_h, s_w), padding=(p_h, p_w))
    return dX[:, :, :H, :W]


# ── autograd Function + nn.Module ──


class PyPTOConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, K, stride, padding, dilation, bias):
        s_h, s_w = _to_pair(stride)
        if _to_pair(dilation) != (1, 1):
            raise NotImplementedError("PyPTOConv2d 不支持 dilation")
        X_c, K_c = X.contiguous(), K.contiguous()
        Y = _conv_forward(X_c, K_c, bias, stride, padding)
        ctx.save_for_backward(X_c, K_c)
        ctx.params = (s_h, s_w, padding, K_c.shape[2], K_c.shape[3])
        ctx.has_bias = bias is not None
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, K = ctx.saved_tensors
        s_h, s_w, padding, kH, kW = ctx.params
        grad_Y = grad_output.contiguous()

        dK = _conv_dW(X, grad_Y, (s_h, s_w), padding, kH, kW)
        dX = _conv_dX(X.shape, grad_Y, K, (s_h, s_w), padding)
        db = grad_Y.sum(dim=(0, 2, 3)).contiguous() if ctx.has_bias else None
        return dX, dK, None, None, None, db


class PyPTOConv2d(nn.Module):
    """PyPTO Conv2d v4 (混合路线): fwd 用 pypto.conv 引擎或 im2col+matmul,
    bwd 用 torch convT + matmul (同第 10 章先例)。接口与 v3 兼容。

    stride 任意, dilation=1, transposed=False (接口参数保留, 不支持时抛错)。
    详见模块 docstring 的实测数据与限制。
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

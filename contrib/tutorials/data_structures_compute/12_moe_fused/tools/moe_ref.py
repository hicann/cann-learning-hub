"""MoE Router 路径的 PyTorch 参考实现（ground truth）与完整 MoE 层数值示例。

Router 路径（本实验算子 moe_router_fused 的参考）:
    scores      = x @ W_gate                 # [N, D] x [D, E] -> [N, E]
    gate_scores = softmax(scores, dim=-1)    # 行内归一
    topk_scores, topk_idx = topk(gate_scores, K, dim=-1, sorted=True)
    topk_weights = topk_scores / sum(topk_scores, dim=-1)

Tie-break 规则（记录在案，M1 验收要求）:
    - 输入为连续随机分布（randn 量化到 FP16），FP32 参考计算下平局概率约为 0；
    - torch.topk(sorted=True) 按值降序输出，本参考与算子均按"值降序"对齐；
    - 若发生平局，torch.topk 对相同值的返回顺序由实现决定，测试数据设计上规避此情况。

完整 MoE 层参考（仅用于报告理论分析的数值示例，不参与算子测试）:
    每个专家为标准 FFN: out = (silu(x @ W1_e)) @ W2_e
    y[n] = sum_k topk_weights[n, k] * expert_{topk_idx[n,k]}(x[n])

用法:
    python tools/moe_ref.py --cpu            # CPU 参考 + 自洽性检查
    python tools/moe_ref.py --npu            # NPU 参考 + CPU/NPU 一致性检查
    python tools/moe_ref.py --npu --bench    # 额外输出 4 算子序列 baseline 耗时
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Router 路径参考
# ---------------------------------------------------------------------------
def moe_router(x, W_gate, K, compute_dtype=torch.float32):
    """MoE Router: matmul -> softmax -> topk -> renorm.

    Args:
        x: [N, D] 输入 token（FP16 量化后的值，以 float32 张量承载）
        W_gate: [D, E] 门控权重
        K: top-K 的 K
        compute_dtype: 计算精度（参考实现统一 FP32）

    Returns:
        topk_idx:    [N, K] int32
        topk_weights:[N, K] float32（已归一，行内和为 1）
        gate_scores: [N, E] float32（softmax 输出，调试用）
    """
    x = x.to(compute_dtype)
    W_gate = W_gate.to(compute_dtype)
    scores = torch.matmul(x, W_gate)                       # [N, E]
    gate_scores = F.softmax(scores, dim=-1)                # [N, E]
    topk_scores, topk_idx = torch.topk(gate_scores, K, dim=-1, sorted=True)
    topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    return topk_idx.to(torch.int32), topk_weights, gate_scores


def moe_layer_ref(x, W_gate, experts, K):
    """完整 MoE 层参考（专家 FFN + 加权求和），仅用于报告数值示例。

    Args:
        x: [N, D]
        W_gate: [D, E]
        experts: (W1, W2)，W1: [E, D, F], W2: [E, F, D]
        K: top-K
    Returns:
        y: [N, D]
    """
    topk_idx, topk_weights, _ = moe_router(x, W_gate, K)
    W1, W2 = experts
    E_exp, D_in, F_hid = W1.shape
    N = x.shape[0]
    y = torch.zeros_like(x, dtype=torch.float32)
    for e in range(E_exp):
        sel = (topk_idx == e)                      # [N, K] bool
        tok_ids, k_ids = torch.nonzero(sel, as_tuple=True)
        if tok_ids.numel() == 0:
            continue
        xe = x[tok_ids].to(torch.float32)          # [m, D]
        he = F.silu(xe @ W1[e].to(torch.float32))  # [m, F]
        oe = he @ W2[e].to(torch.float32)          # [m, D]
        w = topk_weights[tok_ids, k_ids].to(torch.float32).unsqueeze(1)
        y.index_add_(0, tok_ids, w * oe)
    return y


# ---------------------------------------------------------------------------
# 数据生成（与 gen_test_data.py 共享同一逻辑，保证口径一致）
# ---------------------------------------------------------------------------
def make_inputs(N, D, E, K, seed=42):
    """生成 FP16 量化的 (x, W_gate)，返回 float32 张量（承载 FP16 值）。

    选用 FP16 而非 BF16 的原因（设计简化，与 08 章 attention_custom 一致）：
    910B + CANN 9.0.0 的向量 Cast 指令不支持 BF16<->FP32 转换（编译报
    "not support bf16 type cast"），而 FP16<->FP32 是已验证可用的路径；
    且 FP16 尾数 10 位高于 BF16 的 8 位，参考比对余量更大。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=g)
    W_gate = torch.randn(D, E, generator=g) / (D ** 0.5)
    # 量化到 FP16：算子实际消费的是 FP16 值，参考必须用同一份量化后的值
    x = x.half().float()
    W_gate = W_gate.half().float()
    return x, W_gate


def check_cpu_self_consistency(N=256, D=512, E=16, K=2):
    """CPU 参考自洽性：topk 索引可复现、权重行内和为 1、与 NumPy 参考一致。"""
    x, W_gate = make_inputs(N, D, E, K)
    idx, w, gate = moe_router(x, W_gate, K)
    assert w.sum(dim=-1).abs().sub(1).max().item() < 1e-5, "权重行内和不为 1"

    # NumPy 参考（独立实现）
    scores = x.numpy() @ W_gate.numpy()
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    gate_np = e / e.sum(axis=1, keepdims=True)
    # 值降序，平局按小索引优先（与"无平局"假设等价）
    order = np.argsort(-gate_np, axis=1, kind="stable")
    idx_np = order[:, :K].astype(np.int32)
    w_np = gate_np[np.arange(N)[:, None], idx_np]
    w_np = (w_np / w_np.sum(axis=1, keepdims=True)).astype(np.float32)

    idx_match = (idx.numpy() == idx_np).mean()
    w_close = np.allclose(w.numpy(), w_np, rtol=1e-3, atol=1e-3)
    print(f"[CPU self-check] N={N} D={D} E={E} K={K}")
    print(f"  topk_idx 与 NumPy 参考一致率: {idx_match:.4%}")
    print(f"  topk_weights allclose(rtol=1e-3, atol=1e-3): {w_close}")
    assert idx_match == 1.0 and w_close, "CPU 自洽性检查失败"
    return True


def check_cpu_npu(N=256, D=512, E=16, K=2, device_id=0):
    """CPU 与 NPU 参考实现一致性（torch_npu 4 算子序列 vs CPU 参考）。

    精度说明：NPU 侧 router 参考统一用 FP32 计算（输入为 FP16 量化值，softmax
    近平局可能导致索引顺序差异，属预期误差而非实现错误）。
    索引比对采用 tie-aware 规则：索引不同但两侧对应 softmax 分值差 < 1e-4 的
    记为"平局差异"（近平局下 torch.topk 顺序不保证），单独统计并报告。
    """
    import torch_npu  # noqa: F401

    x, W_gate = make_inputs(N, D, E, K)
    idx_cpu, w_cpu, gate_cpu = moe_router(x, W_gate, K)

    x_npu = x.float().npu(device_id)
    W_npu = W_gate.float().npu(device_id)
    scores = torch.matmul(x_npu, W_npu)
    gate = F.softmax(scores, dim=-1)
    topk_v, topk_i = torch.topk(gate, K, dim=-1, sorted=True)
    topk_w = topk_v / topk_v.sum(dim=-1, keepdim=True)
    torch.npu.synchronize(device_id)

    idx_npu = topk_i.cpu().to(torch.int32)
    w_npu = topk_w.float().cpu()
    gate_npu = gate.cpu()

    diff_mask = (idx_cpu != idx_npu)
    n_diff = int(diff_mask.sum().item())
    # tie-aware：索引不同但两侧所选分值几乎相同（近平局）
    tie_diff = 0
    for n, k in zip(*torch.nonzero(diff_mask, as_tuple=True)):
        v_cpu = gate_cpu[n, idx_cpu[n, k]].item()
        v_npu = gate_npu[n, idx_npu[n, k]].item()
        if abs(v_cpu - v_npu) < 1e-4:
            tie_diff += 1
    n_real_diff = n_diff - tie_diff

    idx_match = 1.0 - n_diff / (N * K)
    w_ok = torch.allclose(w_cpu, w_npu, rtol=1e-3, atol=1e-3)
    print(f"[CPU vs NPU] N={N} D={D} E={E} K={K} (NPU 侧 FP32 参考)")
    print(f"  topk_idx 一致率: {idx_match:.4%}  (不一致 {n_diff} 个, "
          f"其中近平局差异 {tie_diff} 个, 真实差异 {n_real_diff} 个)")
    print(f"  topk_weights allclose(rtol=1e-3, atol=1e-3): {w_ok}")
    assert n_real_diff == 0 and w_ok, "CPU/NPU 参考存在真实不一致"
    return True


def bench_baseline(N, D, E, K, device_id=0, warmup=50, repeat=200):
    """未融合 4 算子序列（torch_npu）baseline 耗时，M5 使用。

    序列: matmul -> softmax -> topk -> div，输入为 FP16（与融合算子接口一致）。
    """
    import torch_npu  # noqa: F401

    x, W_gate = make_inputs(N, D, E, K)
    x_npu = x.half().npu(device_id)
    W_npu = W_gate.half().npu(device_id)

    def four_ops():
        scores = torch.matmul(x_npu, W_npu)
        gate = F.softmax(scores, dim=-1)
        topk_v, _ = torch.topk(gate, K, dim=-1, sorted=True)
        return topk_v / topk_v.sum(dim=-1, keepdim=True)

    for _ in range(warmup):
        four_ops()
    torch.npu.synchronize(device_id)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        four_ops()
        torch.npu.synchronize(device_id)
        times.append((time.perf_counter() - t0) * 1e3)  # ms

    times = np.array(times)
    print(f"[baseline 4-op] N={N} D={D} E={E} K={K}: "
          f"mean={times.mean():.4f} ms  median={np.median(times):.4f} ms  "
          f"p99={np.percentile(times, 99):.4f} ms")
    return times


def demo_full_moe():
    """完整 MoE 层数值示例（报告第 2 章引用）。"""
    N, D, E_exp, F_hid, K = 8, 64, 8, 128, 2
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, D, generator=g).half().float()
    W_gate = torch.randn(D, E_exp, generator=g).half().float()
    W1 = torch.randn(E_exp, D, F_hid, generator=g).half().float()
    W2 = torch.randn(E_exp, F_hid, D, generator=g).half().float()

    y = moe_layer_ref(x, W_gate, (W1, W2), K)

    # 朴素逐 token 计算对照
    idx, w, _ = moe_router(x, W_gate, K)
    y2 = torch.zeros_like(y)
    for n in range(N):
        for k in range(K):
            e = idx[n, k].item()
            h = F.silu(x[n] @ W1[e])
            y2[n] += w[n, k] * (h @ W2[e])
    ok = torch.allclose(y, y2, rtol=1e-4, atol=1e-4)
    print(f"[full MoE layer demo] 向量化参考 vs 逐 token 参考一致: {ok}")
    print(f"  y[:2, :4] =\n{y[:2, :4]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--npu", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if not (args.cpu or args.npu):
        args.cpu = True

    ok = True
    if args.cpu:
        ok &= check_cpu_self_consistency()
        ok &= demo_full_moe()
    if args.npu:
        if not torch_npu_available():
            print("torch_npu 不可用，跳过 NPU 检查")
            return 1
        ok &= check_cpu_npu(device_id=args.device)
    if args.bench:
        if not args.npu:
            print("--bench 需要 --npu")
            return 1
        for (N, D, E, K) in [(128, 512, 16, 2), (1024, 1024, 32, 4), (4096, 2048, 8, 2)]:
            bench_baseline(N, D, E, K, device_id=args.device)
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


def torch_npu_available():
    try:
        import torch_npu  # noqa: F401
        return torch_npu.npu.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
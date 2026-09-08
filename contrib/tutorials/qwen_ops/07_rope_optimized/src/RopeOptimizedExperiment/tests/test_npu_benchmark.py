#!/usr/bin/env python3
"""
NPU-Resident RoPE 性能对比测试

对比两条路径在 NPU 上的 E2E 前向耗时：
  - 原生 torch_npu RoPE
  - 自定义 NPU-resident RoPE（零 H2D/D2H）

多轮 warmup + bench，输出 mean/median/std。

注意：
  - 首次调用包含 GE JIT 编译，warmup 会吸收这部分开销
  - 两次 model load 后 GE kernel 已缓存，对比公平

用法:
  cd ~/Projects/QwenRoPeCustomOpt
  bash scripts/run_test.sh tests/test_npu_benchmark.py --seq-lens "8,32,64"
"""

import os, sys, time, statistics, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _setup_env import auto_setup; auto_setup()

import torch
import torch_npu

from torch_extension import load_torch_ops


def load_model_to_npu(model_path):
    """延迟 import + 模型加载到 NPU。"""
    from transformers import AutoModelForCausalLM
    mp = os.path.expanduser(model_path)
    return AutoModelForCausalLM.from_pretrained(
        mp, torch_dtype=torch.float32,
        trust_remote_code=True, local_files_only=True,
    ).eval().to("npu")


def benchmark_native(model, input_ids, warmup, bench):
    """原生 torch_npu RoPE 前向 benchmark。"""
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids).logits
        torch_npu.npu.synchronize()

    times = []
    for _ in range(bench):
        gc.collect()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids).logits
        torch_npu.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def benchmark_custom(model, input_ids, warmup, bench):
    """自定义 NPU-resident RoPE 前向 benchmark。"""
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids).logits
        torch_npu.npu.synchronize()

    times = []
    for _ in range(bench):
        gc.collect()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids).logits
        torch_npu.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    import argparse
    ap = argparse.ArgumentParser(description="NPU-Resident RoPE benchmark")
    ap.add_argument("--model", default="~/Models/Qwen2.5-0.5B")
    ap.add_argument("--seq-lens", default="8,32,64",
                    help="逗号分隔的序列长度 (default: 8,32,64)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="warmup 轮数 (default: 5)")
    ap.add_argument("--bench", type=int, default=10,
                    help="benchmark 轮数 (default: 10)")
    ap.add_argument("--skip-custom", action="store_true",
                    help="只测原生，跳过自定义算子")
    args = ap.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 65)
    print("  NPU-Resident RoPE — E2E Forward Performance Benchmark")
    print(f"  warmup={args.warmup}  bench={args.bench}  seq_lens={seq_lens}")
    print("=" * 65)

    load_torch_ops()

    # ── 加载 tokenizer（只需一次） ──
    from transformers import AutoTokenizer
    mp = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, local_files_only=True)

    # ── 注册自定义 RoPE（只需一次） ──
    rope_qk = torch.ops.qwen_rope_custom_opt.rope_qk_compact
    rope_qk(
        torch.randn(56, 64, device="npu"),
        torch.randn( 8, 64, device="npu"),
        torch.randn( 4, 64, device="npu"),
        torch.randn( 4, 64, device="npu"),
        4, 14, 2,
    )

    def custom_rope(q, k, cos, sin, unsqueeze_dim=1):
        hd = cos.shape[-1]; qh = q.shape[1]; kh = k.shape[1]; s = q.shape[2]
        cf = cos.reshape(-1, hd).contiguous(); sf = sin.reshape(-1, hd).contiguous()
        qf = q.reshape(-1, hd).contiguous(); kf = k.reshape(-1, hd).contiguous()
        qr, kr = rope_qk(qf, kf, cf, sf, s, qh, kh)
        return qr.view_as(q), kr.view_as(k)

    import transformers.models.qwen2.modeling_qwen2 as mq

    results = []

    for seq_len in seq_lens:
        prompt = "The capital " + " of France" * ((seq_len - 2) // 2 + 1)
        input_ids = tokenizer(prompt.strip(), return_tensors="pt")["input_ids"][:, :seq_len]
        actual_seq = input_ids.shape[1]
        ids_npu = input_ids.to("npu")

        print(f"\n{'─' * 65}")
        print(f"  seq_len={actual_seq}  input_shape={list(input_ids.shape)}")
        print(f"  prompt='{prompt.strip()[:50]}...'" if len(prompt) > 50 else f"  prompt='{prompt.strip()}'")

        # ── 原生 ──
        print(f"  [原生 NPU] warmup x{args.warmup} + bench x{args.bench} ...")
        m_native = load_model_to_npu(args.model)
        t_native = benchmark_native(m_native, ids_npu, args.warmup, args.bench)
        del m_native; gc.collect(); torch_npu.npu.empty_cache()

        nm = statistics.mean(t_native)
        ns = statistics.stdev(t_native)
        print(f"    mean={nm:.1f}ms  median={statistics.median(t_native):.1f}ms  "
              f"min={min(t_native):.1f}ms  max={max(t_native):.1f}ms  std={ns:.1f}ms")

        custom_times = None
        if not args.skip_custom:
            # ── 自定义 NPU-resident ──
            print(f"  [自定义 NPU-resident] warmup x{args.warmup} + bench x{args.bench} ...")
            mq.apply_rotary_pos_emb = custom_rope
            m_custom = load_model_to_npu(args.model)
            t_custom = benchmark_custom(m_custom, ids_npu, args.warmup, args.bench)
            del m_custom; gc.collect(); torch_npu.npu.empty_cache()

            cm = statistics.mean(t_custom)
            cs = statistics.stdev(t_custom)
            su = nm / cm if cm > 0 else float("inf")
            print(f"    mean={cm:.1f}ms  median={statistics.median(t_custom):.1f}ms  "
                  f"min={min(t_custom):.1f}ms  max={max(t_custom):.1f}ms  std={cs:.1f}ms")
            print(f"    \u0394 = {cm - nm:+.1f}ms  speedup = {su:.2f}x")
            custom_times = (cm, cs, su)

        results.append((actual_seq, nm, ns, custom_times))

    # ── 汇总表 ──
    print(f"\n{'=' * 65}")
    print(f"  {'seq':>6}  {'native_ms':>10}  {'custom_ms':>10}  {'delta_ms':>10}  {'speedup':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")
    for sl, nm, ns, ct in results:
        if ct:
            cm, cs, su = ct
            print(f"  {sl:>6}  {nm:>10.1f}  {cm:>10.1f}  {cm - nm:>+10.1f}  {su:>8.2f}x")
        else:
            print(f"  {sl:>6}  {nm:>10.1f}  {'(skip)':>10}  {'':>10}  {'':>8}")
    print(f"{'═' * 65}\n")
    print("  说明: NPU-resident 路径仅在 model.to('npu') 时生效。")
    print("  Q/K/cos/sin 均在 NPU 上 → AutogradPrivateUse1 dispatch")
    print("  → zero H2D/D2H → 消除 CPU↔NPU 数据搬运开销\n")


if __name__ == "__main__":
    main()

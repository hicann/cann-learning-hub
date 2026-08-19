#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


def reference(q, k, v, causal):
    groups = q.size(1) // k.size(1)
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    score = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
    if causal:
        q_pos = torch.arange(q.size(-2)).view(-1, 1)
        k_pos = torch.arange(k.size(-2)).view(1, -1)
        score = score.masked_fill(k_pos > q_pos + k.size(-2) - q.size(-2), float("-inf"))
    return torch.softmax(score, dim=-1) @ v


def run_case(batch, q_heads, kv_heads, q_len, kv_len, head_dim, causal):
    torch.manual_seed(42)
    q = torch.randn(batch, q_heads, q_len, head_dim)
    k = torch.randn(batch, kv_heads, kv_len, head_dim)
    v = torch.randn_like(k)
    golden = reference(q, k, v, causal)
    actual = torch.ops.gqa_attention_optimized_custom.gqa_attention(q, k, v, 0.0, causal)
    diff = (actual - golden).abs()
    print(f"B={batch} Hq={q_heads} Hkv={kv_heads} Sq={q_len} Sk={kv_len} D={head_dim} causal={causal} max_abs={diff.max().item():.8e} mean_abs={diff.mean().item():.8e}")
    return torch.allclose(actual, golden, atol=3e-3, rtol=3e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--q-len", type=int, default=32)
    parser.add_argument("--kv-len", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()
    load_torch_ops()
    cases = [(args.q_len, args.kv_len, True), (args.q_len, args.kv_len, False), (1, args.kv_len, True)]
    ok = all(run_case(args.batch, args.q_heads, args.kv_heads, q_len, kv_len, args.head_dim, causal) for q_len, kv_len, causal in cases)
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

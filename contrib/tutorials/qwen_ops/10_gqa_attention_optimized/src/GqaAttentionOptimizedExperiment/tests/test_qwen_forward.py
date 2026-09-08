#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_torch_op import reference
from torch_extension import load_torch_ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()
    torch.manual_seed(42)
    q = torch.randn(args.batch, args.q_heads, args.seq, args.head_dim)
    k = torch.randn(args.batch, args.kv_heads, args.seq, args.head_dim)
    v = torch.randn_like(k)
    load_torch_ops()
    native = reference(q, k, v, True)
    custom = torch.ops.gqa_attention_optimized_custom.gqa_attention(q, k, v, 0.0, True)
    diff = (custom - native).abs()
    print(f"max_abs_diff={diff.max().item():.8e}")
    print(f"mean_abs_diff={diff.mean().item():.8e}")
    ok = torch.allclose(custom, native, atol=3e-3, rtol=3e-3)
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

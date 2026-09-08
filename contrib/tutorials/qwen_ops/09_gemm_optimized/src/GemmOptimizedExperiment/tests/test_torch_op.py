#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


def run_case(m: int, k: int, n: int, atol: float, rtol: float) -> bool:
    torch.manual_seed(42)
    device = "npu"

    a = torch.randn(m, k, dtype=torch.float32, device=device) * 0.1
    b = torch.randn(k, n, dtype=torch.float32, device=device) * 0.1

    golden = torch.matmul(a, b)
    actual = torch.ops.gemm_custom.gemm(a.contiguous(), b.contiguous())
    torch.npu.synchronize()

    diff = (actual - golden).abs()
    rel = diff / (golden.abs() + 1e-12)

    print(
        f"M={m} K={k} N={n} "
        f"max_abs={diff.max().item():.8e} "
        f"mean_abs={diff.mean().item():.8e} "
        f"max_rel={rel.max().item():.8e}"
    )

    ok = torch.allclose(actual, golden, atol=atol, rtol=rtol)
    if not ok:
        print("actual =", actual.cpu().reshape(-1)[:8])
        print("golden =", golden.cpu().reshape(-1)[:8])

    print("PASS" if ok else "FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    load_torch_ops()
    ok = run_case(args.m, args.k, args.n, args.atol, args.rtol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
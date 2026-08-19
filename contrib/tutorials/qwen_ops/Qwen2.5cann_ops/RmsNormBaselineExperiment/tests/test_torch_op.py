#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


def reference(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
    return input_tensor * torch.rsqrt(variance + eps) * weight


def run_case(shape, eps: float, atol: float, rtol: float) -> bool:
    torch.manual_seed(42)
    input_tensor = torch.randn(*shape, dtype=torch.float32)
    weight = torch.empty(shape[-1], dtype=torch.float32).uniform_(0.8, 1.2)
    golden = reference(input_tensor, weight, eps)
    actual = torch.ops.rmsnorm_custom.rms_norm(input_tensor, weight, eps)
    diff = (actual - golden).abs()
    rel = diff / (golden.abs() + 1e-12)
    print(f"shape={shape} max_abs={diff.max().item():.8e} mean_abs={diff.mean().item():.8e} max_rel={rel.max().item():.8e}")
    ok = torch.allclose(actual, golden, atol=atol, rtol=rtol)
    print("PASS" if ok else "FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    args = parser.parse_args()

    load_torch_ops()
    shapes = [(args.hidden,), (args.rows, args.hidden), (2, 4, args.hidden)]
    ok = all(run_case(shape, args.eps, args.atol, args.rtol) for shape in shapes)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

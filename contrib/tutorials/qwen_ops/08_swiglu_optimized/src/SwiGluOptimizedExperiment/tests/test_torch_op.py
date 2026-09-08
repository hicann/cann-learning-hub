#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


def reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate * torch.sigmoid(gate) * up


def run_case(shape, atol: float, rtol: float) -> bool:
    torch.manual_seed(42)
    device = "npu"
    gate = torch.randn(*shape, dtype=torch.float32, device=device)
    up = torch.randn(*shape, dtype=torch.float32, device=device)

    golden = reference(gate, up)
    actual = torch.ops.swiglu_custom.swiglu(gate, up)
    torch.npu.synchronize()

    diff = (actual - golden).abs()
    rel = diff / (golden.abs() + 1e-12)

    print(
        f"shape={shape} "
        f"max_abs={diff.max().item():.8e} "
        f"mean_abs={diff.mean().item():.8e} "
        f"max_rel={rel.max().item():.8e}"
    )

    ok = torch.allclose(actual, golden, atol=atol, rtol=rtol)
    if not ok:
        print("gate  =", gate.cpu().reshape(-1)[:8])
        print("up    =", up.cpu().reshape(-1)[:8])
        print("actual=", actual.cpu().reshape(-1)[:8])
        print("golden=", golden.cpu().reshape(-1)[:8])

    print("PASS" if ok else "FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    load_torch_ops()
    shapes = [(1, args.hidden), (args.rows, args.hidden), (2, 4, args.hidden)]
    ok = all(run_case(shape, args.atol, args.rtol) for shape in shapes)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
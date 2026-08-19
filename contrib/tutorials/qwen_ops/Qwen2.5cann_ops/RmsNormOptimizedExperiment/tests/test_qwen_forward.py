#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


class MiniQwenRmsNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
        torch.nn.init.uniform_(self.weight, 0.8, 1.2)
        self.variance_epsilon = eps

    def forward_native(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.variance_epsilon) * self.weight

    def forward_custom(self, x):
        return torch.ops.rmsnorm_custom.rms_norm(x.contiguous(), self.weight.contiguous(), self.variance_epsilon)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=896)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    load_torch_ops()
    torch.manual_seed(42)
    module = MiniQwenRmsNorm(args.hidden, args.eps).eval()
    x = torch.randn(args.batch, args.seq, args.hidden, dtype=torch.float32)

    with torch.no_grad():
        native = module.forward_native(x)
        custom = module.forward_custom(x)

    diff = (custom - native).abs()
    print(f"max_abs_diff={diff.max().item():.8e}")
    print(f"mean_abs_diff={diff.mean().item():.8e}")
    ok = torch.allclose(custom, native, atol=2e-3, rtol=2e-3)
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

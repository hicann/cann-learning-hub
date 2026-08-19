#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


class MiniLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_kn = torch.nn.Parameter(torch.randn(in_features, out_features) * 0.1)

    def forward_native(self, x):
        return torch.matmul(x, self.weight_kn)

    def forward_custom(self, x):
        original_shape = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        y2d = torch.ops.gemm_custom.gemm(x2d, self.weight_kn.contiguous())
        return y2d.reshape(*original_shape, self.weight_kn.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=896)
    parser.add_argument("--out", type=int, default=4864)
    args = parser.parse_args()

    load_torch_ops()
    torch.manual_seed(42)
    device = "npu"
    layer = MiniLinear(args.hidden, args.out).eval().to(device)
    x = torch.randn(args.batch, args.seq, args.hidden, dtype=torch.float32, device=device) * 0.1

    with torch.no_grad():
        native = layer.forward_native(x)
        custom = layer.forward_custom(x)
        torch.npu.synchronize()

    diff = (custom - native).abs()
    print(f"max_abs_diff={diff.max().item():.8e}")
    print(f"mean_abs_diff={diff.mean().item():.8e}")
    ok = torch.allclose(custom, native, atol=1e-3, rtol=1e-3)
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
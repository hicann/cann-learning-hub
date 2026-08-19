#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


class MiniQwenMlp(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward_native(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward_custom(self, x):
        gate = self.gate_proj(x).contiguous()
        up = self.up_proj(x).contiguous()
        hidden = torch.ops.swiglu_custom.swiglu(gate, up)
        return self.down_proj(hidden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=896)
    parser.add_argument("--intermediate", type=int, default=4864)
    args = parser.parse_args()

    load_torch_ops()
    torch.manual_seed(42)
    device = "npu"

    model = MiniQwenMlp(args.hidden, args.intermediate).eval().to(device)
    x = torch.randn(args.batch, args.seq, args.hidden, dtype=torch.float32, device=device)

    with torch.no_grad():
        native = model.forward_native(x)
        custom = model.forward_custom(x)
        torch.npu.synchronize()

    diff = (custom - native).abs()
    print(f"max_abs_diff={diff.max().item():.8e}")
    print(f"mean_abs_diff={diff.mean().item():.8e}")
    ok = torch.allclose(custom, native, atol=2e-3, rtol=2e-3)
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
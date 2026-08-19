#!/usr/bin/env python3
"""QwenRoPeCustom — team-align: explicit forward_custom model (no monkey-patch)"""
import argparse, sys
from pathlib import Path
import torch
import torch_npu  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


class MiniQwenRoPE(torch.nn.Module):
    """Wrap RoPE: forward_native uses PyTorch, forward_custom uses our NPU op."""

    def __init__(self, head_dim: int = 64, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def _compute_cos_sin(self, seq_len: int, device):
        """Compute cos/sin table (replicates Qwen2RotaryEmbedding logic)."""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2,
                                                       dtype=torch.float32, device=device)
                                       / self.head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Native PyTorch RoPE."""
        batch, seq_len, n_heads, hd = x.shape
        cos, sin = self._compute_cos_sin(seq_len, x.device)
        cos = cos.view(1, seq_len, 1, hd)
        sin = sin.view(1, seq_len, 1, hd)
        return x * cos + self.rotate_half(x) * sin

    def forward_custom(self, x: torch.Tensor) -> torch.Tensor:
        """Custom NPU RoPE op (x, cos, sin pre-flattened)."""
        batch, seq_len, n_heads, hd = x.shape
        cos, sin = self._compute_cos_sin(seq_len, x.device)
        x_flat = x.reshape(-1, hd).contiguous()
        cos_flat = cos.expand(batch, seq_len, n_heads, hd).reshape(-1, hd).contiguous()
        sin_flat = sin.expand(batch, seq_len, n_heads, hd).reshape(-1, hd).contiguous()
        return torch.ops.qwen_rope_custom.rope_baseline(x_flat, cos_flat, sin_flat).view_as(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()

    load_torch_ops()
    torch.manual_seed(42)
    device = "npu"

    model = MiniQwenRoPE(args.head_dim).to(device)
    x = torch.randn(args.batch, args.seq, args.heads, args.head_dim,
                    dtype=torch.float32, device=device)

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

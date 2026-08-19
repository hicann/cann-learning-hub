#!/usr/bin/env python3
"""
RoPE 单算子正确性测试 — torch.ops.qwen_rope_custom.rope_baseline vs NumPy golden

用法:
  python tests/test_torch_op.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops
import torch


# ═══════════════════════════════════════════════════════════════
#  NumPy Reference (对标 PyTorch apply_rotary_pos_emb)
# ═══════════════════════════════════════════════════════════════

def rope_reference(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """y = x * cos + rotate_half(x) * sin  (float64 golden, 输出 float32)"""
    x = x.astype(np.float64)
    cos = cos.astype(np.float64)
    sin = sin.astype(np.float64)
    half = x.shape[-1] // 2
    y = np.empty_like(x)
    y[..., :half] = x[..., :half] * cos[..., :half] - x[..., half:] * sin[..., :half]
    y[..., half:] = x[..., half:] * cos[..., half:] + x[..., :half] * sin[..., half:]
    return y.astype(np.float32)


def make_cos_sin(seq_len: int, head_dim: int, base: float = 1e6):
    """生成 Qwen2.5 格式 cos/sin 表: [seq_len, head_dim]"""
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freqs = np.outer(np.arange(seq_len, dtype=np.float64), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32)
    return np.cos(emb), np.sin(emb)


# ═══════════════════════════════════════════════════════════════
#  Test
# ═══════════════════════════════════════════════════════════════

def test_correctness(seq_len=128, head_dim=64, base=1e6, rtol=1e-3, atol=1e-3):
    rng = np.random.default_rng(42)
    x_np = rng.normal(0, 1, size=(seq_len, head_dim)).astype(np.float32)
    cos_np, sin_np = make_cos_sin(seq_len, head_dim, base=base)

    y_golden = rope_reference(x_np, cos_np, sin_np)

    y_npu = rope_baseline(
        torch.from_numpy(x_np),
        torch.from_numpy(cos_np),
        torch.from_numpy(sin_np),
    ).numpy()

    diff = np.abs(y_npu - y_golden)
    rel = diff / (np.abs(y_golden) + 1e-12)

    print(f"  tokens={seq_len} head_dim={head_dim} base={base:.0e}")
    print(f"  max_abs_diff  = {float(diff.max()):.8e}")
    print(f"  mean_abs_diff = {float(diff.mean()):.8e}")
    print(f"  max_rel_diff  = {float(rel.max()):.8e}")

    if np.allclose(y_npu, y_golden, rtol=rtol, atol=atol):
        print("  ✅ PASS")
        return True
    else:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  ❌ FAIL at {idx}: NPU={y_npu[idx]:.6f} golden={y_golden[idx]:.6f}")
        return False


def main():
    print("=== QwenRoPeCustom — Torch Op Correctness Test ===\n")

    try:
        load_torch_ops()
    except Exception as e:
        print(f"[SKIP] Cannot load ops: {e}")
        sys.exit(0)

    global rope_baseline
    rope_baseline = torch.ops.qwen_rope_custom.rope_baseline

    shapes = [(128, 64), (128, 128), (256, 64), (64, 128)]
    all_pass = True
    for sl, hd in shapes:
        print(f"\n[TEST]")
        if not test_correctness(sl, hd):
            all_pass = False

    print(f"\n{'=== ALL PASS ===' if all_pass else '=== SOME FAILED ==='}")
    sys.exit(0 if all_pass else 1)


rope_baseline = None  # set after load_torch_ops()

if __name__ == "__main__":
    main()

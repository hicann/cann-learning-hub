#!/usr/bin/env python3
"""
msprof profiling target: Q/K fused RoPE torch op
Measures the fused torch op path for breakdown analysis.

Usage (via msprof wrapper):
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/profile_target.py
"""
import sys
import time
from pathlib import Path
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch_extension import load_torch_ops
load_torch_ops()

rope_qk = torch.ops.qwen_rope_custom_opt.rope_qk_compact

# Qwen2.5-0.5B representative shapes
SEQ, QH, KH, HD = 128, 24, 4, 64
N_WARMUP = 5
N_REPEAT = 50

rng = torch.Generator().manual_seed(42)
q = torch.randn(SEQ * QH, HD, generator=rng)
k = torch.randn(SEQ * KH, HD, generator=rng)
cos = torch.randn(SEQ, HD, generator=rng)
sin = torch.randn(SEQ, HD, generator=rng)

print(f"[TARGET] q={list(q.shape)} k={list(k.shape)} cos/sin={list(cos.shape)} warmup={N_WARMUP} repeat={N_REPEAT}")

for _ in range(N_WARMUP):
    q_out, k_out = rope_qk(q, k, cos, sin, SEQ, QH, KH)

t0 = time.perf_counter()
for i in range(N_REPEAT):
    q_out, k_out = rope_qk(q, k, cos, sin, SEQ, QH, KH)
t1 = time.perf_counter()

elapsed = t1 - t0
avg_ms = (elapsed / N_REPEAT) * 1000
print(f"[TARGET] {N_REPEAT} iters: total={elapsed*1000:.1f}ms avg={avg_ms:.3f}ms/call")

assert q_out.shape == q.shape
assert k_out.shape == k.shape
print(f"[TARGET] OK: q={list(q_out.shape)} k={list(k_out.shape)}")

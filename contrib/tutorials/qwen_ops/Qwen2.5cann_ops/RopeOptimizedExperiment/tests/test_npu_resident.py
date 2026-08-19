#!/usr/bin/env python3
"""Test NPU-resident RoPE op: correctness vs NumPy golden."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _setup_env import auto_setup; auto_setup()
from torch_extension import load_torch_ops
load_torch_ops()

import torch, torch_npu
import numpy as np

def rope_ref(x, cos, sin):
    x = x.astype(np.float64); cos=cos.astype(np.float64); sin=sin.astype(np.float64)
    half = x.shape[-1]//2
    y = np.empty_like(x)
    y[...,:half] = x[...,:half]*cos[...,:half] - x[...,half:]*sin[...,:half]
    y[...,half:] = x[...,half:]*cos[...,half:] + x[...,:half]*sin[...,half:]
    return y.astype(np.float32)

print("=== NPU-Resident RoPE Correctness Test ===")

# Test basic
seq, hd = 128, 64
rng = np.random.default_rng(42)
x_np = rng.normal(0, 1, size=(seq, hd)).astype(np.float32)
cos_np = np.cos(np.arange(seq*hd, dtype=np.float32).reshape(seq,hd)*0.1)
sin_np = np.sin(np.arange(seq*hd, dtype=np.float32).reshape(seq,hd)*0.1)
golden = rope_ref(x_np, cos_np, sin_np)

baseline = torch.ops.qwen_rope_custom_opt.rope_baseline

# CPU path
y_cpu = baseline(torch.from_numpy(x_np), torch.from_numpy(cos_np), torch.from_numpy(sin_np))
print(f"\n[CPU] max_diff={np.abs(y_cpu.numpy()-golden).max():.2e}")

# NPU-resident path
x_npu = torch.from_numpy(x_np).to('npu')
cos_npu = torch.from_numpy(cos_np).to('npu')
sin_npu = torch.from_numpy(sin_np).to('npu')
y_npu = baseline(x_npu, cos_npu, sin_npu)
print(f"[NPU] output device: {y_npu.device}")
y_npu_cpu = y_npu.cpu().numpy()
diff = np.abs(y_npu_cpu - golden)
print(f"[NPU] max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}")

assert np.allclose(y_npu_cpu, golden, rtol=1e-3, atol=1e-3), "NPU FAIL"
print("PASS: NPU-resident matches golden")

# Test compact
compact = torch.ops.qwen_rope_custom_opt.rope_compact
batch, heads = 2, 4
x4d = rng.normal(0, 1, size=(batch, heads, seq, hd)).astype(np.float32)
x_flat = x4d.reshape(-1, hd)
cos_c = np.broadcast_to(cos_np[None,:,:], (batch, seq, hd)).reshape(-1, hd)
sin_c = np.broadcast_to(sin_np[None,:,:], (batch, seq, hd)).reshape(-1, hd)
cos_c_exp = np.broadcast_to(cos_np[None,None,:,:], (batch, heads, seq, hd)).reshape(-1, hd)
sin_c_exp = np.broadcast_to(sin_np[None,None,:,:], (batch, heads, seq, hd)).reshape(-1, hd)
golden_c = rope_ref(x_flat, cos_c_exp, sin_c_exp)

x_flat_npu = torch.from_numpy(x_flat).to('npu')
cos_c_npu = torch.from_numpy(cos_c).to('npu')
sin_c_npu = torch.from_numpy(sin_c).to('npu')
y_c_npu = compact(x_flat_npu, cos_c_npu, sin_c_npu, seq, heads)
print(f"\n[COMPACT NPU] output device: {y_c_npu.device}")
y_c_cpu = y_c_npu.cpu().numpy()
diff_c = np.abs(y_c_cpu - golden_c)
print(f"[COMPACT NPU] max_diff={diff_c.max():.2e}  mean_diff={diff_c.mean():.2e}")
assert np.allclose(y_c_cpu, golden_c, rtol=1e-3, atol=1e-3), "COMPACT NPU FAIL"
print("PASS: NPU-resident compact matches golden")

# Test fused QK
fused = torch.ops.qwen_rope_custom_opt.rope_qk_compact
q_heads, kv_heads = 14, 2
q_flat = rng.normal(0, 1, size=(seq*q_heads, hd)).astype(np.float32)
k_flat = rng.normal(0, 1, size=(seq*kv_heads, hd)).astype(np.float32)
cos_ck = cos_np  # compact: [seq, hd]
sin_ck = sin_np
golden_q = rope_ref(q_flat, np.broadcast_to(cos_ck[None,:,:], (q_heads,seq,hd)).reshape(-1,hd), np.broadcast_to(sin_ck[None,:,:], (q_heads,seq,hd)).reshape(-1,hd))
golden_k = rope_ref(k_flat, np.broadcast_to(cos_ck[None,:,:], (kv_heads,seq,hd)).reshape(-1,hd), np.broadcast_to(sin_ck[None,:,:], (kv_heads,seq,hd)).reshape(-1,hd))

q_npu = torch.from_numpy(q_flat).to('npu')
k_npu = torch.from_numpy(k_flat).to('npu')
cos_npu2 = torch.from_numpy(cos_ck).to('npu')
sin_npu2 = torch.from_numpy(sin_ck).to('npu')
q_out, k_out = fused(q_npu, k_npu, cos_npu2, sin_npu2, seq, q_heads, kv_heads)
print(f"\n[FUSED QK NPU] q device: {q_out.device}  k device: {k_out.device}")

q_cpu = q_out.cpu().numpy(); k_cpu = k_out.cpu().numpy()
q_diff = np.abs(q_cpu - golden_q); k_diff = np.abs(k_cpu - golden_k)
print(f"[FUSED QK NPU] q max_diff={q_diff.max():.2e}  k max_diff={k_diff.max():.2e}")
assert np.allclose(q_cpu, golden_q, rtol=1e-3, atol=1e-3), "Q FAIL"
assert np.allclose(k_cpu, golden_k, rtol=1e-3, atol=1e-3), "K FAIL"
print("PASS: NPU-resident fused QK matches golden")

print("\n=== ALL TESTS PASSED ===")

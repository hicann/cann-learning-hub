#!/usr/bin/env python3
"""Benchmark only the Qwen2.5 RoPE call path.

The two RoPE extensions expose different torch namespaces and may load
different kernel libraries.  ``--compare`` therefore runs each variant in a
fresh Python process, avoiding library/namespace cross-contamination.

The timed region deliberately contains only the RoPE API call(s): no model
loading, tokenizer, projection, attention, or input/trig-table construction.
For the baseline it is the two Q/K ``rope_baseline`` calls used by Qwen; for
the optimized variant it is one ``rope_qk_compact`` call.  CPU tensors are
used intentionally so the reported number includes each implementation's
actual wrapper H2D/D2H behavior.

CANN environment is auto-detected by sourcing <cann>/set_env.sh and re-execing
with the full CANN environment.  Set ASCEND_HOME_PATH to override auto-detection.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "RopeBaselineExperiment"
OPTIMIZED = ROOT / "RopeOptimizedExperiment"

# Sentinel to prevent infinite re-exec loop
_ENV_GUARD = "_ROPE_BENCH_CANN_INJECTED"


# ── CANN auto-detection + os.execve restart ─────────────────────
def _detect_cann_path():
    """Return CANN toolkit root, or None."""
    candidates = [
        "/usr/local/Ascend/ascend-toolkit/latest",
        os.path.expanduser("~/Ascend/ascend-toolkit/latest"),
        os.path.expanduser("~/Ascend/ascend-toolkit/cann-8.5.0"),
        os.path.expanduser("~/ascend-toolkit/cann-8.5.0"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    cann_glob = os.path.expanduser("~/CANN")
    if os.path.isdir(cann_glob):
        versions = sorted(
            [d for d in os.listdir(cann_glob) if d.startswith("cann-")],
            reverse=True,
        )
        if versions:
            return os.path.join(cann_glob, versions[0])
    return None


def _source_cann_env(cann):
    """Source <cann>/set_env.sh via bash and return the resulting env dict."""
    set_env = os.path.join(cann, "set_env.sh")
    if not os.path.isfile(set_env):
        return {}
    try:
        result = subprocess.run(
            ["bash", "-c", f"source '{set_env}' >/dev/null 2>&1 && env"],
            capture_output=True, text=True, timeout=10,
        )
        env = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, _, val = line.partition("=")
                env[key] = val
        return env
    except Exception:
        return {}


def _ensure_cann_env():
    """Auto-detect CANN, source set_env.sh, and re-exec if needed."""
    if os.environ.get(_ENV_GUARD) == "1":
        return  # Already restarted

    cann = os.environ.get("ASCEND_HOME_PATH") or _detect_cann_path()
    if cann is None:
        print("[WARN] CANN not found. Set ASCEND_HOME_PATH.", file=sys.stderr)
        return

    cann_env = _source_cann_env(cann)
    if not cann_env:
        print("[WARN] Failed to source CANN set_env.sh.", file=sys.stderr)
        return

    # Merge: CANN env + project LD_LIBRARY_PATH + existing env
    new_env = os.environ.copy()
    for k, v in cann_env.items():
        new_env[k] = v
    # Add project out/lib
    lib_parts = []
    for proj in (BASELINE, OPTIMIZED):
        d = str(proj / "out" / "lib")
        if os.path.isdir(d):
            lib_parts.append(d)
    if lib_parts:
        existing_ld = new_env.get("LD_LIBRARY_PATH", "")
        new_env["LD_LIBRARY_PATH"] = ":".join(lib_parts) + (":" + existing_ld if existing_ld else "")
    new_env[_ENV_GUARD] = "1"

    if new_env == os.environ:
        return  # No change needed

    print(f"[INFO] Auto-detected CANN: {cann} — re-executing with full env", file=sys.stderr)
    os.execve(sys.executable, [sys.executable] + sys.argv, new_env)


# ── CLI ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 RoPE-only benchmark")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--variant", choices=("baseline", "optimized"))
    mode.add_argument("--compare", action="store_true",
                      help="benchmark both variants in isolated Python processes")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    return parser.parse_args()


def validate(args):
    for name in ("batch", "seq_len", "q_heads", "kv_heads", "head_dim", "warmup", "repeat"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.head_dim % 2:
        raise ValueError("--head-dim must be even")


def load_variant(variant):
    project = BASELINE if variant == "baseline" else OPTIMIZED
    sys.path.insert(0, str(project))
    from torch_extension import load_torch_ops
    load_torch_ops()


def run_variant(args):
    import torch

    load_variant(args.variant)
    torch.manual_seed(20260715)
    q_rows = args.batch * args.q_heads * args.seq_len
    k_rows = args.batch * args.kv_heads * args.seq_len
    trig_rows = args.batch * args.seq_len
    q = torch.randn(q_rows, args.head_dim, dtype=torch.float32)
    k = torch.randn(k_rows, args.head_dim, dtype=torch.float32)
    cos_compact = torch.randn(trig_rows, args.head_dim, dtype=torch.float32)
    sin_compact = torch.randn(trig_rows, args.head_dim, dtype=torch.float32)

    if args.variant == "baseline":
        cos_q = cos_compact.view(args.batch, args.seq_len, args.head_dim).unsqueeze(1).expand(
            -1, args.q_heads, -1, -1).reshape(q_rows, args.head_dim).contiguous()
        sin_q = sin_compact.view(args.batch, args.seq_len, args.head_dim).unsqueeze(1).expand(
            -1, args.q_heads, -1, -1).reshape(q_rows, args.head_dim).contiguous()
        cos_k = cos_compact.view(args.batch, args.seq_len, args.head_dim).unsqueeze(1).expand(
            -1, args.kv_heads, -1, -1).reshape(k_rows, args.head_dim).contiguous()
        sin_k = sin_compact.view(args.batch, args.seq_len, args.head_dim).unsqueeze(1).expand(
            -1, args.kv_heads, -1, -1).reshape(k_rows, args.head_dim).contiguous()
        op = torch.ops.qwen_rope_custom.rope_baseline

        def invoke():
            return op(q, cos_q, sin_q), op(k, cos_k, sin_k)
    else:
        op = torch.ops.qwen_rope_custom_opt.rope_qk_compact

        def invoke():
            return op(q, k, cos_compact, sin_compact, args.seq_len, args.q_heads, args.kv_heads)

    for _ in range(args.warmup):
        invoke()

    samples_us = []
    for _ in range(args.repeat):
        start = time.perf_counter_ns()
        invoke()
        samples_us.append((time.perf_counter_ns() - start) / 1_000.0)

    result = {
        "variant": args.variant,
        "scope": "RoPE Q/K operator call path only; CPU wrapper H2D/kernel/D2H included",
        "shape": {"batch": args.batch, "seq_len": args.seq_len, "q_heads": args.q_heads,
                  "kv_heads": args.kv_heads, "head_dim": args.head_dim},
        "warmup": args.warmup,
        "repeat": args.repeat,
        "mean_us": statistics.mean(samples_us),
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "stdev_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def compare(args):
    forwarded = ["--batch", str(args.batch), "--seq-len", str(args.seq_len), "--q-heads", str(args.q_heads),
                 "--kv-heads", str(args.kv_heads), "--head-dim", str(args.head_dim), "--warmup", str(args.warmup),
                 "--repeat", str(args.repeat)]
    env = os.environ.copy()
    results = {}
    for variant in ("baseline", "optimized"):
        command = [sys.executable, str(Path(__file__).resolve()), "--variant", variant, *forwarded]
        completed = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results[variant] = result
        print(completed.stdout, end="")
    speedup = results["baseline"]["mean_us"] / results["optimized"]["mean_us"]
    print(json.dumps({"comparison": "baseline_mean_us / optimized_mean_us", "speedup": speedup},
                     ensure_ascii=False, sort_keys=True))


def main():
    args = parse_args()
    validate(args)
    _ensure_cann_env()
    if args.compare:
        compare(args)
    else:
        run_variant(args)


if __name__ == "__main__":
    main()

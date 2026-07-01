#!/usr/bin/env python3
"""
Ascend NPU Profiling Trace Analyzer.

Extracts compute vs memory vs communication vs host-device sync breakdown
from a torch_npu ASCEND_PROFILER_OUTPUT/trace_view.json file.

Usage:
  # Auto-scan and compare latest 2 traces
  python3 tools/analyze_trace.py
  python3 tools/analyze_trace.py --base-dir /path/to/outputs

  # Analyze a single trace
  python3 tools/analyze_trace.py outputs/profiling_traces/*/ASCEND_PROFILER_OUTPUT/trace_view.json

  # Compare two specific traces
  python3 tools/analyze_trace.py --compare trace_baseline.json trace_fused.json
"""

import json
import argparse
import os
from collections import defaultdict

# Default base directory to scan for trace files
DEFAULT_TRACE_BASE_DIR = "/data/hanboyou/0630_cannlab/torchtitan-npu/outputs"


# ── Operation classification rules for Ascend NPU ──────────────────────
OPS_CLASSIFICATION = [
    # --- Compute: MatMul / Cube ---
    ("MatMul",              "Compute",   "MatMul"),
    ("BatchMatMul",         "Compute",   "MatMul"),
    ("mm",                  "Compute",   "MatMul"),
    ("linear",              "Compute",   "MatMul"),
    ("matmul",              "Compute",   "MatMul"),
    ("Linear",              "Compute",   "MatMul"),
    ("addmm",               "Compute",   "MatMul"),
    # --- Compute: Attention ---
    ("attention",           "Compute",   "Attention"),
    ("Attention",           "Compute",   "Attention"),
    ("flash_attn",          "Compute",   "Attention"),
    ("sdpa",                "Compute",   "Attention"),
    ("ScaledDotProduct",    "Compute",   "Attention"),
    ("attn",                "Compute",   "Attention"),
    # --- Compute: Activation ---
    ("silu",                "Compute",   "Activation"),
    ("SiLU",                "Compute",   "Activation"),
    ("gelu",                "Compute",   "Activation"),
    ("GELU",                "Compute",   "Activation"),
    ("swiglu",              "Compute",   "Activation"),
    ("SwiGLU",              "Compute",   "Activation"),
    # --- Compute: Element-wise / Fusion ---
    ("rotary",              "Compute",   "RoPE"),
    ("Rotary",              "Compute",   "RoPE"),
    ("rope",                "Compute",   "RoPE"),
    ("npu_rotary",          "Compute",   "RoPE"),
    ("rotary_mul",          "Compute",   "RoPE"),
    # --- Memory: Norm ---
    ("rms_norm",            "Memory",    "RMSNorm"),
    ("RMSNorm",             "Memory",    "RMSNorm"),
    ("layer_norm",          "Memory",    "LayerNorm"),
    ("LayerNorm",           "Memory",    "LayerNorm"),
    ("npu_rms_norm",        "Memory",    "RMSNorm"),
    ("add_rms_norm",        "Memory",    "RMSNorm"),
    # --- Memory: Element-wise ---
    ("add",                 "Memory",    "ElemWise"),
    ("mul",                 "Memory",    "ElemWise"),
    ("div",                 "Memory",    "ElemWise"),
    ("sub",                 "Memory",    "ElemWise"),
    ("neg",                 "Memory",    "ElemWise"),
    ("relu",                "Memory",    "ElemWise"),
    ("tanh",                "Memory",    "ElemWise"),
    ("sigmoid",             "Memory",    "ElemWise"),
    # --- Memory: Copy / Transfer ---
    ("copy_",               "Memory",    "DataCopy"),
    ("clone",               "Memory",    "DataCopy"),
    ("to(",                 "Memory",    "DataCopy"),
    ("contiguous",          "Memory",    "DataCopy"),
    ("reshape",             "Memory",    "Reshape"),
    ("view",                "Memory",    "Reshape"),
    ("transpose",           "Memory",    "Reshape"),
    ("permute",             "Memory",    "Reshape"),
    ("unsqueeze",           "Memory",    "Reshape"),
    # --- Memory: Embedding ---
    ("embedding",           "Memory",    "Embedding"),
    ("Embedding",           "Memory",    "Embedding"),
    # --- Sync: Host-device ---
    ("Synchronize",         "Sync",      "StreamSync"),
    ("item",                "Sync",      "HostDevice"),
    ("item()",              "Sync",      "HostDevice"),
    ("_local_scalar_dense", "Sync",      "HostDevice"),
    ("EVENT_WAIT",          "Sync",      "EventWait"),
    ("EVENT",               "Sync",      "EventWait"),
    # --- Communication ---
    ("all_reduce",          "Comm",      "AllReduce"),
    ("AllReduce",           "Comm",      "AllReduce"),
    ("all_gather",          "Comm",      "AllGather"),
    ("AllGather",           "Comm",      "AllGather"),
    ("reduce_scatter",      "Comm",      "ReduceScatter"),
    ("ReduceScatter",       "Comm",      "ReduceScatter"),
    ("HCCL",                "Comm",      "HCCL"),
    ("hccl",                "Comm",      "HCCL"),
    # --- Free ---
    ("Free",                "Memory",    "Free"),
    ("Malloc",              "Memory",    "Alloc"),
    # --- DataLoader ---
    ("DataLoader",          "Data",      "DataLoad"),
    ("enumerate",           "Data",      "DataLoad"),
]


def classify_op(name: str) -> tuple[str, str]:
    """Return (category, sub_category) for an operator name."""
    for pattern, cat, sub in OPS_CLASSIFICATION:
        if pattern in name:
            return cat, sub
    return ("Other", "Other")


def analyze_trace(path: str) -> dict:
    """Analyze a single trace file and return breakdown stats."""
    with open(path) as f:
        data = json.load(f)

    stats = defaultdict(lambda: defaultdict(float))
    total_time = 0
    total_events = len(data)
    top_ops = []

    # Extra diagnostics
    acl_kernel_count = 0
    fused_ops = defaultdict(int)
    fragmented_ops = defaultdict(int)

    for event in data:
        dur = event.get("dur", 0)
        name = event.get("name", "?")

        # Count ACL kernels
        if "aclnn" in name or "AscendCL" in name:
            acl_kernel_count += 1

        # Detect fused operators
        for fused_pattern in ["npu_rms_norm", "add_rms_norm", "npu_rotary"]:
            if fused_pattern in name:
                fused_ops[fused_pattern] += 1

        # Track fragmented ACL ops
        for frag_pattern in ["aclnnPow", "aclnnMul", "aclnnCast",
                             "aclnnReduceMean", "aclnnAddcDiv"]:
            if frag_pattern in name:
                fragmented_ops[frag_pattern] += 1

        if dur <= 0:
            continue

        cat, sub = classify_op(name)
        stats[cat][sub] += dur
        total_time += dur
        top_ops.append((dur, name, cat, sub))

    top_ops.sort(reverse=True)

    breakdown = {}
    for cat in ["Compute", "Memory", "Sync", "Comm", "Data", "Other"]:
        cat_time = sum(stats[cat].values())
        pct = cat_time / total_time * 100 if total_time > 0 else 0
        subs = {sub: {"time_us": t, "pct": t/total_time*100 if total_time>0 else 0}
                for sub, t in sorted(stats[cat].items(), key=lambda x: -x[1])}
        breakdown[cat] = {"time_us": cat_time, "pct": pct, "subs": subs}

    return {
        "path": path,
        "total_events": total_events,
        "total_time_us": total_time,
        "total_time_ms": total_time / 1000,
        "breakdown": breakdown,
        "top_ops": top_ops[:30],
        "aclkernel_count": acl_kernel_count,
        "fused_ops": dict(fused_ops),
        "fragmented_ops": dict(fragmented_ops),
    }


def print_analysis(result: dict):
    """Pretty print analysis results."""
    print(f"\n{'='*70}")
    print(f"Trace: {result['path']}")
    print(f"Events: {result['total_events']:,}  |  Time: {result['total_time_ms']:.1f} ms")
    print(f"{'='*70}")

    print(f"\n{'─'*50}")
    print(f"  {'Category':<18} {'Time (ms)':>10} {'Pct':>8}  Sub-breakdown")
    print(f"{'─'*50}")

    for cat in ["Compute", "Memory", "Sync", "Comm", "Data", "Other"]:
        b = result["breakdown"][cat]
        if b["pct"] < 0.01:
            continue
        print(f"  {cat:<18} {b['time_us']/1000:>10.2f} {b['pct']:>7.2f}%")
        for sub, s in list(b["subs"].items())[:5]:
            if s["pct"] > 0.01:
                print(f"    └─ {sub:<22} {s['time_us']/1000:>7.2f}ms ({s['pct']:.1f}%)")

    print(f"\n{'─'*50}")
    print(f"  Top 15 Ops by Duration")
    print(f"{'─'*50}")
    for i, (dur, name, cat, sub) in enumerate(result["top_ops"][:15]):
        print(f"  {i+1:2}. [{cat}/{sub}] {dur/1000:>8.2f}ms  {name[:60]}")

    compute_time = result["breakdown"]["Compute"]["time_us"]
    memory_time = result["breakdown"]["Memory"]["time_us"]
    sync_time = result["breakdown"]["Sync"]["time_us"]
    if compute_time > 0:
        ratio = memory_time / compute_time
        print(f"\n{'─'*50}")
        print(f"  Memory/Compute ratio: {ratio:.2f}")
        print(f"  Sync overhead: {sync_time/1000:.1f}ms ({result['breakdown']['Sync']['pct']:.1f}%)")
        if ratio > 0.8:
            print(f"  >> Memory-bound (ratio > 0.8)")
        elif ratio < 0.5:
            print(f"  >> Compute-bound (ratio < 0.5)")
        else:
            print(f"  >> Balanced")


def compare_traces(path1: str, path2: str):
    """Compare two profiling traces with fusion verification signals."""
    r1 = analyze_trace(path1)
    r2 = analyze_trace(path2)

    print(f"\n{'='*70}")
    print(f"  PROFILING COMPARISON")
    print(f"{'='*70}")

    # ── Signal 1: ACL Kernel count ──
    acl1 = r1["aclkernel_count"]
    acl2 = r2["aclkernel_count"]
    acl_chg = (acl2 - acl1) / acl1 * 100 if acl1 else 0
    print(f"\n  ── 验证信号一：ACL Kernel 调用次数 ──")
    print(f"  ACL kernel 调用:  {acl1:,} → {acl2:,}  ({acl_chg:+.0f} / {acl_chg:+.1f}%)")

    # ── Signal 2: Fused op detection ──
    print(f"\n  ── 验证信号二：CPU 侧融合算子 ──")
    fused_ops_to_check = ["npu_rms_norm", "add_rms_norm"]
    for op_name in fused_ops_to_check:
        in1 = r1["fused_ops"].get(op_name, 0)
        in2 = r2["fused_ops"].get(op_name, 0)
        if in1 > 0 or in2 > 0:
            status1 = f"{in1} 次" if in1 > 0 else "无"
            status2 = f"{in2} 次" if in2 > 0 else "无"
            if in1 == 0 and in2 > 0:
                verdict = "✅ baseline 无 → fused 有，融合正确生效"
            elif in1 > 0 and in2 == 0:
                verdict = "❌ baseline 有 → fused 无，融合可能未启用"
            else:
                verdict = "⚠️  两者都存在，检查转换是否正确"
            print(f"  {op_name}:  {status1} → {status2}  {verdict}")

    # ── Signal 3: Fragmentation reduction ──
    print(f"\n  ── 验证信号三：碎片化算子减少 ──")
    frag_patterns = [
        ("Pow", "aclnnPow"),
        ("Mul", "aclnnMul"),
        ("Cast", "aclnnCast"),
        ("ReduceMean", "aclnnReduceMean"),
        ("AddcDiv", "aclnnAddcDiv"),
    ]
    for label, pattern in frag_patterns:
        c1 = r1["fragmented_ops"].get(pattern, 0)
        c2 = r2["fragmented_ops"].get(pattern, 0)
        if c1 > 0 or c2 > 0:
            if c1 == 0 and c2 == 0:
                continue
            elif c1 > 0 and c2 == 0:
                print(f"  {label} 碎片: {c1} → {c2}  (100% 消除)")
            elif c2 == 0:
                print(f"  {label} 碎片: {c1} → {c2}")
            else:
                pct = (c1 - c2) / c1 * 100
                print(f"  {label} 碎片: {c1} → {c2}  ({pct:+.0f}%)")

    # ── Performance data ──
    print(f"\n  ── 性能数据 ──")
    print(f"  {'Metric':<25} {'Trace 1':>15} {'Trace 2':>15} {'Change':>12}")
    print(f"  {'─'*68}")

    for metric, m1, m2, fmt in [
        ("Events", r1["total_events"], r2["total_events"], ",d"),
        ("Total time (ms)", r1["total_time_ms"], r2["total_time_ms"], ".1f"),
        ("ACL kernels", acl1, acl2, ",d"),
    ]:
        chg = (m2 - m1) / m1 * 100 if m1 else 0
        m1s = f"{m1:{fmt}}" if isinstance(m1, int) else f"{m1:{fmt}}"
        m2s = f"{m2:{fmt}}" if isinstance(m2, int) else f"{m2:{fmt}}"
        print(f"  {metric:<25} {m1s:>15} {m2s:>15} {chg:>+11.1f}%")

    for cat in ["Compute", "Memory", "Sync", "Comm", "Data"]:
        t1 = r1["breakdown"][cat]["time_us"] / 1000
        t2 = r2["breakdown"][cat]["time_us"] / 1000
        chg = (t2 - t1) / t1 * 100 if t1 else 0
        print(f"  {cat:<25} {t1:>14.2f}ms {t2:>14.2f}ms {chg:>+11.1f}%")

    ratio1 = r1["breakdown"]["Memory"]["time_us"] / max(r1["breakdown"]["Compute"]["time_us"], 1)
    ratio2 = r2["breakdown"]["Memory"]["time_us"] / max(r2["breakdown"]["Compute"]["time_us"], 1)
    print(f"\n  Memory/Compute ratio: {ratio1:.2f} → {ratio2:.2f}")

    print(f"{'─'*70}")
    print(f"\n  Trace 1 (baseline): {path1}")
    print(f"  Trace 2 (fused):    {path2}")


def find_latest_traces(base_dir: str, n: int = 2) -> list[str]:
    """Find the latest N trace_view.json files under base_dir.

    Searches recursively for trace_view.json and returns them sorted
    by modification time (oldest first → baseline, newest last → fused).
    """
    traces = []
    for root, dirs, files in os.walk(base_dir):
        if "trace_view.json" in files:
            full_path = os.path.join(root, "trace_view.json")
            traces.append((os.path.getmtime(full_path), full_path))

    if not traces:
        raise FileNotFoundError(
            f"No trace_view.json found under {base_dir}. "
            f"Run profiling first to generate traces."
        )

    traces.sort()
    return [path for _, path in traces[-n:]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ascend NPU Profiling Trace Analyzer")
    parser.add_argument("trace", nargs="?", help="Path to trace_view.json")
    parser.add_argument("--compare", nargs=2, help="Compare two specific traces")
    parser.add_argument("--base-dir", default=DEFAULT_TRACE_BASE_DIR,
                        help=f"Base directory to scan for trace files (default: {DEFAULT_TRACE_BASE_DIR})")
    args = parser.parse_args()

    if args.compare:
        compare_traces(args.compare[0], args.compare[1])
    elif args.trace:
        result = analyze_trace(args.trace)
        print_analysis(result)
    else:
        traces = find_latest_traces(args.base_dir)
        if len(traces) < 2:
            print(f"Error: Need at least 2 trace files to compare. Found {len(traces)} in {args.base_dir}")
            print("Usage: python3 analyze_trace.py <trace_view.json>")
            print("       python3 analyze_trace.py --compare <trace1.json> <trace2.json>")
            exit(1)
        print(f"\nAuto-detected traces from: {args.base_dir}")
        print(f"  Baseline: {traces[0]}")
        print(f"  Fused:    {traces[1]}")
        compare_traces(traces[0], traces[1])

#!/usr/bin/env python3
"""
analyze_profiling.py — 自动发现并对比 baseline 与 fused 的 profiling trace。

用法：
  1. 确认 profile_traces 目录下有两次训练的 trace 输出：
     NGPU=2 ASCEND_RT_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh \
       --module torchtitan_npu.models.qwen3 \
       --config sft_qwen3_1_7b_wordle \
       --training.steps=10 \
       --profiling.enable_profiling

  2. 运行本脚本：
     python analyze_profiling.py --profile_dir ./outputs/profile_traces

脚本会：
  - 自动发现所有含 trace_view.json 的子目录（按修改时间排序）
  - 先跑的 = baseline，后跑的 = fused
  - 按 kernel name 分类统计算子耗时
  - 对比两组数据，验证融合算子是否生效
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ─── 融合算子特征 ─────────────────────────────────────────────
# CPU 层面上：融合后出现 npu::npu_rms_norm
FUSED_CPU_OPS = ["npu::npu_rms_norm"]

# 基线中 RMSNorm 碎片化时产生的 AscendCL 微内核
BASELINE_FRAGMENTED_ACL = [
    "AscendCL@aclnnPowTensorScalar",   # pow 微内核
    "AscendCL@aclnnMul",               # mul 微内核（RMSNorm 的 gamma 乘法）
    "AscendCL@aclnnInplaceMul",
]


def log(msg: str, level: str = "INFO"):
    """统一日志格式，让用户感受到每一步的执行进度。"""
    prefix = {"INFO": "  📋", "OK": "  ✅", "WARN": "  ⚠️", "STEP": "🔍", "RESULT": "📊"}
    p = prefix.get(level, "  •")
    print(f"{p} {msg}")


def discover_traces(profile_dir: Path) -> list[Path]:
    """在 profile_dir 下发现所有 trace_view.json，按文件修改时间排序。

    返回排序后的 Path 列表，先跑的在前（baseline），后跑的在后（fused）。
    """
    log(f"扫描 profiling 输出目录: {profile_dir}", "STEP")

    traces = []
    for subdir in sorted(profile_dir.iterdir()):
        if not subdir.is_dir():
            continue
        trace_file = subdir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
        try:
            if trace_file.exists():
                mtime = trace_file.stat().st_mtime
                traces.append((mtime, trace_file))
                log(f"发现 trace: {subdir.name} "
                    f"(时间戳: {mtime:.0f})")
        except PermissionError:
            log(f"跳过（无权限访问）: {subdir.name}", "WARN")
            continue

    if len(traces) < 2:
        log(f"警告：只找到 {len(traces)} 个 trace 文件，需要至少 2 个才能对比", "WARN")
        if len(traces) == 0:
            log("请确认已运行 baseline 和 fused 两次 profiling 训练", "WARN")
            sys.exit(1)

    # 按修改时间排序
    traces.sort(key=lambda t: t[0])
    return [t[1] for t in traces]


def load_trace(trace_path: Path) -> list[dict]:
    """加载 trace_view.json。

    ASCEND Profiler 输出的 trace 是 JSON 数组，每个元素包含：
      name: 事件名（如 "aten::rms_norm", "AscendCL@aclnnPowTensorScalar"）
      cat:  类别（"cpu_op", "async_npu", "fwdbwd", 或空字符串=NPU调度）
      dur:  耗时（μs），可以是字符串
      ts:   时间戳（字符串格式的 float）
    """
    log(f"加载 trace 文件: {trace_path.parent.parent.name}/{trace_path.parent.name}/{trace_path.name}", "STEP")

    with open(trace_path, "r") as f:
        data = json.load(f)

    # 处理两种格式：直接的数组 或 {traceEvents: [...]}
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        events = data.get("traceEvents", data.get("trace_events", []))
    else:
        log(f"  错误：无法识别的 trace 格式", "WARN")
        return []

    log(f"  文件大小: {trace_path.stat().st_size / 1024 / 1024:.1f} MiB", "INFO")
    log(f"  总事件数: {len(events):,}", "INFO")

    # 统计事件类型分布
    cats = defaultdict(int)
    for evt in events:
        cat = evt.get("cat", "")
        if not cat:
            cat = "(npu_scheduling)"
        cats[cat] += 1
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:6]:
        log(f"    {cat:<25s} {count:>8,} 条", "INFO")

    return events


def categorize_events(events: list[dict]) -> dict[str, list[dict]]:
    """将 trace 事件按算子名分类。

    关注两类事件：
      - cpu_op: PyTorch 层算子调用（如 atn::rms_norm, npu::npu_rms_norm）
      - AscendCL@aclnn*: NPU kernel 调用（如 PowTensorScalar, Mul）
    """
    operators = defaultdict(list)

    for evt in events:
        name = evt.get("name", "")
        cat = evt.get("cat", "")

        # 只看 cpu_op 和 AscendCL kernel 事件
        if cat != "cpu_op" and not name.startswith("AscendCL@aclnn"):
            continue
        if not name:
            continue

        operators[name].append(evt)

    return dict(operators)


def _safe_dur(evt: dict) -> float:
    """安全提取 dur，兼容 string/float/int。"""
    d = evt.get("dur", 0)
    if isinstance(d, str):
        try:
            return float(d)
        except (ValueError, TypeError):
            return 0.0
    return float(d)


def aggregate_stats(operators: dict[str, list[dict]]) -> dict:
    """从分类后的算子事件中提取统计信息。"""
    stats = {
        "total_kernel_events": 0,
        "total_duration_us": 0.0,
        "acl_kernel_launches": 0,       # AscendCL kernel 调用总数
        "fragmented_pow_us": 0.0,       # 基线中 Pow 微内核耗时
        "fragmented_mul_us": 0.0,       # 基线中 Mul 微内核耗时
        "has_npu_rms_norm": False,
        "top_cpu_ops": [],              # 耗时最长的 top-10 cpu_op
        "top_acl_ops": [],              # 调用最多的 top-10 AscendCL kernel
    }

    # 分别统计 cpu_op 和 ACL kernel
    cpu_times: dict[str, float] = defaultdict(float)
    cpu_counts: dict[str, int] = defaultdict(int)
    acl_counts: dict[str, int] = defaultdict(int)

    for name, evts in operators.items():
        total_dur = sum(_safe_dur(e) for e in evts)
        stats["total_kernel_events"] += len(evts)
        stats["total_duration_us"] += total_dur

        if name.startswith("AscendCL@aclnn"):
            acl_counts[name] += len(evts)
            stats["acl_kernel_launches"] += len(evts)
        else:
            # cpu_op
            cpu_times[name] += total_dur
            cpu_counts[name] += len(evts)

    # -- 检测融合算子（在 cpu_op 中）--
    for name in cpu_times:
        if "npu_rms_norm" in name:
            stats["has_npu_rms_norm"] = True

    # -- 碎片化的 ACL kernel --
    for frag_name in BASELINE_FRAGMENTED_ACL:
        if frag_name in acl_counts:
            # 用算子名中的关键词判断类型
            if "Pow" in frag_name:
                stats["fragmented_pow_us"] += acl_counts[frag_name]
            if "Mul" in frag_name:
                stats["fragmented_mul_us"] += acl_counts[frag_name]

    # Top-10 cpu_op（按总耗时）
    sorted_cpu = sorted(cpu_times.items(), key=lambda x: -x[1])[:10]
    stats["top_cpu_ops"] = [
        {"name": name, "time_us": time_us, "calls": cpu_counts[name]}
        for name, time_us in sorted_cpu
    ]

    # Top-10 ACL kernel（按调用次数）
    sorted_acl = sorted(acl_counts.items(), key=lambda x: -x[1])[:10]
    stats["top_acl_ops"] = [
        {"name": name, "calls": count}
        for name, count in sorted_acl
    ]

    return stats


def print_stats(label: str, stats: dict):
    """格式化打印统计结果。"""
    log(f"", "INFO")
    log(f"{'─' * 60}", "INFO")
    log(f"{label}", "RESULT")
    log(f"{'─' * 60}", "INFO")
    log(f"  算子事件总数:     {stats['total_kernel_events']:>8,}", "INFO")
    log(f"  其中 ACL kernel:   {stats['acl_kernel_launches']:>8,}", "INFO")
    log(f"  总耗时:           {stats['total_duration_us']:>10,.0f} μs "
        f"({stats['total_duration_us'] / 1000:,.1f} ms)", "INFO")

    log(f"", "INFO")
    log(f"  CPU 侧耗时 Top-10:", "RESULT")
    for i, k in enumerate(stats["top_cpu_ops"], 1):
        log(f"    {i:2d}. {k['name']:<45s} "
            f"{k['time_us']:>10,.0f} μs  ({k['calls']:>4d} calls)", "INFO")

    log(f"", "INFO")
    log(f"  AscendCL Kernel 调用 Top-10:", "RESULT")
    for i, k in enumerate(stats["top_acl_ops"], 1):
        log(f"    {i:2d}. {k['name']:<55s} {k['calls']:>5d} calls", "INFO")

    # 融合算子检测
    log(f"", "INFO")
    log(f"  融合算子检测:", "RESULT")
    status_rms = "✅ 已生效" if stats["has_npu_rms_norm"] else "❌ 未检测到"
    log(f"    npu_rms_norm:  {status_rms}", "INFO")

    if not stats["has_npu_rms_norm"]:
        log(f"    Pow 碎片调用:  {stats['fragmented_pow_us']:,.0f} 次", "WARN")
        log(f"    Mul 碎片调用:  {stats['fragmented_mul_us']:,.0f} 次", "WARN")


def compare(baseline: dict, fused: dict):
    """对比基线 vs 融合的统计结果。"""
    log(f"", "INFO")
    log(f"{'═' * 60}", "RESULT")
    log(f"  对比结果", "RESULT")
    log(f"{'═' * 60}", "RESULT")

    # ACL kernel 调用次数对比
    delta_acl = baseline["acl_kernel_launches"] - fused["acl_kernel_launches"]
    if baseline["acl_kernel_launches"] > 0:
        pct_acl = delta_acl / baseline["acl_kernel_launches"] * 100
        log(f"  ACL kernel 调用:  {baseline['acl_kernel_launches']:,} → "
            f"{fused['acl_kernel_launches']:,}  "
            f"({delta_acl:+,} / {pct_acl:+.1f}%)", "RESULT")

    # 总事件数对比
    delta_events = baseline["total_kernel_events"] - fused["total_kernel_events"]
    if baseline["total_kernel_events"] > 0:
        pct_ev = delta_events / baseline["total_kernel_events"] * 100
        log(f"  算子事件总数:    {baseline['total_kernel_events']:,} → "
            f"{fused['total_kernel_events']:,}  "
            f"({delta_events:+,} / {pct_ev:+.1f}%)", "RESULT")

    # 总耗时对比
    delta_time = baseline["total_duration_us"] - fused["total_duration_us"]
    if baseline["total_duration_us"] > 0:
        pct_time = delta_time / baseline["total_duration_us"] * 100
        log(f"  总耗时:          {baseline['total_duration_us']:,.0f} → "
            f"{fused['total_duration_us']:,.0f} μs  "
            f"({delta_time:+,.0f} μs / {pct_time:+.1f}%)", "RESULT")

    # 融合算子判断
    log(f"", "INFO")
    log(f"  融合算子验证:", "RESULT")

    if not baseline["has_npu_rms_norm"] and fused["has_npu_rms_norm"]:
        log(f"    npu_rms_norm:  ✅ baseline 无 → fused 有，融合正确生效", "OK")
    elif baseline["has_npu_rms_norm"]:
        log(f"    npu_rms_norm:  ⚠️ baseline 中已存在，可能两次都跑了 fused", "WARN")
    else:
        log(f"    npu_rms_norm:  ❌ fused 中未检测到，请检查 model_converters", "WARN")


    # 碎片减少验证
    pow_delta = baseline["fragmented_pow_us"] - fused["fragmented_pow_us"]
    mul_delta = baseline["fragmented_mul_us"] - fused["fragmented_mul_us"]
    if pow_delta > 0 or mul_delta > 0:
        log(f"", "INFO")
        log(f"  碎片化减少:", "RESULT")
        if pow_delta > 0 and baseline["fragmented_pow_us"] > 0:
            pct = pow_delta / baseline["fragmented_pow_us"] * 100
            log(f"    Pow 碎片: {baseline['fragmented_pow_us']:,.0f} → "
                f"{fused['fragmented_pow_us']:,.0f}  ({pct:.0f}%)", "OK")
        if mul_delta > 0 and baseline["fragmented_mul_us"] > 0:
            pct = mul_delta / baseline["fragmented_mul_us"] * 100
            log(f"    Mul 碎片: {baseline['fragmented_mul_us']:,.0f} → "
                f"{fused['fragmented_mul_us']:,.0f}  ({pct:.0f}%)", "OK")


def main():
    parser = argparse.ArgumentParser(
        description="自动发现并对比 baseline 与 fused profiling trace"
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="./outputs/profile_traces",
        help="profiling 输出根目录 (默认: ./outputs/profile_traces)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="手动指定 baseline trace 路径（覆盖自动发现）",
    )
    parser.add_argument(
        "--fused",
        type=str,
        default=None,
        help="手动指定 fused trace 路径（覆盖自动发现）",
    )
    args = parser.parse_args()

    print()
    log("═" * 55, "STEP")
    log("  Ascend NPU 融合算子 Profiling 分析工具", "STEP")
    log("═" * 55, "STEP")

    profile_dir = Path(args.profile_dir)

    # ── 发现 trace 文件 ──
    if args.baseline and args.fused:
        log("使用手动指定的 trace 路径", "INFO")
        baseline_trace = Path(args.baseline)
        fused_trace = Path(args.fused)
    else:
        traces = discover_traces(profile_dir)
        baseline_trace = traces[0]
        fused_trace = traces[-1]  # 最后一个 = 最晚跑的 = fused

        log(f"", "INFO")
        log(f"  自动排序结果（按时间）:", "OK")
        log(f"    1. baseline: {baseline_trace.parent.parent.name}", "OK")
        log(f"    2. fused:    {fused_trace.parent.parent.name}", "OK")

    # ── 加载 & 分析 ──
    baseline_events = load_trace(baseline_trace)
    baseline_ops = categorize_events(baseline_events)
    baseline_stats = aggregate_stats(baseline_ops)

    fused_events = load_trace(fused_trace)
    fused_ops = categorize_events(fused_events)
    fused_stats = aggregate_stats(fused_ops)

    # ── 输出 ──
    print_stats("📂 BASELINE（基线 — 无融合算子）", baseline_stats)
    print_stats("📂 FUSED（优化 — 已启用融合算子）", fused_stats)
    compare(baseline_stats, fused_stats)

    print()
    log("分析完成。", "OK")
    print()


if __name__ == "__main__":
    main()

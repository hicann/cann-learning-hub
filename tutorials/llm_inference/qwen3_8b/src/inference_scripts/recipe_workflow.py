"""Helpers for the Qwen3-8B recipes-style notebook workflow."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import yaml

from qwen3_npu_adaptation import run_generation_text_sanity_check

from .inference_qwen3_8b import (
    build_output_rows,
    count_prompt_tokens,
    ensure_recipe_log_ok,
    maybe_silence_output,
    parse_recipe_log,
    read_recipe_default_prompts,
    resolve_model_path,
)


def prepare_runtime_yaml(template_path: Path, output_path: Path, *, model_path: str) -> dict[str, Any]:
    """Copy a ready recipes YAML and fill in the resolved model path."""
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config.setdefault("model_config", {})["model_path"] = str(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config

def prepare_profiling_yaml(template_path: Path, output_path: Path, *, enable_profiler: bool) -> dict[str, Any]:
    """Copy a ready recipes YAML and fill in the resolved model path."""
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config.setdefault("model_config", {})["enable_profiler"] = enable_profiler
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config

def print_yaml_focus(yaml_path: Path) -> None:
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    model_config = config["model_config"]
    data_config = config["data_config"]
    scheduler_config = config["scheduler_config"]
    print("YAML 文件 =", yaml_path)
    print("model_path =", model_config.get("model_path"))
    print("enable_profiler =", model_config.get("enable_profiler"))
    print("enable_npu_rmsnorm =", model_config.get("custom_params", {}).get("enable_npu_rmsnorm"))
    print("enable_add_rmsnorm =", model_config.get("custom_params", {}).get("enable_add_rmsnorm"))
    print("dataset =", data_config.get("dataset"))
    print("input_truncated_len =", data_config.get("input_truncated_len"))
    print("max_new_tokens =", scheduler_config.get("max_new_tokens"))
    print("batch_size =", scheduler_config.get("batch_size"))


def find_recipe_case_dir(recipe_root: Path, yaml_path: Path) -> Path | None:
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    model_name = config.get("model_config", {}).get("model_name", "qwen3_8b")
    date = time.strftime("%Y%m%d")
    expected = recipe_root / "models" / "qwen" / "res" / date / f"{model_name}_{yaml_path.stem}"
    if expected.is_dir():
        return expected

    res_root = recipe_root / "models" / "qwen" / "res"
    candidates = sorted(
        res_root.glob(f"**/{model_name}_{yaml_path.stem}"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _read_profiler_rows(csv_path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            op_type = (
                row.get("OP Type")
                or row.get("Op Type")
                or row.get("Name")
                or row.get("Op Name")
                or row.get("Kernel Name")
                or row.get("Task Name")
                or ""
            )
            count_value = row.get("Count") or row.get("Call Count") or "1"
            total_value = (
                row.get("Total Time(us)")
                or row.get("Total(us)")
                or row.get("Duration(us)")
                or row.get("Task Duration(us)")
                or row.get("Execute Time(us)")
                or row.get("time(us)")
                or "0"
            )
            ratio_value = row.get("Ratio(%)") or row.get("Time Ratio(%)") or row.get("ratio(%)") or "0"
            try:
                count = int(float(str(count_value).strip() or 1))
                total_us = float(str(total_value).strip() or 0)
                ratio = float(str(ratio_value).strip() or 0)
            except ValueError:
                continue
            if not op_type or total_us <= 0:
                continue
            grouped_row = grouped.setdefault(
                str(op_type),
                {"op_type": str(op_type), "count": 0, "total_us": 0.0, "ratio": 0.0},
            )
            grouped_row["count"] += count
            grouped_row["total_us"] += total_us
            grouped_row["ratio"] += ratio
    rows = list(grouped.values())
    rows.sort(key=lambda item: item["total_us"], reverse=True)
    total_us = sum(row["total_us"] for row in rows)
    if total_us > 0:
        for row in rows:
            row["ratio"] = row["total_us"] / total_us * 100
    return rows


def discover_profile_artifacts(profile_root: Path) -> dict[str, Any]:
    profile_root = Path(profile_root)
    artifacts = {
        "profile_root": str(profile_root),
        "kernel_details": [],
        "op_summary": [],
        "op_statistic": [],
        "trace": [],
    }
    if not profile_root.exists():
        return artifacts

    patterns = {
        "kernel_details": ["*kernel_details*.csv", "*kernel*detail*.csv"],
        "op_summary": ["op_summary*.csv", "*op_summary*.csv"],
        "op_statistic": ["op_statistic*.csv", "*op_statistic*.csv"],
        "trace": ["trace*.json", "*trace*.json", "*.json"],
    }
    for key, key_patterns in patterns.items():
        paths = []
        for pattern in key_patterns:
            paths.extend(profile_root.rglob(pattern))
        unique_paths = sorted({path.resolve() for path in paths})
        artifacts[key] = [str(path) for path in unique_paths]
    return artifacts


def write_profiler_summary(artifacts: dict[str, Any], output_path: Path, row_limit: int = 40) -> str:
    sources = []
    for key in ("kernel_details", "op_statistic", "op_summary"):
        for item in artifacts.get(key, []):
            sources.append((key, Path(item)))
    for source_kind, source_path in sources:
        rows = _read_profiler_rows(source_path)
        if not rows:
            continue
        lines = [
            f"Profiler summary from {source_kind}",
            f"source: {source_path}",
            f"{'OP Type':<48} {'Count':>10} {'Total(us)':>14} {'Ratio(%)':>10}",
            "-" * 88,
        ]
        for row in rows[:row_limit]:
            lines.append(
                f"{row['op_type']:<48} {row['count']:>10} {row['total_us']:>14.3f} {row['ratio']:>10.3f}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return str(output_path)
    raise RuntimeError("Profiler 目录中没有可解析的 kernel_details/op_statistic/op_summary CSV。")


def collect_recipe_metrics(
    recipe_root: Path,
    work_dir: Path,
    metrics_tag: str,
    yaml_path: Path,
    run_info: dict[str, Any],
    *,
    model_id: str,
    resolved_model_path: str,
    model_source: str,
    optimization_mode: str,
    max_new_tokens: int,
    max_input_tokens: int,
    enable_thinking: bool = False,
    quiet_model_io: bool = True,
) -> dict[str, Any]:
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    model_config = config.get("model_config", {})
    custom_params = model_config.get("custom_params", {})
    scheduler_config = config.get("scheduler_config", {})

    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = work_dir / "metrics"
    outputs_dir = work_dir / "outputs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_recipe_default_prompts()
    recipe_log_path = Path(run_info["recipe_log_path"])
    parsed_log = parse_recipe_log(recipe_log_path)
    from transformers import AutoTokenizer

    with maybe_silence_output(quiet_model_io):
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=False)
    input_token_counts = count_prompt_tokens(tokenizer, prompts, enable_thinking, max_input_tokens)
    output_rows = build_output_rows(prompts, parsed_log["generated_texts"], tokenizer)

    generated_tokens = int(sum(row["generated_tokens"] for row in output_rows))
    input_tokens = int(sum(input_token_counts))
    prefill_time_s = parsed_log["prefill_time_s"]
    decode_step_times_s = parsed_log["decode_step_times_s"]
    measured_time_s = None
    if prefill_time_s is not None or decode_step_times_s:
        measured_time_s = (prefill_time_s or 0.0) + sum(decode_step_times_s)
    total_time_s = measured_time_s if measured_time_s and measured_time_s > 0 else float(run_info["wall_time_s"])
    tokens_per_second = generated_tokens / total_time_s if total_time_s > 0 and generated_tokens > 0 else 0.0
    mean_latency = total_time_s / generated_tokens if generated_tokens > 0 else None

    outputs_path = outputs_dir / f"{metrics_tag}_generations.jsonl"
    with outputs_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    profiler_enabled = bool(model_config.get("enable_profiler"))
    profiler_dir = None
    profiler_summary_path = None
    profiler_artifacts = None
    if profiler_enabled:
        profiler_dir = str(Path(run_info["recipe_case_dir"]) / "prof")
        profiler_artifacts = discover_profile_artifacts(Path(profiler_dir))
        profiler_summary_path = write_profiler_summary(
            profiler_artifacts,
            work_dir / "profiler_summary" / f"{metrics_tag}_summary.txt",
        )

    dense_npu_norm_fused = bool(custom_params.get("enable_npu_rmsnorm")) and bool(
        custom_params.get("enable_add_rmsnorm")
    )
    generation_text_sanity = run_generation_text_sanity_check(output_rows, max_new_tokens=max_new_tokens)
    summary = {
        "metrics_tag": metrics_tag,
        "model_path": model_id,
        "resolved_model_path": resolved_model_path,
        "model_source": model_source,
        "dtype": model_config.get("dtype", "bfloat16"),
        "device": "npu:0",
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "enable_thinking": enable_thinking,
        "thinking": enable_thinking,
        "optimization_mode": optimization_mode,
        "requested_attn_implementation": "recipes_default",
        "resolved_attn_implementation": "recipes_fused_infer_attention_score_v2",
        "recipe_exe_mode": model_config.get("exe_mode"),
        "recipe_root": str(recipe_root),
        "recipe_yaml_path": str(yaml_path),
        "recipe_case_dir": run_info["recipe_case_dir"],
        "recipe_log_path": run_info["recipe_log_path"],
        "recipe_stdout_path": run_info["stdout_path"],
        "recipe_stderr_path": run_info["stderr_path"],
        "rmsnorm_status": {
            "enabled": dense_npu_norm_fused,
            "implementation": "recipes_qwen_dense_npu_norm",
            "rmsnorm_fused": bool(custom_params.get("enable_npu_rmsnorm")),
            "add_rmsnorm_fused": bool(custom_params.get("enable_add_rmsnorm")),
            "rmsnorm_backend": (
                "torch_npu.npu_rms_norm"
                if custom_params.get("enable_npu_rmsnorm")
                else "torch.pow + torch.mean + torch.rsqrt + torch.mul"
            ),
            "add_rmsnorm_backend": (
                "torch_npu.npu_add_rms_norm"
                if custom_params.get("enable_add_rmsnorm")
                else "torch.add + torch.pow + torch.mean + torch.rsqrt + torch.mul"
            ),
        },
        "seed": 2026,
        "batch_size": int(scheduler_config.get("batch_size", len(prompts))),
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "num_prompts": len(prompts),
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "total_time_s": total_time_s,
        "wall_time_s": float(run_info["wall_time_s"]),
        "prefill_time_s": prefill_time_s,
        "decode_time_s": sum(decode_step_times_s) if decode_step_times_s else None,
        "time_to_first_token_s": prefill_time_s,
        "tokens_per_second": tokens_per_second,
        "decode_tokens_per_second": tokens_per_second,
        "mean_per_token_latency_s": mean_latency,
        "profiler_enabled": profiler_enabled,
        "profiler_dir": profiler_dir,
        "profiler_summary_path": profiler_summary_path,
        "profiler_artifacts": profiler_artifacts,
        "outputs_path": str(outputs_path),
        "generation_text_sanity": generation_text_sanity,
        "batch_metrics": [
            {
                "batch_index": 0,
                "prompt_ids": [item["id"] for item in prompts],
                "batch_size": len(prompts),
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "total_time_s": total_time_s,
                "tokens_per_second": tokens_per_second,
                "mean_per_token_latency_s": mean_latency,
                "generated_token_counts": [row["generated_tokens"] for row in output_rows],
            }
        ],
    }
    metrics_path = metrics_dir / f"{metrics_tag}_summary.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("指标文件 =", metrics_path)
    print("结果目录 =", run_info["recipe_case_dir"])
    return summary

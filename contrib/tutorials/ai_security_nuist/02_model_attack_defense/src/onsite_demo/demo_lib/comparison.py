from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .paths import ensure_dir, save_json


def _load_if_path(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _final_train_loss(log: dict[str, Any]) -> float | None:
    history = log.get("epoch_metrics", log.get("epochs_history", []))
    if history:
        return float(history[-1].get("train_loss", 0.0))
    losses = log.get("train_loss_by_epoch", [])
    if losses:
        return float(losses[-1])
    return None


def _row(eval_summary: dict[str, Any], train_log: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_type": eval_summary.get("trigger_type"),
        "clean_accuracy": eval_summary.get("clean_accuracy"),
        "attack_success_rate": eval_summary.get("attack_success_rate", eval_summary.get("asr")),
        "avg_target_confidence_on_triggered": eval_summary.get("avg_target_confidence_on_triggered"),
        "demo_final_train_loss": _final_train_loss(train_log),
        "checkpoint_path": eval_summary.get("checkpoint_path") or train_log.get("checkpoint_path"),
    }


def save_comparison_report(summary: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(Path(output_dir))
    rows = summary["comparison_rows"]
    report_path = output_dir / "comparison_report.md"
    lines = [
        "# MindSpore BadNets 现场演示对比报告",
        "",
        "## 结论说明",
        "",
        "- square 是基础白块触发器 baseline。",
        "- checkerboard 是改进触发器。",
        "- 正式结论以服务器完整训练结果为准。",
        "- 现场 demo 只验证流程可执行，不替代正式实验指标。",
        "",
        "## 对比表",
        "",
        "| trigger_type | clean_accuracy | attack_success_rate | avg_target_confidence_on_triggered | demo_final_train_loss | checkpoint_path |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {trigger_type} | {clean_accuracy} | {attack_success_rate} | "
            "{avg_target_confidence_on_triggered} | {demo_final_train_loss} | {checkpoint_path} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 课堂口径",
            "",
            "白块触发器先作为 baseline 展示攻击流程和评测结果；棋盘触发器随后作为 improved 版本展示。",
            "两种触发器使用同一个 demo subset，保证课堂对比公平。",
            "现场小样本训练只用于证明数据读取、触发器注入、模型加载、前向传播、反向传播、loss 更新、评测和 checkpoint 保存流程完整。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_comparison_summary(
    square_eval: dict[str, Any] | str | Path,
    checkerboard_eval: dict[str, Any] | str | Path,
    square_log: dict[str, Any] | str | Path,
    checkerboard_log: dict[str, Any] | str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = ensure_dir(Path(output_dir))
    square_eval_data = _load_if_path(square_eval)
    checker_eval_data = _load_if_path(checkerboard_eval)
    square_log_data = _load_if_path(square_log)
    checker_log_data = _load_if_path(checkerboard_log)

    rows = [_row(square_eval_data, square_log_data), _row(checker_eval_data, checker_log_data)]
    summary = {
        "comparison_rows": rows,
        "notes": [
            "square 是基础白块触发器 baseline",
            "checkerboard 是改进触发器",
            "正式结论以服务器完整训练结果为准",
            "现场 demo 只验证流程可执行",
        ],
    }
    save_json(summary, output_dir / "comparison_summary.json")

    csv_path = output_dir / "comparison_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = save_comparison_report(summary, output_dir)
    summary["comparison_summary_path"] = str(output_dir / "comparison_summary.json")
    summary["comparison_table_path"] = str(csv_path)
    summary["comparison_report_path"] = str(report_path)
    save_json(summary, output_dir / "comparison_summary.json")
    return summary

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .inference import apply_optional_trigger, chw_to_hwc, load_image_chw_01
from .subset import list_image_records


def _maybe_display_table(rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
        from IPython.display import display

        display(pd.DataFrame(rows))
    except Exception:
        if not rows:
            print("无数据可展示。")
            return
        headers = list(rows[0].keys())
        print(" | ".join(headers))
        print("-" * max(24, 3 * len(headers) + 12))
        for row in rows:
            print(" | ".join(str(row.get(header, "")) for header in headers))


def _show_image(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title)
    ax.axis("off")


def _label_name(label: int) -> str:
    return f"label {int(label):02d}"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def show_dataset_overview(subset_dir: str | Path, max_images: int = 12):
    """Show a grid of clean demo-subset images.

    Figure titles intentionally use English/label ids so cloud notebooks do not
    depend on Chinese fonts.
    """

    import matplotlib.pyplot as plt

    records = list_image_records(subset_dir)[: int(max_images)]
    if not records:
        print(f"没有可展示的图片：{subset_dir}")
        return None

    cols = min(6, len(records))
    rows = int(np.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.35))
    axes_array = np.asarray(axes).reshape(-1)

    for ax in axes_array:
        ax.axis("off")
    for index, (image_path, label) in enumerate(records):
        image = chw_to_hwc(load_image_chw_01(image_path))
        _show_image(axes_array[index], image, f"idx {index}\n{_label_name(label)}")

    fig.suptitle("GTSRB Demo Subset Samples", fontsize=13)
    plt.tight_layout()
    return fig


def show_trigger_comparison(image_path: str | Path):
    """Show clean / square / checkerboard images side by side."""

    import matplotlib.pyplot as plt

    clean_chw = load_image_chw_01(image_path)
    square_chw = apply_optional_trigger(clean_chw, trigger_type="square", trigger_size=4, alpha=0.8)
    checker_chw = apply_optional_trigger(clean_chw, trigger_type="checkerboard", trigger_size=4, alpha=0.8)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    _show_image(axes[0], chw_to_hwc(clean_chw), "Clean")
    _show_image(axes[1], chw_to_hwc(square_chw), "Square Trigger")
    _show_image(axes[2], chw_to_hwc(checker_chw), "Checkerboard Trigger")
    plt.tight_layout()
    return fig


def show_trigger_comparison_grid(image_paths: list[str | Path], max_images: int = 6):
    """Show multiple clean / square / checkerboard rows."""

    import matplotlib.pyplot as plt

    paths = [Path(path) for path in image_paths[: int(max_images)]]
    if not paths:
        print("没有可展示的触发器对比图片。")
        return None

    fig, axes = plt.subplots(len(paths), 3, figsize=(8.5, 2.5 * len(paths)))
    axes = np.asarray(axes).reshape(len(paths), 3)
    for row, image_path in enumerate(paths):
        clean_chw = load_image_chw_01(image_path)
        square_chw = apply_optional_trigger(clean_chw, trigger_type="square", trigger_size=4, alpha=0.8)
        checker_chw = apply_optional_trigger(clean_chw, trigger_type="checkerboard", trigger_size=4, alpha=0.8)
        _show_image(axes[row, 0], chw_to_hwc(clean_chw), "Clean")
        _show_image(axes[row, 1], chw_to_hwc(square_chw), "Square Trigger")
        _show_image(axes[row, 2], chw_to_hwc(checker_chw), "Checkerboard Trigger")
    plt.tight_layout()
    return fig


def show_stage_prediction(clean_result: dict[str, Any], triggered_result: dict[str, Any], stage_name: str):
    """Show clean and triggered predictions for a stage."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    short_stage = "Square" if "Square" in stage_name else "Checkerboard"
    _show_image(axes[0], clean_result["processed_image"], f"{short_stage} Clean\npred={clean_result['pred_label']}")
    _show_image(
        axes[1],
        triggered_result["processed_image"],
        f"{short_stage} Triggered\npred={triggered_result['pred_label']}",
    )
    plt.tight_layout()

    print(f"【{stage_name}】Clean Top5：")
    _maybe_display_table(clean_result.get("top5", []))
    print(f"【{stage_name}】Triggered Top5：")
    _maybe_display_table(triggered_result.get("top5", []))
    return fig


def plot_demo_training_curve(*log_json_paths: str | Path, labels: list[str] | None = None):
    """Plot one or more onsite demo training curves."""

    import matplotlib.pyplot as plt

    paths = [Path(path) for path in log_json_paths if path is not None]
    if len(paths) == 1 and isinstance(log_json_paths[0], (list, tuple)):
        paths = [Path(path) for path in log_json_paths[0]]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.9))
    metric_specs = [
        ("train_loss", "Train Loss"),
        ("clean_accuracy", "Clean Accuracy"),
        ("asr", "ASR"),
    ]
    for index, log_path in enumerate(paths):
        if not log_path.exists():
            print(f"训练日志不存在，跳过：{log_path}")
            continue
        data = _load_json(log_path)
        epoch_metrics = list(data.get("epoch_metrics", data.get("epochs_history", [])))
        if not epoch_metrics:
            continue
        label = labels[index] if labels and index < len(labels) else data.get("trigger_type", log_path.parent.name)
        x_values = [int(item["epoch"]) for item in epoch_metrics]
        for axis, (metric_key, title) in zip(axes, metric_specs):
            y_values = [item.get(metric_key) for item in epoch_metrics]
            if all(value is None for value in y_values):
                continue
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=str(label))
            axis.set_xlabel("Epoch")
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            if metric_key in {"clean_accuracy", "asr"}:
                axis.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Loss / Score")
    if any(axis.lines for axis in axes):
        axes[0].legend()
    fig.suptitle("Demo Training Curves", fontsize=13)
    plt.tight_layout()
    return fig


def plot_metric_bar_comparison(comparison_summary_json: str | Path):
    """Plot clean accuracy and ASR bar comparison."""

    import matplotlib.pyplot as plt

    data = _load_json(comparison_summary_json)
    rows = data.get("comparison_rows", data if isinstance(data, list) else [])
    if not rows:
        print("comparison summary 中没有可绘制的数据。")
        return None

    labels = [str(row.get("trigger_type", "")) for row in rows]
    clean_values = [float(row.get("clean_accuracy", 0.0)) for row in rows]
    asr_values = [float(row.get("attack_success_rate", 0.0)) for row in rows]
    x_values = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x_values - width / 2, clean_values, width, label="Clean Accuracy")
    ax.bar(x_values + width / 2, asr_values, width, label="ASR")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Clean Accuracy and ASR Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig


def show_final_summary_table(final_summary_json: str | Path):
    """Display the official server final summary."""

    path = Path(final_summary_json)
    data = _load_json(path)
    rows = data if isinstance(data, list) else data.get("experiments", [])
    filtered = []
    for item in rows:
        if item.get("experiment") in {"square_main", "checkerboard_main"}:
            filtered.append(
                {
                    "experiment": item.get("experiment"),
                    "trigger_type": item.get("trigger_type"),
                    "clean_accuracy": item.get("clean_accuracy"),
                    "ASR": item.get("asr", item.get("attack_success_rate")),
                    "best_epoch": item.get("best_epoch"),
                    "checkpoint": item.get("ckpt_path", item.get("checkpoint")),
                    "train_log_path": item.get("train_log_path"),
                    "console_log_path": item.get("console_log_path"),
                }
            )
    _maybe_display_table(filtered)
    return filtered


def show_comparison_table(comparison_summary_json: str | Path):
    """Display onsite comparison summary."""

    data = _load_json(comparison_summary_json)
    rows = data.get("comparison_rows", data if isinstance(data, list) else [])
    compact_rows = []
    for row in rows:
        compact_rows.append(
            {
                "trigger_type": row.get("trigger_type"),
                "clean_accuracy": f"{float(row.get('clean_accuracy', 0.0)):.4f}",
                "attack_success_rate": f"{float(row.get('attack_success_rate', 0.0)):.4f}",
                "avg_target_confidence": f"{float(row.get('avg_target_confidence_on_triggered', 0.0)):.4f}",
                "demo_final_train_loss": f"{float(row.get('demo_final_train_loss', 0.0)):.4f}",
            }
        )
    _maybe_display_table(compact_rows)
    return compact_rows

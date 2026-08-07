from __future__ import annotations

from typing import Any

import numpy as np


PALETTE = {
    "clean": "#4C78A8",
    "square": "#F58518",
    "checkerboard": "#54A24B",
    "highlight": "#E45756",
    "panel": "#F8F9FB",
    "border": "#D6DBE4",
    "ink": "#1F2933",
    "neutral": "#9AA5B1",
    "good": "#2E8B57",
    "warn": "#C83F49",
}

LIGHT_STRIP_SUMMARY_COLUMNS = [
    "experiment",
    "model_source",
    "suspicious_score",
    "demo_threshold",
    "threshold_source",
    "detected_triggered",
    "formal_detection_rate",
    "formal_fpr",
]


def _apply_demo_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#B8C0CC",
            "axes.labelcolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "grid.color": "#E7EBF0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _experiment_color(experiment: str) -> str:
    return PALETTE["square"] if "square" in str(experiment).lower() else PALETTE["checkerboard"]


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return numeric


def _format_float(value: Any, digits: int = 4, default: str = "n/a") -> str:
    numeric = _as_float(value)
    if numeric is None:
        return default
    return f"{numeric:.{digits}f}"


def _format_threshold(result: dict[str, Any]) -> str:
    if str(result.get("threshold_source", "")) == "not_calibrated":
        return "threshold: not calibrated"
    threshold = _as_float(result.get("demo_threshold"))
    if threshold is None:
        return "threshold: not calibrated"
    return f"threshold: {threshold:.4f}"


def _trigger_caption(result: dict[str, Any]) -> str:
    trigger = str(result.get("trigger_type", "unknown")).strip().lower()
    if trigger == "square":
        return "square"
    if trigger == "checkerboard":
        return "checkerboard"
    return trigger or "unknown"


def _candidate_caption(result: dict[str, Any]) -> str:
    candidate = str(
        result.get("relative_path")
        or result.get("candidate_image_name")
        or result.get("candidate_image")
        or "n/a"
    )
    candidate = candidate.replace("\\", "/")
    return candidate.split("/")[-1] if "/" in candidate else candidate


def _empty_state_figure(title: str, message: str, figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color=PALETTE["ink"])
    ax.set_title(title)
    return fig


def _table_figure(
    rows: list[dict[str, Any]],
    *,
    title: str,
    column_order: list[str] | None = None,
    figsize: tuple[float, float] = (12.0, 3.8),
):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.5, 0.5, "No rows available.", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        return fig

    columns = column_order or list(rows[0].keys())
    display_rows = []
    for row in rows:
        display_rows.append(
            [
                (
                    "True"
                    if row.get(column) is True
                    else "False"
                    if row.get(column) is False
                    else _format_float(row.get(column))
                    if isinstance(row.get(column), (float, np.floating))
                    else str(row.get(column, ""))
                )
                for column in columns
            ]
        )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=display_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.38)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])
        cell.set_linewidth(0.9)
        if row_index == 0:
            cell.set_facecolor("#EAF0F7")
            cell.set_text_props(weight="bold", color=PALETTE["ink"])
        else:
            cell.set_facecolor("white" if row_index % 2 else "#F9FBFD")
            cell.set_text_props(color=PALETTE["ink"])
    ax.set_title(title, pad=10)
    return fig


def show_strip_light_candidate_pair(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    clean_image = result.get("clean_candidate_image")
    triggered_image = result.get("triggered_candidate_image")
    if clean_image is None or triggered_image is None:
        return _empty_state_figure(
            "Light STRIP++ candidate pair",
            "Live light demo images are unavailable in this result payload.",
            (8.6, 3.4),
        )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1), constrained_layout=True)
    titles = [
        f"Clean candidate\ntrue={int(result.get('true_label', -1))}",
        f"Triggered candidate\ntrigger={result.get('trigger_type', 'unknown')}",
    ]
    images = [clean_image, triggered_image]
    for ax, title, image in zip(axes, titles, images):
        ax.imshow(np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"Light STRIP++ candidate image pair ({_trigger_caption(result)})",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def show_strip_light_blend_grid(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    perturbations = list(result.get("perturbation_images", []))
    if not perturbations:
        return _empty_state_figure(
            "Light STRIP++ blend samples",
            "No perturbation images are available in this result payload.",
            (9.6, 3.4),
        )

    top1_predictions = list(result.get("top1_predictions", []))
    target_probabilities = list(result.get("target_probabilities", []))
    tiles: list[tuple[str, np.ndarray]] = []
    for index, image in enumerate(perturbations[: min(6, len(perturbations))], start=1):
        top1_prediction = top1_predictions[index - 1] if index - 1 < len(top1_predictions) else "n/a"
        target_probability = target_probabilities[index - 1] if index - 1 < len(target_probabilities) else None
        probability_text = "n/a" if target_probability is None else f"{float(target_probability):.3f}"
        tiles.append(
            (
                f"Blend {index}\ntop1={top1_prediction} p(target)={probability_text}",
                np.asarray(image, dtype=np.float32),
            )
        )

    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.6, 3.3 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    for ax, (title, image) in zip(axes_array, tiles):
        ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")
    fig.suptitle(
        f"Light STRIP++ perturbation / blend examples ({_trigger_caption(result)})",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def show_strip_light_perturbation_grid(result: dict[str, Any]):
    return show_strip_light_blend_grid(result)


def build_strip_light_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "experiment": str(result.get("experiment", "n/a")),
            "model_source": str(result.get("model_source", "n/a")),
            "suspicious_score": _format_float(result.get("suspicious_score")),
            "demo_threshold": _format_float(result.get("demo_threshold")),
            "threshold_source": str(result.get("threshold_source", "n/a")),
            "detected_triggered": result.get("detected_triggered"),
            "formal_detection_rate": _format_float(result.get("formal_strip_detection_rate")),
            "formal_fpr": _format_float(result.get("formal_strip_fpr")),
        }
    ]


def show_strip_light_summary_table(result: dict[str, Any]):
    return _table_figure(
        build_strip_light_summary_rows(result),
        title="Light STRIP++ summary",
        column_order=LIGHT_STRIP_SUMMARY_COLUMNS,
        figsize=(11.8, 2.8),
    )
    return fig


def plot_strip_light_target_probability(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    target_probabilities = np.asarray(result.get("target_probabilities", []), dtype=np.float32)
    top1_predictions = list(result.get("top1_predictions", []))
    fig, ax = plt.subplots(figsize=(11.8, 4.0), constrained_layout=True)
    if target_probabilities.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No target-probability trace is available.", ha="center", va="center", fontsize=11)
        ax.set_title("Live light demo target probability")
        return fig

    x_values = np.arange(1, len(target_probabilities) + 1)
    ax.plot(x_values, target_probabilities, marker="o", linewidth=2.2, color=PALETTE["highlight"])
    ax.fill_between(x_values, target_probabilities, color=PALETTE["highlight"], alpha=0.14)
    if len(top1_predictions) == len(target_probabilities) and len(top1_predictions) <= 8:
        for x_value, probability, prediction in zip(x_values, target_probabilities, top1_predictions):
            ax.annotate(
                f"top1={prediction}",
                xy=(x_value, float(probability)),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_values)
    ax.set_xlabel("Perturbation index")
    ax.set_ylabel(f"P(target={int(result.get('target_label', 0))})")
    ax.set_title(f"Live light demo target probability ({_trigger_caption(result)})")
    ax.grid(axis="y")
    return fig


def show_strip_light_score_card(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    fig = plt.figure(figsize=(11.6, 3.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.15, wspace=0.18)
    card_axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    footer_ax = fig.add_subplot(grid[1, :])

    detected = result.get("detected_triggered")
    detected_text = "true" if detected is True else "false"
    detected_color = PALETTE["warn"] if detected is True else PALETTE["good"]
    card_specs = [
        ("Suspicious score", _format_float(result.get("suspicious_score")), PALETTE["highlight"]),
        ("Threshold", _format_float(result.get("demo_threshold")), PALETTE["clean"]),
        ("Detected", detected_text, detected_color),
        ("Threshold source", str(result.get("threshold_source", "n/a")), PALETTE["checkerboard"]),
    ]

    for ax, (label, value, accent) in zip(card_axes, card_specs):
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle(
                (0.02, 0.08),
                0.96,
                0.84,
                facecolor=PALETTE["panel"],
                edgecolor=accent,
                linewidth=2.0,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(0.08, 0.73, label, transform=ax.transAxes, fontsize=9.2, color=PALETTE["ink"], weight="bold")
        ax.text(0.08, 0.34, value, transform=ax.transAxes, fontsize=12.2, color=PALETTE["ink"])

    footer_ax.axis("off")
    footer_lines = [
        f"candidate: {_candidate_caption(result)}",
        f"experiment: {result.get('experiment', 'n/a')}",
        f"model_source: {result.get('model_source', 'n/a')}",
        f"formal STRIP det_rate={_format_float(result.get('formal_strip_detection_rate'))} | formal FPR={_format_float(result.get('formal_strip_fpr'))}",
    ]
    footer_ax.text(
        0.01,
        0.90,
        "\n".join(footer_lines),
        transform=footer_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.0,
        color=PALETTE["ink"],
    )
    fig.suptitle(f"Light STRIP++ score card ({_trigger_caption(result)})", fontsize=13, fontweight="bold")
    return fig


def show_formal_detection_table(rows: list[dict[str, Any]]):
    return _table_figure(
        rows,
        title="Formal server result table",
        column_order=[
            "experiment",
            "clean_accuracy",
            "asr",
            "strip_detection_rate",
            "strip_fpr",
            "strip_roc_auc",
            "strip_pr_auc",
            "nc_suspected_target",
            "nc_mad_anomaly_index",
            "detection_passed",
        ],
        figsize=(14.0, 4.2),
    )


def plot_formal_strip_metrics(rows: list[dict[str, Any]]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    if not rows:
        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "No formal STRIP rows are available.", ha="center", va="center")
        return fig

    metrics = ["strip_detection_rate", "strip_roc_auc", "strip_pr_auc"]
    experiments = [str(row["experiment"]) for row in rows]
    x_values = np.arange(len(metrics))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2), constrained_layout=True)

    for index, row in enumerate(rows[:2]):
        color = _experiment_color(str(row["experiment"]))
        values = [float(row[metric]) for metric in metrics]
        offset = -width / 2 if index == 0 else width / 2
        axes[0].bar(x_values + offset, values, width=width, color=color, label=str(row["experiment"]))
    axes[0].set_xticks(x_values)
    axes[0].set_xticklabels([metric.replace("strip_", "").replace("_", "\n") for metric in metrics])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Formal server result: STRIP++ metrics")
    axes[0].grid(axis="y")
    axes[0].legend()

    fpr_values = [float(row["strip_fpr"]) for row in rows[:2]]
    colors = [_experiment_color(experiment) for experiment in experiments[:2]]
    axes[1].bar(experiments[:2], fpr_values, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_ylabel("FPR")
    axes[1].set_title("Formal server result: STRIP++ FPR")
    axes[1].grid(axis="y")
    upper = max(0.01, max(fpr_values) * 1.8 + 0.001) if fpr_values else 0.01
    axes[1].set_ylim(0.0, upper)
    for index, value in enumerate(fpr_values):
        axes[1].text(index, value + upper * 0.03, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    return fig


def show_neural_cleanse_summary(rows: list[dict[str, Any]]):
    summary_rows = []
    for row in rows:
        summary_rows.append(
            {
                "experiment": row["experiment"],
                "nc_suspected_target": row["nc_suspected_target"],
                "nc_mad_anomaly_index": row["nc_mad_anomaly_index"],
                "detection_passed": row["detection_passed"],
            }
        )
    return _table_figure(
        summary_rows,
        title="Formal server result: Neural Cleanse summary",
        column_order=["experiment", "nc_suspected_target", "nc_mad_anomaly_index", "detection_passed"],
        figsize=(10.5, 3.6),
    )


def _neural_cleanse_rows(nc_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in ("square_main", "checkerboard_main"):
        value = nc_summary.get(experiment)
        if not isinstance(value, dict):
            continue
        row = dict(value)
        row["experiment"] = experiment
        if row.get("suspected_target") is None and row.get("suspected_target_class") is not None:
            row["suspected_target"] = row.get("suspected_target_class")
        if row.get("detection_passed") is None:
            row["detection_passed"] = bool(row.get("target_label_0_detected", False))
        rows.append(row)
    return rows


def _extract_class_norm_series(row: dict[str, Any]) -> tuple[list[int], list[float]]:
    for key in ("mask_norm_series", "class_mask_norms", "class_norms", "mask_norms", "per_class_mask_norms"):
        payload = row.get(key)
        if isinstance(payload, dict):
            pairs = []
            for raw_key, raw_value in payload.items():
                try:
                    pairs.append((int(raw_key), float(raw_value)))
                except (TypeError, ValueError):
                    continue
            if pairs:
                pairs.sort(key=lambda item: item[0])
                return [item[0] for item in pairs], [item[1] for item in pairs]
        if isinstance(payload, list) and payload:
            values = []
            for raw_value in payload:
                try:
                    values.append(float(raw_value))
                except (TypeError, ValueError):
                    values.append(np.nan)
            if any(np.isfinite(values)):
                return list(range(len(values))), values
    return [], []


def plot_neural_cleanse_mask_norms(nc_summary: dict[str, Any]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    rows = _neural_cleanse_rows(nc_summary)
    if not rows:
        return _empty_state_figure(
            "Formal Neural Cleanse mask norms",
            "Full class mask norm series not included; showing saved NC summary metrics is unavailable in this package.",
            (10.4, 3.8),
        )

    series_rows = []
    for row in rows:
        class_ids, class_values = _extract_class_norm_series(row)
        if class_ids and class_values:
            series_rows.append((row, class_ids, class_values))

    if series_rows:
        fig, axes = plt.subplots(1, len(series_rows), figsize=(6.2 * len(series_rows), 4.4), constrained_layout=True)
        axes_array = np.atleast_1d(axes)
        for ax, (row, class_ids, class_values) in zip(axes_array, series_rows):
            ax.plot(class_ids, class_values, color=_experiment_color(str(row.get("experiment", ""))), linewidth=2.0)
            ax.scatter(class_ids, class_values, color=PALETTE["ink"], s=18, zorder=3)
            suspected_target = row.get("suspected_target", row.get("suspected_target_class"))
            if suspected_target is not None:
                try:
                    target_index = class_ids.index(int(suspected_target))
                except (ValueError, TypeError):
                    target_index = None
                if target_index is not None:
                    ax.scatter(
                        [class_ids[target_index]],
                        [class_values[target_index]],
                        color=PALETTE["highlight"],
                        s=68,
                        zorder=4,
                        label=f"suspected target={int(suspected_target)}",
                    )
                    ax.legend(loc="upper right", fontsize=8.5)
            ax.set_title(str(row.get("experiment", "n/a")))
            ax.set_xlabel("Class index")
            ax.set_ylabel("Mask norm")
            ax.grid(axis="y")
        fig.suptitle("Formal Neural Cleanse class-wise mask norms", fontsize=14, fontweight="bold")
        return fig

    experiments = [str(row.get("experiment", "n/a")) for row in rows]
    anomaly_values = [float(row.get("mad_anomaly_index", 0.0) or 0.0) for row in rows]
    colors = [_experiment_color(experiment) for experiment in experiments]

    fig, ax = plt.subplots(figsize=(10.4, 4.4), constrained_layout=True)
    bars = ax.bar(experiments, anomaly_values, color=colors, edgecolor=PALETTE["border"], linewidth=1.0)
    ax.set_ylabel("MAD anomaly index")
    ax.set_xlabel("Experiment")
    ax.set_title("Formal Neural Cleanse saved summary metrics")
    ax.grid(axis="y")
    upper = max(anomaly_values) * 1.25 if max(anomaly_values) > 0 else 1.0
    ax.set_ylim(0.0, upper)
    for bar, row, value in zip(bars, rows, anomaly_values):
        suspected_target = row.get("suspected_target", row.get("suspected_target_class", "n/a"))
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + upper * 0.03,
            f"target={suspected_target}\nMAD={value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        0.5,
        -0.22,
        "Full class mask norm series not included; showing saved NC summary metrics.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color=PALETTE["ink"],
    )
    return fig


def plot_neural_cleanse_anomaly_comparison(nc_summary: dict[str, Any]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    rows = _neural_cleanse_rows(nc_summary)
    if not rows:
        return _empty_state_figure(
            "Formal Neural Cleanse anomaly comparison",
            "No Neural Cleanse anomaly summary is available.",
            (10.2, 3.8),
        )

    experiments = [str(row.get("experiment", "n/a")) for row in rows]
    anomaly_values = [float(row.get("mad_anomaly_index", 0.0) or 0.0) for row in rows]
    colors = [_experiment_color(experiment) for experiment in experiments]
    threshold = 2.0

    fig, ax = plt.subplots(figsize=(10.2, 4.5), constrained_layout=True)
    bars = ax.bar(experiments, anomaly_values, color=colors, edgecolor=PALETTE["border"], linewidth=1.0)
    ax.axhline(threshold, color=PALETTE["highlight"], linestyle="--", linewidth=1.8, label="threshold=2.0")
    ax.set_ylabel("MAD anomaly index")
    ax.set_xlabel("Experiment")
    ax.set_title("Formal Neural Cleanse MAD anomaly comparison")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    upper = max(max(anomaly_values), threshold) * 1.22 if max(anomaly_values + [threshold]) > 0 else 1.0
    ax.set_ylim(0.0, upper)
    for bar, row, value in zip(bars, rows, anomaly_values):
        suspected_target = row.get("suspected_target", row.get("suspected_target_class", "n/a"))
        detected = bool(row.get("detection_passed", row.get("target_label_0_detected", False)))
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + upper * 0.03,
            f"target={suspected_target}\npassed={detected}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return fig


def show_attack_detection_overview(rows: list[dict[str, Any]]):
    import matplotlib.pyplot as plt

    _apply_demo_style()
    fig = plt.figure(figsize=(13.2, 5.3), dpi=150)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.18, wspace=0.12)
    card_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis("off")

    for ax, row in zip(card_axes, rows[:2]):
        ax.axis("off")
        color = _experiment_color(str(row["experiment"]))
        ax.add_patch(
            plt.Rectangle(
                (0.02, 0.08),
                0.96,
                0.82,
                facecolor=PALETTE["panel"],
                edgecolor=color,
                linewidth=2.2,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(0.07, 0.80, str(row["experiment"]), transform=ax.transAxes, fontsize=12, fontweight="bold", color=PALETTE["ink"])
        ax.text(
            0.07,
            0.65,
            f"attack result: {row.get('attack_result', 'n/a')}",
            transform=ax.transAxes,
            fontsize=10.2,
            color=PALETTE["ink"],
        )
        ax.text(
            0.07,
            0.48,
            f"STRIP++ detection result: {row.get('strip_detection_result', 'n/a')}",
            transform=ax.transAxes,
            fontsize=10.2,
            color=PALETTE["ink"],
        )
        ax.text(
            0.07,
            0.31,
            f"Neural Cleanse result: {row.get('neural_cleanse_result', 'n/a')}",
            transform=ax.transAxes,
            fontsize=10.2,
            color=PALETTE["ink"],
        )
        ax.text(
            0.07,
            0.15,
            f"detection_passed: {bool(row.get('detection_passed', False))}",
            transform=ax.transAxes,
            fontsize=10.2,
            color=PALETTE["good"] if row.get("detection_passed") else PALETTE["warn"],
        )

    table_rows = [
        {
            "experiment": row.get("experiment"),
            "attack_result": row.get("attack_result"),
            "STRIP++ detection result": row.get("strip_detection_result"),
            "Neural Cleanse result": row.get("neural_cleanse_result"),
            "detection_passed": row.get("detection_passed"),
        }
        for row in rows
    ]
    columns = [
        "experiment",
        "attack_result",
        "STRIP++ detection result",
        "Neural Cleanse result",
        "detection_passed",
    ]
    display_rows = []
    for row in table_rows:
        display_rows.append(
            [
                "True" if row.get(column) is True else "False" if row.get(column) is False else str(row.get(column, ""))
                for column in columns
            ]
        )
    table = table_ax.table(
        cellText=display_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.16, 0.98, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])
        cell.set_linewidth(0.9)
        if row_index == 0:
            cell.set_facecolor("#EAF0F7")
            cell.set_text_props(weight="bold", color=PALETTE["ink"])
        else:
            cell.set_facecolor("white" if row_index % 2 else "#F9FBFD")
            cell.set_text_props(color=PALETTE["ink"])
    table_ax.set_title("Final attack + detection overview", pad=12)
    table_ax.text(
        0.5,
        0.04,
        "Detection passed; this result does not claim defense success.",
        transform=table_ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color=PALETTE["ink"],
    )
    fig.suptitle("Attack + detection final overview", fontsize=15, fontweight="bold", y=0.97)
    return fig


__all__ = [
    "build_strip_light_summary_rows",
    "plot_formal_strip_metrics",
    "plot_strip_light_target_probability",
    "show_attack_detection_overview",
    "show_formal_detection_table",
    "plot_neural_cleanse_anomaly_comparison",
    "plot_neural_cleanse_mask_norms",
    "show_neural_cleanse_summary",
    "show_strip_light_blend_grid",
    "show_strip_light_candidate_pair",
    "show_strip_light_perturbation_grid",
    "show_strip_light_score_card",
    "show_strip_light_summary_table",
]

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from .inference import (
    apply_optional_trigger,
    chw_to_hwc,
    load_image_chw_01,
    load_model_from_checkpoint,
    predict_probabilities,
)
from .paths import (
    ensure_dir,
    load_detection_summary,
    load_detection_table,
    load_final_summary,
    save_json,
)
from .subset import list_image_records, pick_first_non_target_image


FORMAL_METRIC_WARNING = (
    "Light STRIP++ demo is for process demonstration only. Formal detection metrics use protocol v3 server results."
)
SCORE_KEYS = (
    "entropy_score",
    "top1_consistency_score",
    "target_stability_score",
    "confidence_margin_score",
    "prediction_variance_score",
)


def load_formal_detection_bundle(
    final_summary: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    attack_summary = list(final_summary) if final_summary is not None else load_final_summary()
    detection_summary = load_detection_summary()
    detection_table = load_detection_table()
    detection_summary = _backfill_neural_cleanse_summary(detection_summary, detection_table)
    formal_rows = build_formal_detection_rows(attack_summary, detection_summary, detection_table)
    overview_rows = build_attack_detection_overview_rows(attack_summary, detection_summary, detection_table)
    return {
        "attack_summary": attack_summary,
        "detection_summary": detection_summary,
        "detection_table": detection_table,
        "formal_rows": formal_rows,
        "overview_rows": overview_rows,
    }


def _attack_lookup(final_summary: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in final_summary:
        experiment = str(item.get("experiment") or item.get("experiment_name") or "").strip()
        if experiment:
            lookup[experiment] = item
    return lookup


def _table_lookup(detection_table: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for row in detection_table:
        category = str(row.get("category", "")).strip()
        variant = str(row.get("variant", "")).strip()
        metric = str(row.get("metric", "")).strip()
        if not category or not variant or not metric:
            continue
        lookup.setdefault(category, {}).setdefault(variant, {})[metric] = row.get("metric_value")
    return lookup


def _pick_metric(
    table_lookup: dict[str, dict[str, dict[str, Any]]],
    category: str,
    variant: str,
    metric: str,
    fallback: Any,
) -> Any:
    category_lookup = table_lookup.get(category, {})
    variant_lookup = category_lookup.get(variant, {})
    if metric in variant_lookup:
        return variant_lookup[metric]
    return fallback


def _backfill_neural_cleanse_summary(
    detection_summary: dict[str, Any],
    detection_table: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = dict(detection_summary)
    metric_lookup = _table_lookup(detection_table)
    existing = dict(summary.get("neural_cleanse_full_43_class", {}))
    nc_summary: dict[str, Any] = {key: value for key, value in existing.items() if key == "acceptance_pass"}
    row_pass_flags: list[bool] = []

    for experiment in ("square_main", "checkerboard_main"):
        row = dict(existing.get(experiment, {}))
        row["experiment"] = experiment
        for metric in (
            "completed_class_count",
            "suspected_target_class",
            "target_label_0_detected",
            "mask_norm_class_0",
            "second_smallest_mask_norm",
            "median_mask_norm",
            "mad_anomaly_index",
            "anomaly_threshold",
            "reversed_success_rate",
        ):
            if metric in row:
                continue
            value = _pick_metric(metric_lookup, "neural_cleanse_full_43_class", experiment, metric, None)
            if value is not None:
                row[metric] = value
        if "suspected_target" not in row and row.get("suspected_target_class") is not None:
            row["suspected_target"] = row.get("suspected_target_class")
        if "detection_passed" not in row:
            row["detection_passed"] = bool(row.get("target_label_0_detected", False))
        if len(row) > 1:
            nc_summary[experiment] = row
            row_pass_flags.append(bool(row.get("target_label_0_detected", False)))

    if row_pass_flags and "acceptance_pass" not in nc_summary:
        nc_summary["acceptance_pass"] = all(row_pass_flags)
    summary["neural_cleanse_full_43_class"] = nc_summary
    return summary


def build_formal_detection_rows(
    final_summary: list[dict[str, Any]] | None = None,
    detection_summary: dict[str, Any] | None = None,
    detection_table: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    attack_summary = list(final_summary) if final_summary is not None else load_final_summary()
    summary = dict(detection_summary) if detection_summary is not None else load_detection_summary()
    table = list(detection_table) if detection_table is not None else load_detection_table()
    attack_lookup = _attack_lookup(attack_summary)
    metric_lookup = _table_lookup(table)
    strip_summary = dict(summary.get("strip_protocol_v3", {}))
    nc_summary = dict(summary.get("neural_cleanse_full_43_class", {}))

    rows: list[dict[str, Any]] = []
    for experiment in ("square_main", "checkerboard_main"):
        attack = dict(attack_lookup.get(experiment, {}))
        strip_item = dict(strip_summary.get(experiment, {}))
        nc_item = dict(nc_summary.get(experiment, {}))
        detection_passed = bool(
            strip_summary.get("acceptance_pass", False)
            and nc_summary.get("acceptance_pass", False)
            and nc_item.get("target_label_0_detected", False)
        )
        rows.append(
            {
                "experiment": experiment,
                "clean_accuracy": attack.get("clean_accuracy"),
                "asr": attack.get("asr", attack.get("attack_success_rate")),
                "strip_detection_rate": _pick_metric(
                    metric_lookup,
                    "strip_protocol_v3",
                    experiment,
                    "detection_rate",
                    strip_item.get("detection_rate"),
                ),
                "strip_fpr": _pick_metric(
                    metric_lookup,
                    "strip_protocol_v3",
                    experiment,
                    "FPR",
                    strip_item.get("FPR"),
                ),
                "strip_roc_auc": _pick_metric(
                    metric_lookup,
                    "strip_protocol_v3",
                    experiment,
                    "ROC_AUC",
                    strip_item.get("ROC_AUC"),
                ),
                "strip_pr_auc": _pick_metric(
                    metric_lookup,
                    "strip_protocol_v3",
                    experiment,
                    "PR_AUC",
                    strip_item.get("PR_AUC"),
                ),
                "nc_suspected_target": _pick_metric(
                    metric_lookup,
                    "neural_cleanse_full_43_class",
                    experiment,
                    "suspected_target_class",
                    nc_item.get("suspected_target_class"),
                ),
                "nc_mad_anomaly_index": _pick_metric(
                    metric_lookup,
                    "neural_cleanse_full_43_class",
                    experiment,
                    "mad_anomaly_index",
                    nc_item.get("mad_anomaly_index"),
                ),
                "detection_passed": detection_passed,
            }
        )
    return rows


def build_attack_detection_overview_rows(
    final_summary: list[dict[str, Any]],
    detection_summary: dict[str, Any],
    detection_table: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detection_rows = build_formal_detection_rows(final_summary, detection_summary, detection_table)
    overview_rows: list[dict[str, Any]] = []
    for row in detection_rows:
        overview_rows.append(
            {
                "experiment": row["experiment"],
                "attack_result": f"clean={float(row['clean_accuracy']):.4f}, asr={float(row['asr']):.4f}",
                "strip_detection_result": (
                    f"det_rate={float(row['strip_detection_rate']):.4f}, "
                    f"fpr={float(row['strip_fpr']):.4f}"
                ),
                "neural_cleanse_result": (
                    f"target={int(row['nc_suspected_target'])}, "
                    f"mad={float(row['nc_mad_anomaly_index']):.4f}"
                ),
                "detection_passed": bool(row["detection_passed"]),
                "note": "Detection passed; this result does not claim defense success.",
            }
        )
    return overview_rows


def _image_record_lookup(split_dir: Path) -> dict[Path, int]:
    return {path.resolve(): int(label) for path, label in list_image_records(split_dir)}


def _infer_label_from_path(image_path: Path) -> int | None:
    for part in reversed(image_path.parts):
        if str(part).isdigit():
            return int(part)
    return None


def collect_reference_images(
    subset_test_dir: str | Path,
    target_label: int,
    count: int,
    seed: int = 42,
    exclude_paths: set[str] | set[Path] | None = None,
) -> list[dict[str, Any]]:
    subset_dir = Path(subset_test_dir)
    excluded = {str(Path(path).resolve()) for path in (exclude_paths or set())}
    records = [
        (path, label)
        for path, label in list_image_records(subset_dir)
        if int(label) != int(target_label) and str(path.resolve()) not in excluded
    ]
    if not records:
        raise ValueError(f"No clean non-target reference images are available in {subset_dir}.")

    rng = random.Random(int(seed))
    if len(records) >= int(count):
        selected = rng.sample(records, int(count))
    else:
        selected = list(records)
        while len(selected) < int(count):
            selected.append(rng.choice(records))

    return [
        {
            "path": str(path),
            "label": int(label),
            "image_chw": load_image_chw_01(path),
        }
        for path, label in selected
    ]


def make_strip_blends(
    candidate_image_chw: np.ndarray,
    reference_images: list[dict[str, Any]],
    k: int,
    blend_alpha: float = 0.5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if not reference_images:
        raise ValueError("At least one reference image is required to build STRIP blends.")

    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    for index in range(int(k)):
        reference = reference_images[index] if index < len(reference_images) else rng.choice(reference_images)
        reference_image = np.asarray(reference["image_chw"], dtype=np.float32)
        blend = np.clip(
            float(blend_alpha) * np.asarray(candidate_image_chw, dtype=np.float32)
            + (1.0 - float(blend_alpha)) * reference_image,
            0.0,
            1.0,
        ).astype(np.float32)
        rows.append(
            {
                "blend_image_chw": blend,
                "reference_path": str(reference["path"]),
                "reference_label": int(reference["label"]),
            }
        )
    return rows


def compute_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    clipped = np.clip(values, 1e-8, 1.0)
    return (-np.sum(clipped * np.log(clipped), axis=1)).astype(np.float32)


def compute_candidate_score(probabilities: np.ndarray, target_label: int) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 2 or probs.size == 0:
        raise ValueError("Candidate probabilities must be a non-empty 2D array.")

    top1_predictions = probs.argmax(axis=1).astype(np.int32)
    sorted_probs = np.sort(probs, axis=1)
    top1_scores = sorted_probs[:, -1]
    top2_scores = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros(len(probs), dtype=np.float32)
    confidence_margins = (top1_scores - top2_scores).astype(np.float32)
    target_probabilities = probs[:, int(target_label)].astype(np.float32)
    entropy_values = compute_entropy(probs)
    class_variance = float(np.mean(np.var(probs, axis=0)))
    entropy_bound = math.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
    counts = np.bincount(top1_predictions, minlength=probs.shape[1])

    component_scores = {
        "entropy_score": float(max(0.0, min(1.0, 1.0 - float(np.mean(entropy_values)) / max(entropy_bound, 1e-8)))),
        "top1_consistency_score": float(np.max(counts) / max(1, len(top1_predictions))),
        "target_stability_score": float(np.mean(target_probabilities)),
        "confidence_margin_score": float(np.mean(confidence_margins)),
        "prediction_variance_score": float(1.0 / (1.0 + 50.0 * max(class_variance, 0.0))),
    }
    suspicious_score = float(np.mean([component_scores[key] for key in SCORE_KEYS]))

    perturbation_rows = []
    for index in range(len(top1_predictions)):
        perturbation_rows.append(
            {
                "index": index + 1,
                "top1": int(top1_predictions[index]),
                "target_probability": float(target_probabilities[index]),
                "entropy": float(entropy_values[index]),
                "confidence_margin": float(confidence_margins[index]),
            }
        )

    return {
        "top1_predictions": [int(value) for value in top1_predictions.tolist()],
        "target_probabilities": [float(value) for value in target_probabilities.tolist()],
        "entropy_values": [float(value) for value in entropy_values.tolist()],
        "confidence_margin_values": [float(value) for value in confidence_margins.tolist()],
        "component_scores": component_scores,
        "suspicious_score": suspicious_score,
        "perturbation_rows": perturbation_rows,
    }


def calibrate_light_threshold(
    model,
    subset_test_dir: str | Path,
    target_label: int,
    k: int,
    reference_count: int,
    calibration_count: int,
    target_fpr: float,
    batch_size: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    subset_dir = Path(subset_test_dir)
    candidates = [
        (path, label)
        for path, label in list_image_records(subset_dir)
        if int(label) != int(target_label)
    ]
    if len(candidates) < 3:
        return {
            "threshold_source": "not_calibrated",
            "demo_threshold": None,
            "detected_triggered": None,
            "calibration_scores": [],
            "calibration_rows": [],
            "calibration_sample_count": len(candidates),
        }

    rng = random.Random(int(seed))
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected = shuffled[: min(int(calibration_count), len(shuffled))]
    if len(selected) < 3:
        return {
            "threshold_source": "not_calibrated",
            "demo_threshold": None,
            "detected_triggered": None,
            "calibration_scores": [],
            "calibration_rows": [],
            "calibration_sample_count": len(selected),
        }

    calibration_rows: list[dict[str, Any]] = []
    calibration_scores: list[float] = []
    for index, (path, label) in enumerate(selected):
        clean_candidate = load_image_chw_01(path)
        references = collect_reference_images(
            subset_dir,
            target_label=target_label,
            count=int(reference_count),
            seed=int(seed + index + 1),
            exclude_paths={path},
        )
        blends = make_strip_blends(
            clean_candidate,
            references,
            k=int(k),
            seed=int(seed + 100 + index),
        )
        probabilities = predict_probabilities(
            model,
            np.stack([item["blend_image_chw"] for item in blends]).astype(np.float32),
            batch_size=batch_size,
        )
        score_bundle = compute_candidate_score(probabilities, target_label=target_label)
        calibration_rows.append(
            {
                "candidate_image": str(path),
                "true_label": int(label),
                "suspicious_score": float(score_bundle["suspicious_score"]),
                **{key: float(score_bundle["component_scores"][key]) for key in SCORE_KEYS},
            }
        )
        calibration_scores.append(float(score_bundle["suspicious_score"]))

    if len(calibration_scores) < 3:
        return {
            "threshold_source": "not_calibrated",
            "demo_threshold": None,
            "detected_triggered": None,
            "calibration_scores": calibration_scores,
            "calibration_rows": calibration_rows,
            "calibration_sample_count": len(calibration_scores),
        }

    demo_threshold = float(np.quantile(np.asarray(calibration_scores, dtype=np.float32), 1.0 - float(target_fpr)))
    if not math.isfinite(demo_threshold):
        return {
            "threshold_source": "not_calibrated",
            "demo_threshold": None,
            "detected_triggered": None,
            "calibration_scores": calibration_scores,
            "calibration_rows": calibration_rows,
            "calibration_sample_count": len(calibration_scores),
        }

    return {
        "threshold_source": "local_demo_clean_calibration",
        "demo_threshold": demo_threshold,
        "detected_triggered": None,
        "calibration_scores": calibration_scores,
        "calibration_rows": calibration_rows,
        "calibration_sample_count": len(calibration_scores),
    }


def run_strip_light_demo(
    model,
    subset_test_dir: str | Path,
    trigger_type: str,
    target_label: int,
    k: int,
    reference_count: int = 8,
    calibration_count: int = 12,
    target_fpr: float = 0.10,
    trigger_size: int = 4,
    alpha: float = 0.8,
    position: str = "bottom_right",
    batch_size: int = 1,
    seed: int = 42,
    candidate_image_path: str | Path | None = None,
    model_source: str | None = None,
) -> dict[str, Any]:
    subset_dir = Path(subset_test_dir)
    candidate_path = Path(candidate_image_path) if candidate_image_path is not None else Path(
        pick_first_non_target_image(subset_dir, target_label=target_label)
    )
    label_lookup = _image_record_lookup(subset_dir)
    true_label = int(label_lookup.get(candidate_path.resolve(), -1))
    if true_label < 0:
        inferred_label = _infer_label_from_path(candidate_path)
        true_label = int(inferred_label) if inferred_label is not None else -1
    if true_label < 0:
        raise ValueError(f"Could not infer the true label for candidate image: {candidate_path}")

    clean_candidate = load_image_chw_01(candidate_path)
    triggered_candidate = apply_optional_trigger(
        clean_candidate,
        trigger_type=trigger_type,
        trigger_size=trigger_size,
        alpha=alpha,
        position=position,
    )
    references = collect_reference_images(
        subset_dir,
        target_label=target_label,
        count=int(reference_count),
        seed=int(seed),
        exclude_paths={candidate_path},
    )
    blends = make_strip_blends(
        triggered_candidate,
        references,
        k=int(k),
        seed=int(seed + 1),
    )
    blend_batch = np.stack([item["blend_image_chw"] for item in blends]).astype(np.float32)
    probabilities = predict_probabilities(model, blend_batch, batch_size=batch_size)
    score_bundle = compute_candidate_score(probabilities, target_label=target_label)
    calibration = calibrate_light_threshold(
        model=model,
        subset_test_dir=subset_dir,
        target_label=target_label,
        k=int(k),
        reference_count=int(reference_count),
        calibration_count=int(calibration_count),
        target_fpr=float(target_fpr),
        batch_size=batch_size,
        seed=int(seed + 10),
    )

    if str(calibration.get("threshold_source")) != "local_demo_clean_calibration":
        raise RuntimeError(
            "Light STRIP++ demo threshold must come from local_demo_clean_calibration. "
            f"Got {calibration.get('threshold_source')!r} for candidate {candidate_path}."
        )

    demo_threshold = calibration["demo_threshold"]
    if demo_threshold is None:
        raise RuntimeError(
            "Light STRIP++ demo threshold calibration returned no threshold. "
            f"Candidate image: {candidate_path}."
        )

    detected_triggered: bool | None = bool(float(score_bundle["suspicious_score"]) >= float(demo_threshold))

    experiment = "square_main" if str(trigger_type).lower().strip() == "square" else "checkerboard_main"
    formal_bundle = load_formal_detection_bundle()
    formal_row = next(
        (row for row in formal_bundle["formal_rows"] if row["experiment"] == experiment),
        {},
    )

    return {
        "experiment": experiment,
        "trigger_type": str(trigger_type),
        "k": int(k),
        "target_label": int(target_label),
        "batch_size": int(batch_size),
        "candidate_image": str(candidate_path),
        "candidate_image_name": candidate_path.name,
        "true_label": true_label,
        "target_probabilities": score_bundle["target_probabilities"],
        "entropy_values": score_bundle["entropy_values"],
        "top1_predictions": score_bundle["top1_predictions"],
        "confidence_margin_values": score_bundle["confidence_margin_values"],
        "component_scores": score_bundle["component_scores"],
        "suspicious_score": float(score_bundle["suspicious_score"]),
        "demo_threshold": float(demo_threshold) if demo_threshold is not None else None,
        "threshold_source": calibration["threshold_source"],
        "detected_triggered": detected_triggered,
        "formal_metric_warning": FORMAL_METRIC_WARNING,
        "model_checkpoint_path": str(getattr(model, "demo_checkpoint_path", "")),
        "model_source": model_source,
        "reference_count": int(reference_count),
        "calibration_sample_count": int(calibration["calibration_sample_count"]),
        "calibration_scores": [float(value) for value in calibration["calibration_scores"]],
        "perturbation_rows": score_bundle["perturbation_rows"],
        "reference_paths": [str(item["path"]) for item in references],
        "reference_labels": [int(item["label"]) for item in references],
        "blend_reference_paths": [row["reference_path"] for row in blends],
        "clean_candidate_image": chw_to_hwc(clean_candidate),
        "triggered_candidate_image": chw_to_hwc(triggered_candidate),
        "reference_images": [chw_to_hwc(item["image_chw"]) for item in references[: min(4, len(references))]],
        "perturbation_images": [chw_to_hwc(row["blend_image_chw"]) for row in blends],
        "formal_strip_detection_rate": formal_row.get("strip_detection_rate"),
        "formal_strip_fpr": formal_row.get("strip_fpr"),
        "formal_strip_roc_auc": formal_row.get("strip_roc_auc"),
        "formal_strip_pr_auc": formal_row.get("strip_pr_auc"),
        "formal_nc_suspected_target": formal_row.get("nc_suspected_target"),
        "formal_nc_mad_anomaly_index": formal_row.get("nc_mad_anomaly_index"),
        "detection_passed": formal_row.get("detection_passed"),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_strip_light_result(result: dict[str, Any], output_path: str | Path) -> Path:
    skip_keys = {"clean_candidate_image", "triggered_candidate_image", "reference_images", "perturbation_images"}
    payload = {
        key: _json_ready(value)
        for key, value in result.items()
        if key not in skip_keys
    }
    path = Path(output_path)
    ensure_dir(path.parent)
    save_json(payload, path)
    return path


__all__ = [
    "FORMAL_METRIC_WARNING",
    "build_attack_detection_overview_rows",
    "build_formal_detection_rows",
    "calibrate_light_threshold",
    "collect_reference_images",
    "compute_candidate_score",
    "compute_entropy",
    "load_formal_detection_bundle",
    "make_strip_blends",
    "run_strip_light_demo",
    "save_strip_light_result",
]

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .inference import apply_optional_trigger, load_image_chw_01, predict_probabilities
from .paths import save_json
from .subset import list_image_records


def evaluate_small_subset(
    model,
    subset_test_dir: str | Path,
    trigger_type: str,
    target_label: int,
    max_samples: int | None = None,
    trigger_size: int = 4,
    alpha: float = 0.8,
    position: str = "bottom_right",
    batch_size: int = 32,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate clean accuracy and ASR on the same small demo subset."""

    records = list_image_records(subset_test_dir)
    if max_samples is not None:
        records = records[: int(max_samples)]
    if not records:
        raise ValueError(f"demo test subset 为空，无法评测：{subset_test_dir}")

    clean_images = np.stack([load_image_chw_01(path) for path, _ in records]).astype(np.float32)
    clean_labels = np.asarray([label for _, label in records], dtype=np.int32)
    clean_probs = predict_probabilities(model, clean_images, batch_size=batch_size)
    clean_preds = clean_probs.argmax(axis=1).astype(np.int32)
    clean_accuracy = float((clean_preds == clean_labels).mean()) if len(clean_labels) else 0.0

    triggered_images: list[np.ndarray] = []
    for image, label in zip(clean_images, clean_labels):
        if int(label) == int(target_label):
            continue
        triggered_images.append(
            apply_optional_trigger(
                image,
                trigger_type=trigger_type,
                trigger_size=trigger_size,
                alpha=alpha,
                position=position,
            )
        )

    if triggered_images:
        triggered_batch = np.stack(triggered_images).astype(np.float32)
        triggered_probs = predict_probabilities(model, triggered_batch, batch_size=batch_size)
        triggered_preds = triggered_probs.argmax(axis=1).astype(np.int32)
        attack_success_rate = float((triggered_preds == int(target_label)).mean())
        avg_target_confidence = float(triggered_probs[:, int(target_label)].mean())
        triggered_count = int(len(triggered_preds))
    else:
        attack_success_rate = 0.0
        avg_target_confidence = 0.0
        triggered_count = 0

    summary = {
        "trigger_type": str(trigger_type),
        "clean_accuracy": clean_accuracy,
        "attack_success_rate": attack_success_rate,
        "asr": attack_success_rate,
        "target_label": int(target_label),
        "avg_target_confidence_on_triggered": avg_target_confidence,
        "num_clean_eval_samples": int(len(clean_labels)),
        "num_triggered_eval_samples": triggered_count,
        "checkpoint_path": str(getattr(model, "demo_checkpoint_path", "")),
        "checkpoint_loaded": bool(getattr(model, "demo_checkpoint_loaded", False)),
        "metric_source": "live_demo_subset",
    }
    if output_path is not None:
        save_json(summary, Path(output_path))
    return summary

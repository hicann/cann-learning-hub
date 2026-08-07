from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("GLOG_v", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import matplotlib

matplotlib.use("Agg")


ONSITE_ROOT = Path(__file__).resolve().parents[1]
if str(ONSITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ONSITE_ROOT))

from demo_lib.detection import run_strip_light_demo, save_strip_light_result
from demo_lib.detection_visualization import show_strip_light_perturbation_grid
from demo_lib.inference import load_model_from_checkpoint
from demo_lib.paths import (
    ensure_dir,
    find_checkpoint,
    load_demo_config,
    resolve_project_root,
)
from demo_lib.subset import create_demo_subset, list_image_records, pick_first_non_target_image


ALLOWED_IMAGE_SOURCES = {"data_test", "demo_subset_test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the onsite light STRIP++ detection demo")
    parser.add_argument("--mode", default="fast", choices=["fast", "enhanced"])
    parser.add_argument("--trigger-type", default="square", choices=["square", "checkerboard"])
    parser.add_argument("--k", type=int, default=4, choices=[4, 6, 8, 10, 16])
    parser.add_argument("--image-source", default="data_test", choices=sorted(ALLOWED_IMAGE_SOURCES))
    parser.add_argument("--device-target", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="live_or_official", choices=["live_or_official"])
    return parser.parse_args()


def _mode_config(config: dict[str, Any], mode: str) -> dict[str, int]:
    modes = dict(config.get("modes", {}))
    if mode not in modes:
        raise ValueError(f"Unknown demo mode: {mode}")
    return {key: int(value) for key, value in modes[mode].items()}


def _stage_config(config: dict[str, Any], trigger_type: str) -> dict[str, Any]:
    return config["square"] if trigger_type == "square" else config["checkerboard"]


def _source_dir(paths: dict[str, Path], image_source: str) -> Path:
    if image_source == "data_test":
        return paths["TEST_DIR"]
    if image_source == "demo_subset_test":
        return paths["DEMO_SUBSET_ROOT"] / "test"
    raise ValueError(f"Unsupported detection image source: {image_source}")


def _latest_live_checkpoint(paths: dict[str, Path], trigger_type: str) -> Path | None:
    stage_dir_name = "square_baseline" if trigger_type == "square" else "checkerboard_improved"
    candidates = sorted(
        paths["DEMO_RUNS_ROOT"].glob(f"**/{stage_dir_name}/demo_last.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_checkpoint(
    *,
    paths: dict[str, Path],
    stage_config: dict[str, Any],
    trigger_type: str,
    checkpoint_mode: str,
) -> tuple[Path, str]:
    if checkpoint_mode != "live_or_official":
        raise ValueError(f"Unsupported checkpoint mode: {checkpoint_mode}")

    live_ckpt = _latest_live_checkpoint(paths, trigger_type)
    if live_ckpt is not None and live_ckpt.exists():
        return live_ckpt, "live_trained_demo_checkpoint"
    return find_checkpoint(stage_config["experiment_name"]), "official_checkpoint_fallback"


def _pick_candidate_image(source_dir: Path, target_label: int) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Requested image source directory is missing: {source_dir}")
    return Path(pick_first_non_target_image(source_dir, target_label=target_label))


def _validate_result_payload(result: dict[str, Any], k: int) -> None:
    target_probabilities = list(result.get("target_probabilities", []))
    if len(target_probabilities) != int(k):
        raise ValueError(f"Expected {k} target probabilities, got {len(target_probabilities)}")
    suspicious_score = float(result.get("suspicious_score"))
    if not math.isfinite(suspicious_score):
        raise ValueError("Suspicious score must be a finite number.")
    if str(result.get("threshold_source")) == "zero_fallback":
        raise ValueError("zero_fallback is not allowed for light STRIP++ thresholding.")
    if result.get("demo_threshold") == 0:
        raise ValueError("demo_threshold must not silently fall back to 0.")
    detected = result.get("detected_triggered")
    if detected not in (True, False, None):
        raise ValueError(f"detected_triggered must be bool or None, got {detected!r}")
    if result.get("model_source") not in {"live_trained_demo_checkpoint", "official_checkpoint_fallback"}:
        raise ValueError(f"Unexpected model_source: {result.get('model_source')!r}")


def main() -> dict[str, Any]:
    args = parse_args()
    paths = resolve_project_root()
    config = load_demo_config()
    detection_config = dict(config["detection"])
    stage_config = _stage_config(config, args.trigger_type)
    mode_config = _mode_config(config, args.mode)
    target_label = int(config["target_label"])

    subset_manifest = paths["DEMO_SUBSET_ROOT"] / "subset_manifest.json"
    if not subset_manifest.exists():
        create_demo_subset(
            train_dir=paths["TRAIN_DIR"],
            test_dir=paths["TEST_DIR"],
            output_dir=paths["DEMO_SUBSET_ROOT"],
            train_per_class=mode_config["train_per_class"],
            test_per_class=mode_config["test_per_class"],
            seed=int(args.seed),
        )

    checkpoint_path, model_source = _resolve_checkpoint(
        paths=paths,
        stage_config=stage_config,
        trigger_type=args.trigger_type,
        checkpoint_mode=args.checkpoint,
    )
    model = load_model_from_checkpoint(
        experiment_name=stage_config["experiment_name"],
        checkpoint_path=checkpoint_path,
        num_classes=43,
        norm_type="group",
        device_target=args.device_target,
        ms_mode=str(config["ms_mode"]),
    )
    actual_device = str(getattr(model, "demo_device_target", args.device_target))
    candidate_image_path = _pick_candidate_image(_source_dir(paths, args.image_source), target_label=target_label)

    result = run_strip_light_demo(
        model=model,
        subset_test_dir=paths["DEMO_SUBSET_ROOT"] / "test",
        trigger_type=args.trigger_type,
        target_label=target_label,
        k=int(args.k),
        reference_count=int(detection_config["strip_light_reference_count"]),
        calibration_count=int(detection_config["strip_light_calibration_count"]),
        target_fpr=float(detection_config["strip_light_target_fpr"]),
        trigger_size=int(stage_config.get("trigger_size", 4)),
        alpha=float(stage_config.get("alpha", 0.8)),
        position=str(stage_config.get("position", "bottom_right")),
        batch_size=int(detection_config.get("strip_light_batch_size", 1)),
        seed=int(args.seed),
        candidate_image_path=candidate_image_path,
        model_source=model_source,
    )
    result["mode"] = args.mode
    result["device_target"] = actual_device
    result["checkpoint_path"] = str(checkpoint_path)
    result["image_source"] = args.image_source

    output_dir = ensure_dir(paths["OUTPUT_ROOT"] / "detection_smoke")
    result_path = output_dir / f"{args.trigger_type}_strip_light_result.json"
    preview_path = output_dir / f"{args.trigger_type}_strip_light_preview.png"
    save_strip_light_result(result, result_path)
    preview_figure = show_strip_light_perturbation_grid(result)
    preview_figure.savefig(preview_path, dpi=160, bbox_inches="tight", facecolor="white")

    _validate_result_payload(result, k=int(args.k))
    payload = {
        "result_json_exists": result_path.exists(),
        "preview_png_exists": preview_path.exists(),
        "result_json_path": str(result_path),
        "preview_png_path": str(preview_path),
        "trigger_type": args.trigger_type,
        "image_source": args.image_source,
        "k": int(args.k),
        "candidate_image": str(candidate_image_path),
        "subset_test_record_count": len(list_image_records(paths["DEMO_SUBSET_ROOT"] / "test")),
        "model_source": model_source,
        "checkpoint_path": str(checkpoint_path),
        "device_target": actual_device,
    }
    print(f"result json = {result_path}")
    print(f"preview png = {preview_path}")
    print(f"model_source = {model_source}")
    print(f"device_target = {actual_device}")
    return payload


if __name__ == "__main__":
    main()

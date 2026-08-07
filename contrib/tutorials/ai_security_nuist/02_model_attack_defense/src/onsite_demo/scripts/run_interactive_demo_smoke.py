from __future__ import annotations

import argparse
import json
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

from demo_lib.inference import load_model_from_checkpoint
from demo_lib.interactive_demo import (
    run_random_attack_test,
    run_single_trigger_attack_test,
    show_single_trigger_attack_result,
)
from demo_lib.paths import ensure_dir, find_checkpoint, load_demo_config, resolve_project_root, save_json
from demo_lib.subset import create_demo_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for onsite single-trigger attack demo")
    parser.add_argument("--mode", default="enhanced", choices=["fast", "enhanced"])
    parser.add_argument(
        "--image-source",
        default="data_test",
        choices=["data_test", "data_train", "demo_subset_test", "demo_subset_train"],
    )
    parser.add_argument("--trigger-type", default="square", choices=["square", "checkerboard"])
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--random-pick", action="store_true")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--include-target-label", action="store_true")
    parser.add_argument("--device-target", default=None)
    return parser.parse_args()


def _mode_config(config: dict[str, Any], mode: str) -> dict[str, int]:
    modes = config.get("modes", {})
    if mode not in modes:
        raise ValueError(f"未知 DEMO_MODE={mode}，可选：{sorted(modes)}")
    return {key: int(value) for key, value in modes[mode].items()}


def _jsonable_result(result: dict[str, Any]) -> dict[str, Any]:
    skip_keys = {"clean_image", "triggered_image"}
    return {key: value for key, value in result.items() if key not in skip_keys}


def _stage_config(config: dict[str, Any], trigger_type: str) -> dict[str, Any]:
    return config["square"] if trigger_type == "square" else config["checkerboard"]


def main() -> dict[str, Any]:
    args = parse_args()
    paths = resolve_project_root()
    config = load_demo_config()
    mode_config = _mode_config(config, args.mode)
    target_label = int(config["target_label"])
    output_dir = ensure_dir(paths["OUTPUT_ROOT"] / "interactive_demo")

    manifest_path = paths["DEMO_SUBSET_ROOT"] / "subset_manifest.json"
    if not manifest_path.exists():
        create_demo_subset(
            train_dir=paths["TRAIN_DIR"],
            test_dir=paths["TEST_DIR"],
            output_dir=paths["DEMO_SUBSET_ROOT"],
            train_per_class=mode_config["train_per_class"],
            test_per_class=mode_config["test_per_class"],
            seed=int(config["seed"]),
        )

    stage_config = _stage_config(config, args.trigger_type)
    ckpt_path = find_checkpoint(stage_config["experiment_name"])
    device_target = args.device_target or str(config.get("device_target", "auto"))
    model = load_model_from_checkpoint(
        experiment_name=stage_config["experiment_name"],
        checkpoint_path=ckpt_path,
        num_classes=43,
        norm_type="group",
        device_target=device_target,
        ms_mode=str(config["ms_mode"]),
    )
    actual_device = str(getattr(model, "demo_device_target", device_target))

    if bool(args.random_pick):
        result = run_random_attack_test(
            image_source=args.image_source,
            trigger_type=args.trigger_type,
            model=model,
            target_label=target_label,
            project_root=paths["PROJECT_ROOT"],
            seed=args.random_seed,
            skip_target_label=not bool(args.include_target_label),
            trigger_size=int(stage_config.get("trigger_size", 4)),
            alpha=float(stage_config.get("alpha", 0.8)),
        )
    else:
        result = run_single_trigger_attack_test(
            image_source=args.image_source,
            image_index=int(args.image_index),
            trigger_type=args.trigger_type,
            model=model,
            target_label=target_label,
            project_root=paths["PROJECT_ROOT"],
            skip_target_label_warning=True,
            trigger_size=int(stage_config.get("trigger_size", 4)),
            alpha=float(stage_config.get("alpha", 0.8)),
        )
        result["random_pick"] = False
        result["random_seed"] = args.random_seed

    preview_path = output_dir / "attack_demo_preview.png"
    show_single_trigger_attack_result(result, save_path=preview_path)

    payload = {
        "mode": args.mode,
        "device_target": actual_device,
        "checkpoint": str(ckpt_path),
        "image_source": args.image_source,
        "image_index": int(result["image_index"]),
        "random_pick": bool(result.get("random_pick", False)),
        "trigger_type": args.trigger_type,
        "true_label": int(result["true_label"]),
        "clean_prediction": int(result["clean_prediction"]),
        "triggered_prediction": int(result["triggered_prediction"]),
        "target_label": int(result["target_label"]),
        "triggered_target_confidence": float(result["triggered_target_confidence"]),
        "attack_success": bool(result["attack_success"]),
        "warning_message": str(result.get("warning_message", "")),
        "result": _jsonable_result(result),
        "preview_path": str(preview_path),
    }
    save_json(payload, output_dir / "attack_demo_result.json")
    print(f"interactive smoke result：{output_dir / 'attack_demo_result.json'}")
    print(f"interactive smoke preview：{preview_path}")
    print(f"trigger_type：{args.trigger_type}")
    print(f"attack success：{payload['attack_success']}")
    return payload


if __name__ == "__main__":
    main()

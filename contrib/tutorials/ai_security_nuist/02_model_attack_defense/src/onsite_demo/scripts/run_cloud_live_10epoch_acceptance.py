from __future__ import annotations

import argparse
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

from demo_lib.detection import load_formal_detection_bundle, run_strip_light_demo, save_strip_light_result
from demo_lib.inference import load_model_from_checkpoint
from demo_lib.interactive_demo import run_random_attack_test, show_single_trigger_attack_result
from demo_lib.paths import (
    ensure_dir,
    load_demo_config,
    load_final_summary,
    make_run_timestamp,
    resolve_demo_mode_settings,
    resolve_project_root,
    save_json,
)
from demo_lib.runtime import configure_mindspore_device
from demo_lib.subset import create_demo_subset
from demo_lib.training import run_stage_pipeline


BANNED_CLAIMS = (
    "\u9632\u5fa1\u6210\u529f",
    "\u6a21\u578b\u5df2\u4fee\u590d",
    "\u540e\u95e8\u5df2\u6d88\u9664",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cloud_live 10-epoch onsite acceptance flow.")
    parser.add_argument("--mode", default="cloud_live")
    parser.add_argument("--device-target", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    return parser.parse_args()


def _jsonable_attack_result(result: dict[str, Any]) -> dict[str, Any]:
    skip_keys = {"clean_image", "triggered_image"}
    return {key: value for key, value in result.items() if key not in skip_keys}


def _line_has_bad_claim(line: str) -> bool:
    text = str(line).strip()
    if not text:
        return False
    for phrase in BANNED_CLAIMS:
        if phrase not in text:
            continue
        if "不代表" in text or "不意味着" in text or "不是" in text or "不声称" in text:
            return False
        return True
    return False


def _no_bad_claims(paths: dict[str, Path]) -> bool:
    notebook_text = "\n".join(
        (paths["PROJECT_ROOT"] / name).read_text(encoding="utf-8")
        for name in ("01.01_badnets_attack.ipynb", "01.02_neural_cleanse_detection.ipynb")
    )
    readme_text = (paths["ONSITE_DEMO_ROOT"] / "README_ONSITE_DEMO.md").read_text(encoding="utf-8")
    return not any(_line_has_bad_claim(line) for line in (notebook_text + "\n" + readme_text).splitlines())


def _run_live_attack_smoke(
    *,
    project_root: Path,
    stage_result: dict[str, Any],
    stage_config: dict[str, Any],
    output_dir: Path,
    target_label: int,
    seed: int,
    device_target: str,
    ms_mode: str,
) -> dict[str, Any]:
    model = load_model_from_checkpoint(
        experiment_name=stage_config["experiment_name"],
        checkpoint_path=stage_result["checkpoint_path"],
        num_classes=43,
        norm_type="group",
        device_target=device_target,
        ms_mode=ms_mode,
    )
    result = run_random_attack_test(
        image_source="demo_subset_test",
        trigger_type=stage_config["trigger_type"],
        model=model,
        target_label=target_label,
        project_root=project_root,
        seed=seed,
        skip_target_label=True,
        trigger_size=int(stage_config.get("trigger_size", 4)),
        alpha=float(stage_config.get("alpha", 0.8)),
    )
    preview_path = output_dir / "live_attack_smoke.png"
    show_single_trigger_attack_result(result, save_path=preview_path)
    payload = {
        "trigger_type": stage_config["trigger_type"],
        "image_source": "demo_subset_test",
        "model_source": stage_result["model_source"],
        "checkpoint_path": stage_result["checkpoint_path"],
        "demo_checkpoint_path": stage_result["train_log"]["demo_checkpoint_path"],
        "attack_success": bool(result["attack_success"]),
        "preview_path": str(preview_path),
        "result": _jsonable_attack_result(result),
    }
    save_json(payload, output_dir / "live_attack_smoke.json")
    return payload


def _run_live_detection_smoke(
    *,
    stage_result: dict[str, Any],
    stage_config: dict[str, Any],
    subset_test_dir: Path,
    output_dir: Path,
    target_label: int,
    detection_config: dict[str, Any],
    detection_k: int,
    seed: int,
    device_target: str,
    ms_mode: str,
) -> dict[str, Any]:
    model = load_model_from_checkpoint(
        experiment_name=stage_config["experiment_name"],
        checkpoint_path=stage_result["checkpoint_path"],
        num_classes=43,
        norm_type="group",
        device_target=device_target,
        ms_mode=ms_mode,
    )
    result = run_strip_light_demo(
        model=model,
        subset_test_dir=subset_test_dir,
        trigger_type=stage_config["trigger_type"],
        target_label=target_label,
        k=int(detection_k),
        reference_count=int(detection_config["strip_light_reference_count"]),
        calibration_count=int(detection_config["strip_light_calibration_count"]),
        target_fpr=float(detection_config["strip_light_target_fpr"]),
        trigger_size=int(stage_config.get("trigger_size", 4)),
        alpha=float(stage_config.get("alpha", 0.8)),
        position=str(stage_config.get("position", "bottom_right")),
        batch_size=int(detection_config.get("strip_light_batch_size", 1)),
        seed=seed,
        candidate_image_path=stage_result["clean_result"]["image_path"],
        model_source=stage_result["model_source"],
    )
    result_path = output_dir / "light_strip_detection.json"
    save_strip_light_result(result, result_path)
    return {
        "result_path": str(result_path),
        "result": result,
    }


def main() -> dict[str, Any]:
    args = parse_args()
    paths = resolve_project_root()
    config = load_demo_config()
    mode_settings = resolve_demo_mode_settings(config, args.mode)
    detection_config = dict(config["detection"])
    summary_rows = load_final_summary()

    if int(mode_settings["epochs"]) < 10:
        raise ValueError(f"cloud_live mode must default to >= 10 epochs, got {mode_settings['epochs']}")

    print("1. Configure MindSpore")
    actual_device = configure_mindspore_device(args.device_target, str(config["ms_mode"]))
    print(f"device_target = {actual_device}")
    print(f"mode = {mode_settings['mode']}")

    acceptance_root = ensure_dir(paths["OUTPUT_ROOT"] / "cloud_live_acceptance" / make_run_timestamp())
    subset_root = ensure_dir(acceptance_root / "demo_subset")

    print("2. Create demo subset")
    manifest = create_demo_subset(
        train_dir=paths["TRAIN_DIR"],
        test_dir=paths["TEST_DIR"],
        output_dir=subset_root,
        train_per_class=mode_settings["train_per_class"],
        test_per_class=mode_settings["test_per_class"],
        seed=args.seed,
    )
    print(f"demo subset train/test totals = {manifest['train_total']} / {manifest['test_total']}")

    print("3. Train square stage")
    square = run_stage_pipeline(
        stage_config=config["square"],
        run_root=acceptance_root,
        subset_root=subset_root,
        target_label=int(config["target_label"]),
        epochs=mode_settings["epochs"],
        batch_size=mode_settings["batch_size"],
        device_target=actual_device,
        ms_mode=str(config["ms_mode"]),
        seed=int(args.seed),
        max_steps_per_epoch=args.max_steps_per_epoch,
        eval_each_epoch=mode_settings["eval_each_epoch"],
        save_epoch_metrics=mode_settings["save_epoch_metrics"],
        save_training_curve=mode_settings["save_training_curve"],
        prefer_live_trained_checkpoint=mode_settings["prefer_live_trained_checkpoint"],
    )

    print("4. Train checkerboard stage")
    checkerboard = run_stage_pipeline(
        stage_config=config["checkerboard"],
        run_root=acceptance_root,
        subset_root=subset_root,
        target_label=int(config["target_label"]),
        epochs=mode_settings["epochs"],
        batch_size=mode_settings["batch_size"],
        device_target=actual_device,
        ms_mode=str(config["ms_mode"]),
        seed=int(args.seed) + 1,
        max_steps_per_epoch=args.max_steps_per_epoch,
        eval_each_epoch=mode_settings["eval_each_epoch"],
        save_epoch_metrics=mode_settings["save_epoch_metrics"],
        save_training_curve=mode_settings["save_training_curve"],
        prefer_live_trained_checkpoint=mode_settings["prefer_live_trained_checkpoint"],
    )

    print("5. Run live attack smoke")
    square_attack = _run_live_attack_smoke(
        project_root=paths["PROJECT_ROOT"],
        stage_result=square,
        stage_config=config["square"],
        output_dir=Path(square["output_dir"]),
        target_label=int(config["target_label"]),
        seed=int(args.seed),
        device_target=actual_device,
        ms_mode=str(config["ms_mode"]),
    )
    checkerboard_attack = _run_live_attack_smoke(
        project_root=paths["PROJECT_ROOT"],
        stage_result=checkerboard,
        stage_config=config["checkerboard"],
        output_dir=Path(checkerboard["output_dir"]),
        target_label=int(config["target_label"]),
        seed=int(args.seed) + 1,
        device_target=actual_device,
        ms_mode=str(config["ms_mode"]),
    )

    print("6. Run live light STRIP++ detection smoke")
    square_detection = _run_live_detection_smoke(
        stage_result=square,
        stage_config=config["square"],
        subset_test_dir=subset_root / "test",
        output_dir=Path(square["output_dir"]),
        target_label=int(config["target_label"]),
        detection_config=detection_config,
        detection_k=int(mode_settings["strip_light_k"]),
        seed=int(args.seed),
        device_target=actual_device,
        ms_mode=str(config["ms_mode"]),
    )
    checkerboard_detection = _run_live_detection_smoke(
        stage_result=checkerboard,
        stage_config=config["checkerboard"],
        subset_test_dir=subset_root / "test",
        output_dir=Path(checkerboard["output_dir"]),
        target_label=int(config["target_label"]),
        detection_config=detection_config,
        detection_k=int(mode_settings["strip_light_k"]),
        seed=int(args.seed) + 1,
        device_target=actual_device,
        ms_mode=str(config["ms_mode"]),
    )

    print("7. Load formal detection results")
    formal_bundle = load_formal_detection_bundle(summary_rows)
    formal_lookup = {row["experiment"]: row for row in formal_bundle["formal_rows"]}
    square_formal = dict(formal_lookup["square_main"])
    checkerboard_formal = dict(formal_lookup["checkerboard_main"])

    summary = {
        "acceptance_root": str(acceptance_root),
        "mode": mode_settings["mode"],
        "square_epochs_completed": int(square["train_log"]["epochs_completed"]),
        "checkerboard_epochs_completed": int(checkerboard["train_log"]["epochs_completed"]),
        "square_training_curve_exists": Path(square["training_curve_path"]).exists(),
        "checkerboard_training_curve_exists": Path(checkerboard["training_curve_path"]).exists(),
        "square_checkpoint_exists": Path(square["train_log"]["demo_checkpoint_path"]).exists(),
        "checkerboard_checkpoint_exists": Path(checkerboard["train_log"]["demo_checkpoint_path"]).exists(),
        "square_attack_demo_uses_live_checkpoint": square_attack["model_source"] == "live_trained_demo_checkpoint",
        "checkerboard_attack_demo_uses_live_checkpoint": (
            checkerboard_attack["model_source"] == "live_trained_demo_checkpoint"
        ),
        "square_detection_uses_live_checkpoint": (
            square_detection["result"].get("model_source") == "live_trained_demo_checkpoint"
        ),
        "checkerboard_detection_uses_live_checkpoint": (
            checkerboard_detection["result"].get("model_source") == "live_trained_demo_checkpoint"
        ),
        "formal_detection_loaded": True,
        "formal_strip_metrics_loaded": True,
        "neural_cleanse_results_loaded": True,
        "square_formal_strip_detection_rate": square_formal["strip_detection_rate"],
        "checkerboard_formal_strip_detection_rate": checkerboard_formal["strip_detection_rate"],
        "square_formal_nc_target": square_formal["nc_suspected_target"],
        "checkerboard_formal_nc_target": checkerboard_formal["nc_suspected_target"],
        "no_bad_claims": _no_bad_claims(paths),
    }
    summary["cloud_live_10epoch_acceptance_pass"] = all(
        [
            summary["square_epochs_completed"] >= 10,
            summary["checkerboard_epochs_completed"] >= 10,
            summary["square_training_curve_exists"],
            summary["checkerboard_training_curve_exists"],
            summary["square_checkpoint_exists"],
            summary["checkerboard_checkpoint_exists"],
            summary["square_attack_demo_uses_live_checkpoint"],
            summary["checkerboard_attack_demo_uses_live_checkpoint"],
            summary["square_detection_uses_live_checkpoint"],
            summary["checkerboard_detection_uses_live_checkpoint"],
            summary["formal_detection_loaded"],
            summary["no_bad_claims"],
        ]
    )
    summary_path = acceptance_root / "cloud_live_acceptance_summary.json"
    save_json(summary, summary_path)

    print(f"square stage dir = {square['output_dir']}")
    print(f"checkerboard stage dir = {checkerboard['output_dir']}")
    print(f"acceptance summary = {summary_path}")
    print(f"cloud_live_10epoch_acceptance_pass = {summary['cloud_live_10epoch_acceptance_pass']}")
    return summary


if __name__ == "__main__":
    main()

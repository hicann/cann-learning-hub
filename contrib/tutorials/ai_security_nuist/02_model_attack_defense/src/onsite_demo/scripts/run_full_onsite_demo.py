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

from demo_lib.comparison import build_comparison_summary
from demo_lib.detection import load_formal_detection_bundle, run_strip_light_demo, save_strip_light_result
from demo_lib.inference import load_model_from_checkpoint
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
from demo_lib.visualization import plot_demo_training_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full onsite attack + light detection demo.")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--device-target", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    return parser.parse_args()


def _save_training_curve(run_root: Path, square_stage: dict[str, Any], checkerboard_stage: dict[str, Any]) -> Path:
    import matplotlib.pyplot as plt

    figure = plot_demo_training_curve(
        square_stage["demo_train_log_path"],
        checkerboard_stage["demo_train_log_path"],
        labels=["square", "checkerboard"],
    )
    curve_path = ensure_dir(run_root / "comparison") / "demo_training_curve.png"
    figure.savefig(curve_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return curve_path


def _run_stage_detection(
    *,
    model_checkpoint: Path,
    model_source: str,
    stage_result: dict[str, Any],
    stage_config: dict[str, Any],
    detection_config: dict[str, Any],
    target_label: int,
    device_target: str,
    ms_mode: str,
    seed: int,
    subset_test_dir: Path,
    output_dir: Path,
    detection_k: int,
) -> tuple[dict[str, Any], Path]:
    model = load_model_from_checkpoint(
        experiment_name=stage_config["experiment_name"],
        checkpoint_path=model_checkpoint,
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
        model_source=model_source,
    )
    result_path = output_dir / f"{stage_config['trigger_type']}_strip_light_result.json"
    save_strip_light_result(result, result_path)
    return result, result_path


def main() -> dict[str, Any]:
    args = parse_args()
    paths = resolve_project_root()
    config = load_demo_config()
    mode_settings = resolve_demo_mode_settings(
        config,
        args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    detection_config = dict(config["detection"])

    print("1. Configure MindSpore")
    actual_device = configure_mindspore_device(args.device_target, str(config["ms_mode"]))
    print(f"PROJECT_ROOT = {paths['PROJECT_ROOT']}")
    print(f"device_target = {actual_device}")
    print(f"mode = {mode_settings['mode']}")
    print(f"epochs = {mode_settings['epochs']}")
    print(f"batch_size = {mode_settings['batch_size']}")

    print("2. Load formal attack summary")
    summary = load_final_summary()
    print(f"formal attack rows = {len(summary)}")

    run_root = ensure_dir(paths["DEMO_RUNS_ROOT"] / make_run_timestamp())
    subset_root = ensure_dir(run_root / "demo_subset")

    print("3. Create demo subset")
    manifest = create_demo_subset(
        train_dir=paths["TRAIN_DIR"],
        test_dir=paths["TEST_DIR"],
        output_dir=subset_root,
        train_per_class=mode_settings["train_per_class"],
        test_per_class=mode_settings["test_per_class"],
        seed=args.seed,
    )
    print(f"demo subset train/test totals = {manifest['train_total']} / {manifest['test_total']}")

    print("4. Run square stage")
    square = run_stage_pipeline(
        stage_config=config["square"],
        run_root=run_root,
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

    print("5. Run checkerboard stage")
    checkerboard = run_stage_pipeline(
        stage_config=config["checkerboard"],
        run_root=run_root,
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

    print("6. Build comparison summary")
    comparison_dir = ensure_dir(run_root / "comparison")
    comparison = build_comparison_summary(
        square_eval=square["eval_summary"],
        checkerboard_eval=checkerboard["eval_summary"],
        square_log=square["train_log"],
        checkerboard_log=checkerboard["train_log"],
        output_dir=comparison_dir,
    )
    training_curve_path = _save_training_curve(run_root, square, checkerboard)

    detection_dir = ensure_dir(run_root / "detection")

    print("7. Run square light STRIP++ detection")
    square_detection_result, square_detection_result_path = _run_stage_detection(
        model_checkpoint=Path(square["checkpoint_path"]),
        model_source=str(square["model_source"]),
        stage_result=square,
        stage_config=config["square"],
        detection_config=detection_config,
        target_label=int(config["target_label"]),
        device_target=actual_device,
        ms_mode=str(config["ms_mode"]),
        seed=int(args.seed),
        subset_test_dir=subset_root / "test",
        output_dir=detection_dir,
        detection_k=mode_settings["strip_light_k"],
    )

    print("8. Run checkerboard light STRIP++ detection")
    checkerboard_detection_result, checkerboard_detection_result_path = _run_stage_detection(
        model_checkpoint=Path(checkerboard["checkpoint_path"]),
        model_source=str(checkerboard["model_source"]),
        stage_result=checkerboard,
        stage_config=config["checkerboard"],
        detection_config=detection_config,
        target_label=int(config["target_label"]),
        device_target=actual_device,
        ms_mode=str(config["ms_mode"]),
        seed=int(args.seed) + 1,
        subset_test_dir=subset_root / "test",
        output_dir=detection_dir,
        detection_k=mode_settings["strip_light_k"],
    )

    print("9. Load formal detection summary")
    formal_bundle = load_formal_detection_bundle(summary)
    formal_lookup = {row["experiment"]: row for row in formal_bundle["formal_rows"]}
    square_formal = dict(formal_lookup["square_main"])
    checkerboard_formal = dict(formal_lookup["checkerboard_main"])

    detection_summary = {
        "mode": mode_settings["mode"],
        "epochs": mode_settings["epochs"],
        "batch_size": mode_settings["batch_size"],
        "train_per_class": mode_settings["train_per_class"],
        "test_per_class": mode_settings["test_per_class"],
        "square_light_strip_result_path": str(square_detection_result_path),
        "checkerboard_light_strip_result_path": str(checkerboard_detection_result_path),
        "square_model_source": square["model_source"],
        "checkerboard_model_source": checkerboard["model_source"],
        "square_model_source_reason": square["model_source_reason"],
        "checkerboard_model_source_reason": checkerboard["model_source_reason"],
        "square_demo_checkpoint_path": square["train_log"]["demo_checkpoint_path"],
        "checkerboard_demo_checkpoint_path": checkerboard["train_log"]["demo_checkpoint_path"],
        "square_epochs_completed": int(square["train_log"]["epochs_completed"]),
        "checkerboard_epochs_completed": int(checkerboard["train_log"]["epochs_completed"]),
        "square_training_curve_exists": Path(square["training_curve_path"]).exists(),
        "checkerboard_training_curve_exists": Path(checkerboard["training_curve_path"]).exists(),
        "square_checkpoint_exists": Path(square["train_log"]["demo_checkpoint_path"]).exists(),
        "checkerboard_checkpoint_exists": Path(checkerboard["train_log"]["demo_checkpoint_path"]).exists(),
        "square_attack_demo_uses_live_checkpoint": square["model_source"] == "live_trained_demo_checkpoint",
        "checkerboard_attack_demo_uses_live_checkpoint": checkerboard["model_source"] == "live_trained_demo_checkpoint",
        "square_detection_uses_live_checkpoint": square_detection_result.get("model_source") == "live_trained_demo_checkpoint",
        "checkerboard_detection_uses_live_checkpoint": (
            checkerboard_detection_result.get("model_source") == "live_trained_demo_checkpoint"
        ),
        "formal_detection_loaded": True,
        "formal_strip_metrics_loaded": True,
        "neural_cleanse_results_loaded": True,
        "formal_square_detection_rate": square_formal["strip_detection_rate"],
        "formal_square_fpr": square_formal["strip_fpr"],
        "formal_checkerboard_detection_rate": checkerboard_formal["strip_detection_rate"],
        "formal_checkerboard_fpr": checkerboard_formal["strip_fpr"],
        "neural_cleanse_square_target": square_formal["nc_suspected_target"],
        "neural_cleanse_checkerboard_target": checkerboard_formal["nc_suspected_target"],
        "square_live_suspicious_score": square_detection_result["suspicious_score"],
        "checkerboard_live_suspicious_score": checkerboard_detection_result["suspicious_score"],
        "square_live_threshold_source": square_detection_result["threshold_source"],
        "checkerboard_live_threshold_source": checkerboard_detection_result["threshold_source"],
        "comparison_summary_path": comparison["comparison_summary_path"],
        "training_curve_path": str(training_curve_path),
        "subset_manifest_path": str(subset_root / "subset_manifest.json"),
        "note": (
            "Live attack and light STRIP++ use the onsite-trained demo checkpoint when it exists. "
            "Formal conclusions still come from saved server artifacts."
        ),
    }
    detection_summary_path = run_root / "detection_summary.json"
    save_json(detection_summary, detection_summary_path)

    print(f"square clean accuracy = {square['eval_summary']['clean_accuracy']:.4f}")
    print(f"square ASR = {square['eval_summary']['attack_success_rate']:.4f}")
    print(f"checkerboard clean accuracy = {checkerboard['eval_summary']['clean_accuracy']:.4f}")
    print(f"checkerboard ASR = {checkerboard['eval_summary']['attack_success_rate']:.4f}")
    print(f"comparison_report.md = {comparison['comparison_report_path']}")
    print(f"training_curve.png = {training_curve_path}")
    print(f"detection_summary.json = {detection_summary_path}")

    return {
        "run_root": str(run_root),
        "mode_settings": mode_settings,
        "square": square,
        "checkerboard": checkerboard,
        "comparison": comparison,
        "training_curve_path": str(training_curve_path),
        "detection_summary_path": str(detection_summary_path),
        "square_detection_result_path": str(square_detection_result_path),
        "checkerboard_detection_result_path": str(checkerboard_detection_result_path),
    }


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(str(exc))
        raise SystemExit(2)

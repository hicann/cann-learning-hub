from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("GLOG_v", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ONSITE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ONSITE_ROOT.parents[1]
for path in (PROJECT_ROOT, ONSITE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run notebook Exp5 light STRIP++ live detection in an isolated process.")
    parser.add_argument("--trigger-type", default="square", choices=["square", "checkerboard"])
    parser.add_argument("--image-source", default="data_test", choices=["data_test", "demo_subset", "demo_subset_test"])
    parser.add_argument("--random-pick", type=_bool_arg, default=True)
    parser.add_argument("--image-index", type=int, default=1)
    parser.add_argument("--skip-target-label", type=_bool_arg, default=True)
    parser.add_argument("--k", type=int, default=8, choices=[4, 6, 8, 10, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", default="")
    parser.add_argument("--run-profile", default="cloud_live", choices=["fast", "enhanced", "cloud_live"])
    parser.add_argument("--device-target", default="Ascend")
    parser.add_argument("--ms-mode", default="PYNATIVE", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--subset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _stage_dir_name(trigger_type: str) -> str:
    return "square_baseline" if trigger_type == "square" else "checkerboard_improved"


def _stage_config(config: dict[str, Any], trigger_type: str) -> dict[str, Any]:
    return config["square"] if trigger_type == "square" else config["checkerboard"]


def _normalize_image_source(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "demo_subset_test":
        return "demo_subset"
    return normalized


def main() -> None:
    args = parse_args()
    from demo_lib.detection import save_strip_light_result
    from demo_lib.interactive_demo import run_free_detection_test
    from demo_lib.notebook_bootstrap import _latest_demo_checkpoint
    from demo_lib.paths import (
        ensure_dir,
        find_checkpoint,
        load_demo_config,
        resolve_demo_mode_settings,
        resolve_project_root,
    )
    from demo_lib.subset import create_demo_subset
    from demo_lib.inference import load_model_from_checkpoint

    paths = resolve_project_root()
    config = load_demo_config()
    mode_settings = resolve_demo_mode_settings(config, args.run_profile)
    target_label = int(config["target_label"])
    trigger = str(args.trigger_type).strip().lower()
    stage_cfg = _stage_config(config, trigger)
    subset_root = ensure_dir(Path(args.subset_root))
    subset_test_dir = subset_root / "test"

    if not subset_test_dir.exists():
        create_demo_subset(
            train_dir=paths["TRAIN_DIR"],
            test_dir=paths["TEST_DIR"],
            output_dir=subset_root,
            train_per_class=int(mode_settings["train_per_class"]),
            test_per_class=int(mode_settings["test_per_class"]),
            seed=int(args.seed),
        )

    fallback_ckpt = find_checkpoint(stage_cfg["experiment_name"])
    live_ckpt = _latest_demo_checkpoint(paths, _stage_dir_name(trigger))
    if live_ckpt is not None and bool(mode_settings["prefer_live_trained_checkpoint"]):
        checkpoint_path = live_ckpt
        model_source = "live_trained_demo_checkpoint"
    else:
        checkpoint_path = fallback_ckpt
        model_source = "official_checkpoint_fallback"

    model = load_model_from_checkpoint(
        experiment_name=stage_cfg["experiment_name"],
        checkpoint_path=checkpoint_path,
        num_classes=43,
        norm_type="group",
        device_target=str(args.device_target),
        ms_mode=str(args.ms_mode),
        allow_cpu_fallback=False,
        pool_type="reduce_mean",
    )
    actual_device = str(getattr(model, "demo_device_target", args.device_target))

    random_seed = None if str(args.random_seed).strip() == "" else int(args.random_seed)
    result = run_free_detection_test(
        project_root=paths["PROJECT_ROOT"],
        subset_test_dir=subset_test_dir,
        detection_model_bundles={
            trigger: {
                "model": model,
                "model_source": model_source,
                "checkpoint_path": str(checkpoint_path),
                "trigger_size": int(stage_cfg.get("trigger_size", 4)),
                "alpha": float(stage_cfg.get("alpha", 0.8)),
                "position": str(stage_cfg.get("position", "bottom_right")),
            }
        },
        detection_config=config["detection"],
        image_source=_normalize_image_source(args.image_source),
        random_pick=bool(args.random_pick),
        image_index=int(args.image_index),
        trigger_type=trigger,
        skip_target_label=bool(args.skip_target_label),
        random_seed=random_seed,
        target_label=target_label,
        k=int(args.k),
        seed=int(args.seed),
    )
    result["device_target"] = actual_device
    result["checkpoint_path"] = str(checkpoint_path)
    result["ms_mode"] = str(args.ms_mode).upper()

    output_dir = ensure_dir(Path(args.output_dir))
    result_path = output_dir / f"{trigger}_strip_light_result.notebook.live.json"
    preview_path = output_dir / f"{trigger}_strip_light_preview.notebook.live.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        from demo_lib.detection_visualization import show_strip_light_perturbation_grid

        preview_figure = show_strip_light_perturbation_grid(result)
        preview_figure.savefig(preview_path, dpi=160, bbox_inches="tight", facecolor="white")
        result["preview_png_path"] = str(preview_path)
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") != "matplotlib":
            raise
        result["preview_png_path"] = ""
        print("PREVIEW_SKIPPED=matplotlib_missing")

    save_strip_light_result(result, result_path)

    print(f"RESULT_JSON={result_path}")
    if result.get("preview_png_path"):
        print(f"PREVIEW_PNG={preview_path}")
    print(f"DEVICE_TARGET={actual_device}")
    print(f"MS_MODE={str(args.ms_mode).upper()}")
    print(f"MODEL_SOURCE={model_source}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any


def ensure_project_imports() -> Path:
    os.environ.setdefault("GLOG_v", "3")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore")

    module_path = Path(__file__).resolve()
    module_onsite_root = module_path.parents[1]
    module_project_root = module_path.parents[3]
    if (module_onsite_root / "demo_lib" / "paths.py").exists():
        for import_root in (module_project_root, module_onsite_root):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
        return module_project_root

    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        onsite_candidate = candidate / "src" / "onsite_demo"
        if not onsite_candidate.exists():
            onsite_candidate = (
                candidate
                / "tutorials"
                / "ai_model_backdoor_attack_and_detection"
                / "01_model_backdoor_attack_and_detection"
                / "src"
                / "onsite_demo"
            )
        if (onsite_candidate / "demo_lib" / "paths.py").exists():
            project_root = onsite_candidate.parents[1]
            for import_root in (project_root, onsite_candidate):
                if str(import_root) not in sys.path:
                    sys.path.insert(0, str(import_root))
            return project_root
    raise FileNotFoundError(
        "Could not locate src/onsite_demo/demo_lib. Run this notebook from the chapter root "
        "or from the project root."
    )


def _display_table(rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
        from IPython.display import display

        display(pd.DataFrame(rows))
    except Exception:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, indent=2))


def _widget_fallback_markdown():
    from IPython.display import Markdown

    return Markdown(
        "当前环境没有可用的 ipywidgets，下面切换到普通 cell 兜底展示。"
        "安装 `ipywidgets` 和 `jupyterlab_widgets` 后重开 notebook 可恢复交互。"
    )


def _latest_demo_checkpoint(paths: dict[str, Path], stage_dir_name: str) -> Path | None:
    candidates = sorted(
        paths["DEMO_RUNS_ROOT"].glob(f"notebook*/{stage_dir_name}/demo_last.ckpt"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_detection_stage_bundle(ns: dict[str, Any], trigger_type: str) -> dict[str, Any]:
    from .inference import load_model_from_checkpoint
    from .paths import find_checkpoint

    trigger = str(trigger_type).strip().lower()
    if trigger not in {"square", "checkerboard"}:
        raise ValueError("trigger_type must be 'square' or 'checkerboard'.")

    config = ns["CONFIG"]
    paths = ns["paths"]
    mode_settings = ns["MODE_SETTINGS"]
    stage_cfg = config[trigger]
    stage_dir_name = "square_baseline" if trigger == "square" else "checkerboard_improved"
    fallback_ckpt = find_checkpoint(stage_cfg["experiment_name"])
    live_ckpt = _latest_demo_checkpoint(paths, stage_dir_name)
    if live_ckpt is not None and bool(mode_settings["prefer_live_trained_checkpoint"]):
        checkpoint_path = live_ckpt
        model_source = "live_trained_demo_checkpoint"
        model_reason = f"loaded latest experiment4 checkpoint: {live_ckpt}"
    else:
        checkpoint_path = fallback_ckpt
        model_source = "official_checkpoint_fallback"
        model_reason = "experiment4 live checkpoint not found; using packaged official checkpoint"

    model = load_model_from_checkpoint(
        experiment_name=stage_cfg["experiment_name"],
        checkpoint_path=checkpoint_path,
        num_classes=43,
        norm_type="group",
        device_target=ns["ACTUAL_DEVICE"],
        ms_mode=ns["MS_MODE"],
        allow_cpu_fallback=False,
        pool_type="reduce_mean",
    )
    return {
        "model": model,
        "model_source": model_source,
        "model_reason": model_reason,
        "checkpoint_path": str(checkpoint_path),
        "trigger_size": int(stage_cfg.get("trigger_size", 4)),
        "alpha": float(stage_cfg.get("alpha", 0.8)),
        "position": str(stage_cfg.get("position", "bottom_right")),
    }


def ensure_exp5_context(
    ns: dict[str, Any],
    *,
    require_subset: bool = False,
    require_mindspore: bool = False,
    require_models: bool = False,
    model_trigger_type: str | None = None,
) -> dict[str, Any]:
    ensure_project_imports()

    import numpy as np
    from IPython.display import Markdown, display

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        class _PlotFallback:
            @staticmethod
            def close(*_args: Any, **_kwargs: Any) -> None:
                return None

        plt = _PlotFallback()

    from .detection import (
        build_attack_detection_overview_rows,
        load_formal_detection_bundle,
        save_strip_light_result,
    )
    from .detection_visualization import (
        plot_formal_strip_metrics,
        plot_neural_cleanse_anomaly_comparison,
        plot_neural_cleanse_mask_norms,
        show_attack_detection_overview,
        show_formal_detection_table,
        show_neural_cleanse_summary,
    )
    from .interactive_demo import (
        collect_attack_images,
        run_detection_widget_demo_if_available,
        run_free_detection_test,
        show_attack_image_gallery,
        show_strip_light_detection_result,
    )
    from .paths import (
        ensure_dir,
        get_detection_paths,
        load_demo_config,
        load_final_summary,
        make_run_timestamp,
        resolve_demo_mode_settings,
        resolve_project_root,
    )
    from .subset import create_demo_subset

    try:
        import ipywidgets as widgets  # noqa: F401

        widgets_available = True
        widget_import_error = ""
    except Exception as exc:  # noqa: BLE001
        widgets_available = False
        widget_import_error = repr(exc)

    paths = ns.get("paths") or resolve_project_root()
    config = ns.get("CONFIG") or load_demo_config()
    run_profile = str(ns.get("RUN_PROFILE") or os.environ.get("ONSITE_DEMO_PROFILE", "cloud_live")).strip().lower()
    if run_profile not in {"fast", "enhanced", "cloud_live"}:
        raise RuntimeError(f"Unsupported RUN_PROFILE: {run_profile}")
    demo_mode = str(ns.get("DEMO_MODE") or run_profile)
    mode_settings = ns.get("MODE_SETTINGS") or resolve_demo_mode_settings(config, demo_mode)
    target_label = int(ns.get("TARGET_LABEL", config["target_label"]))
    seed = int(ns.get("SEED", config["seed"]))
    ms_mode = str(ns.get("MS_MODE", config.get("ms_mode", "GRAPH")))
    train_per_class = int(ns.get("TRAIN_PER_CLASS", mode_settings["train_per_class"]))
    test_per_class = int(ns.get("TEST_PER_CLASS", mode_settings["test_per_class"]))
    batch_size = int(ns.get("BATCH_SIZE", mode_settings["batch_size"]))

    demo_run_root = ns.get("DEMO_RUN_ROOT")
    if demo_run_root is None:
        demo_run_root = ensure_dir(paths["DEMO_RUNS_ROOT"] / f"notebook_exp5_{make_run_timestamp()}")
    else:
        demo_run_root = Path(demo_run_root)
    subset_root = ns.get("NOTEBOOK_DEMO_SUBSET_ROOT") or paths.get("DEMO_SUBSET_ROOT") or (demo_run_root / "demo_subset")
    subset_root = ensure_dir(Path(subset_root))
    paths["DEMO_SUBSET_ROOT"] = subset_root
    detection_output_dir = ensure_dir(Path(ns.get("DETECTION_OUTPUT_DIR", demo_run_root / "detection")))
    detection_config = dict(ns.get("DETECTION_CONFIG", config["detection"]))
    detection_config["strip_light_batch_size"] = int(detection_config.get("strip_light_batch_size", 1))

    ns.update(
        {
            "os": os,
            "sys": sys,
            "json": json,
            "Path": Path,
            "plt": plt,
            "np": np,
            "Markdown": Markdown,
            "display": display,
            "WIDGETS_AVAILABLE": bool(ns.get("WIDGETS_AVAILABLE", widgets_available)),
            "WIDGET_IMPORT_ERROR": str(ns.get("WIDGET_IMPORT_ERROR", widget_import_error)),
            "paths": paths,
            "PROJECT_ROOT": ns.get("PROJECT_ROOT", paths["PROJECT_ROOT"]),
            "CONFIG": config,
            "RUN_PROFILE": run_profile,
            "DEMO_MODE": demo_mode,
            "MODE_SETTINGS": mode_settings,
            "TARGET_LABEL": target_label,
            "SEED": seed,
            "MS_MODE": ms_mode,
            "PREFERRED_DEVICE": str(ns.get("PREFERRED_DEVICE", "Ascend")),
            "TRAIN_PER_CLASS": train_per_class,
            "TEST_PER_CLASS": test_per_class,
            "BATCH_SIZE": batch_size,
            "DEMO_RUN_ROOT": demo_run_root,
            "NOTEBOOK_DEMO_SUBSET_ROOT": subset_root,
            "DETECTION_OUTPUT_DIR": detection_output_dir,
            "DETECTION_CONFIG": detection_config,
            "DETECTION_BATCH_SIZE": int(detection_config.get("strip_light_batch_size", 1)),
            "DETECTION_PATHS": get_detection_paths(),
            "summary": ns.get("summary") or load_final_summary(),
            "display_table": ns.get("display_table") or _display_table,
            "widget_fallback_markdown": ns.get("widget_fallback_markdown") or _widget_fallback_markdown,
            "collect_attack_images": collect_attack_images,
            "run_detection_widget_demo_if_available": run_detection_widget_demo_if_available,
            "run_free_detection_test": run_free_detection_test,
            "show_attack_image_gallery": show_attack_image_gallery,
            "show_strip_light_detection_result": show_strip_light_detection_result,
            "create_demo_subset": create_demo_subset,
            "build_attack_detection_overview_rows": build_attack_detection_overview_rows,
            "load_formal_detection_bundle": load_formal_detection_bundle,
            "save_strip_light_result": save_strip_light_result,
            "plot_formal_strip_metrics": plot_formal_strip_metrics,
            "plot_neural_cleanse_anomaly_comparison": plot_neural_cleanse_anomaly_comparison,
            "plot_neural_cleanse_mask_norms": plot_neural_cleanse_mask_norms,
            "show_attack_detection_overview": show_attack_detection_overview,
            "show_formal_detection_table": show_formal_detection_table,
            "show_neural_cleanse_summary": show_neural_cleanse_summary,
        }
    )

    if require_subset:
        subset_test_dir = subset_root / "test"
        if not subset_test_dir.exists():
            create_demo_subset(
                train_dir=paths["TRAIN_DIR"],
                test_dir=paths["TEST_DIR"],
                output_dir=subset_root,
                train_per_class=train_per_class,
                test_per_class=test_per_class,
                seed=seed,
            )

    if require_mindspore or require_models:
        from .runtime import configure_mindspore_device

        try:
            import mindspore as ms
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("MindSpore is not importable. Use a MindSpore notebook kernel.") from exc

        actual_device = ns.get("ACTUAL_DEVICE") or configure_mindspore_device(
            preferred=ns["PREFERRED_DEVICE"],
            ms_mode=ms_mode,
            allow_cpu_fallback=False,
        )
        ns.update({"ms": ms, "ACTUAL_DEVICE": actual_device})

    if require_models:
        bundles = dict(ns.get("DETECTION_MODEL_BUNDLES", {}))
        triggers = [model_trigger_type] if model_trigger_type else ["square", "checkerboard"]
        for trigger in triggers:
            trigger_key = str(trigger).strip().lower()
            if trigger_key not in bundles:
                bundles[trigger_key] = _load_detection_stage_bundle(ns, trigger_key)
        ns["DETECTION_MODEL_BUNDLES"] = bundles

    return ns

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _is_trash_path(path: Path) -> bool:
    return ".Trash-1000" in str(path).replace("\\", "/")


def _candidate_roots() -> list[Path]:
    starts = [Path.cwd(), Path(__file__).resolve()]
    candidates: list[Path] = []
    for start in starts:
        current = start if start.is_dir() else start.parent
        for path in [current, *current.parents]:
            if _is_trash_path(path):
                continue
            if path not in candidates:
                candidates.append(path)
    return candidates


def _looks_like_project_root(path: Path) -> bool:
    return (path / "src" / "train.py").exists() and (path / "src" / "model.py").exists()


def _first_existing(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


def resolve_project_root() -> dict[str, Path]:
    """Resolve the onsite package root without relying on absolute paths."""

    project_root: Path | None = None
    for candidate in _candidate_roots():
        if _looks_like_project_root(candidate):
            project_root = candidate
            break

    if project_root is None:
        raise FileNotFoundError(
            "Could not locate the onsite demo package root. Start Python from the project root, "
            "src/onsite_demo/, or the chapter root, and make sure src/train.py and src/model.py exist."
        )
    if _is_trash_path(project_root):
        raise RuntimeError(f"Resolved project root points into .Trash-1000, which is not allowed: {project_root}")

    source_root = project_root / "src"
    assets_root = source_root / "assets"
    onsite_demo_root = source_root / "onsite_demo"
    output_root = onsite_demo_root / "outputs"
    server_artifacts_dir = _first_existing(
        [
            assets_root / "evidence" / "server_artifacts",
            source_root / "server_artifacts",
            source_root / "outputs" / "experiments",
        ],
        assets_root / "evidence" / "server_artifacts",
    )
    final_summary_json = _first_existing(
        [
            source_root / "final_summary.json",
            source_root / "outputs" / "metrics" / "final_summary.json",
            assets_root / "results" / "metrics" / "final_summary.json",
            assets_root / "results" / "final_summary.json",
        ],
        assets_root / "results" / "metrics" / "final_summary.json",
    )
    detection_results_dir = assets_root / "results" / "detection"
    detection_summary_json = detection_results_dir / "detection_complete_summary.json"
    detection_table_csv = detection_results_dir / "detection_complete_table.csv"
    detection_report_md = detection_results_dir / "detection_complete_report.md"

    return {
        "PROJECT_ROOT": project_root,
        "SOURCE_ROOT": source_root,
        "ASSETS_ROOT": assets_root,
        "ONSITE_DEMO_ROOT": onsite_demo_root,
        "OUTPUT_ROOT": output_root,
        "DEMO_SUBSET_ROOT": output_root / "demo_subset",
        "DEMO_RUNS_ROOT": output_root / "demo_runs",
        "SRC_DIR": source_root,
        "TRAIN_DIR": source_root / "data" / "train",
        "TEST_DIR": source_root / "data" / "test",
        "FINAL_SUMMARY_JSON": final_summary_json,
        "SERVER_ARTIFACTS_DIR": server_artifacts_dir,
        "DETECTION_RESULTS_DIR": detection_results_dir,
        "DETECTION_SUMMARY_JSON": detection_summary_json,
        "DETECTION_TABLE_CSV": detection_table_csv,
        "DETECTION_REPORT_MD": detection_report_md,
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_demo_config(config_path: Path | None = None) -> dict[str, Any]:
    paths = resolve_project_root()
    path = config_path or paths["ONSITE_DEMO_ROOT"] / "configs" / "demo_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Onsite demo config file is missing: {path}")
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Onsite demo config must be a JSON object: {path}")
    return data


def resolve_demo_mode_settings(
    config: dict[str, Any],
    mode: str | None = None,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    train_per_class: int | None = None,
    test_per_class: int | None = None,
) -> dict[str, Any]:
    modes = dict(config.get("modes", {}))
    selected_mode = str(mode or config.get("presentation_default_mode") or "enhanced").strip()
    if selected_mode not in modes:
        raise ValueError(f"Unknown onsite demo mode: {selected_mode}. Available modes: {sorted(modes)}")

    profile = {}
    if isinstance(config.get(selected_mode), dict):
        profile.update(dict(config[selected_mode]))
    profile.update(dict(modes[selected_mode]))

    def _pick_int(key: str, override: int | None, fallback: int | None = None) -> int:
        if override is not None:
            return int(override)
        if key in profile:
            return int(profile[key])
        if key in config:
            return int(config[key])
        if fallback is not None:
            return int(fallback)
        raise KeyError(f"Missing integer config key: {key}")

    def _pick_bool(key: str, fallback: bool) -> bool:
        if key in profile:
            return bool(profile[key])
        if key in config:
            return bool(config[key])
        return bool(fallback)

    detection = dict(config.get("detection", {}))
    return {
        "mode": selected_mode,
        "train_per_class": _pick_int("train_per_class", train_per_class),
        "test_per_class": _pick_int("test_per_class", test_per_class),
        "epochs": _pick_int("epochs", epochs),
        "batch_size": _pick_int("batch_size", batch_size),
        "eval_each_epoch": _pick_bool("eval_each_epoch", True),
        "save_epoch_metrics": _pick_bool("save_epoch_metrics", True),
        "save_training_curve": _pick_bool("save_training_curve", True),
        "prefer_live_trained_checkpoint": _pick_bool(
            "prefer_live_trained_checkpoint",
            bool(detection.get("prefer_live_trained_checkpoint", True)),
        ),
        "strip_light_k": _pick_int(
            "strip_light_k",
            None,
            fallback=int(detection.get("strip_light_default_k", 8)),
        ),
    }


def load_final_summary() -> list[dict[str, Any]]:
    paths = resolve_project_root()
    summary_path = paths["FINAL_SUMMARY_JSON"]
    if not summary_path.exists():
        raise FileNotFoundError(
            "Could not find final_summary.json. Place it under src/, src/outputs/metrics/, "
            "or src/assets/results/metrics/."
        )

    data = load_json(summary_path)
    if isinstance(data, dict):
        for key in ("experiments", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported final_summary.json format: {summary_path}")


def get_detection_paths() -> dict[str, Path]:
    paths = resolve_project_root()
    return {
        "DETECTION_RESULTS_DIR": paths["DETECTION_RESULTS_DIR"],
        "DETECTION_SUMMARY_JSON": paths["DETECTION_SUMMARY_JSON"],
        "DETECTION_TABLE_CSV": paths["DETECTION_TABLE_CSV"],
        "DETECTION_REPORT_MD": paths["DETECTION_REPORT_MD"],
    }


def _require_existing_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} file is missing: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} path is not a file: {path}")
    return path


def load_detection_summary() -> dict[str, Any]:
    detection_paths = get_detection_paths()
    summary_path = _require_existing_file(detection_paths["DETECTION_SUMMARY_JSON"], "detection summary")
    data = load_json(summary_path)
    if not isinstance(data, dict):
        raise ValueError(f"Detection summary must be a JSON object: {summary_path}")
    return data


def _parse_csv_value(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    for parser in (int, float):
        try:
            return parser(text)
        except ValueError:
            continue
    return value


def load_detection_table() -> list[dict[str, Any]]:
    detection_paths = get_detection_paths()
    table_path = _require_existing_file(detection_paths["DETECTION_TABLE_CSV"], "detection table")
    rows: list[dict[str, Any]] = []
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _parse_csv_value(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"Detection table CSV is empty: {table_path}")
    return rows


def make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stage_display_name(trigger_type: str) -> str:
    normalized = trigger_type.lower().strip()
    if normalized == "square":
        return "Square Trigger Baseline"
    if normalized == "checkerboard":
        return "Checkerboard Trigger Improved"
    return trigger_type


def stage_output_name(trigger_type: str) -> str:
    normalized = trigger_type.lower().strip()
    if normalized == "square":
        return "square_baseline"
    if normalized == "checkerboard":
        return "checkerboard_improved"
    return normalized


def _checkpoint_candidates_from_summary(experiment_name: str) -> list[Path]:
    paths = resolve_project_root()
    candidates: list[Path] = []
    try:
        summary = load_final_summary()
    except Exception:
        return candidates

    for item in summary:
        if str(item.get("experiment", "")).strip() != experiment_name:
            continue
        raw = item.get("ckpt_path") or item.get("checkpoint") or item.get("checkpoint_path")
        if not raw:
            continue
        raw_path = Path(str(raw))
        candidates.append(raw_path if raw_path.is_absolute() else paths["PROJECT_ROOT"] / raw_path)
        candidates.append(paths["SERVER_ARTIFACTS_DIR"] / f"{experiment_name}_{raw_path.name}")
        candidates.append(paths["SERVER_ARTIFACTS_DIR"] / raw_path.name)
    return candidates


def find_checkpoint(experiment_name: str, checkpoint_path: str | Path | None = None) -> Path:
    """Find the official checkpoint for an experiment and fail with a clear hint."""

    paths = resolve_project_root()
    candidates: list[Path] = []
    if checkpoint_path:
        raw = Path(checkpoint_path)
        candidates.append(raw if raw.is_absolute() else paths["PROJECT_ROOT"] / raw)

    candidates.extend(_checkpoint_candidates_from_summary(experiment_name))
    candidates.extend(
        [
            paths["SERVER_ARTIFACTS_DIR"] / f"{experiment_name}_best_epoch_10.ckpt",
            paths["SOURCE_ROOT"] / "outputs" / "experiments" / experiment_name / "best_epoch_10.ckpt",
        ]
    )

    experiment_dir = paths["SOURCE_ROOT"] / "outputs" / "experiments" / experiment_name
    if experiment_dir.exists():
        candidates.extend(sorted(experiment_dir.glob("best_epoch_*.ckpt"), reverse=True))
    if paths["SERVER_ARTIFACTS_DIR"].exists():
        candidates.extend(sorted(paths["SERVER_ARTIFACTS_DIR"].glob(f"{experiment_name}*.ckpt"), reverse=True))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate

    checked = "\n".join(f"  - {path}" for path in candidates[:12])
    raise FileNotFoundError(
        f"Could not find an official checkpoint for {experiment_name}.\n"
        f"Checked:\n{checked}\n"
        "Upload the server-produced checkpoint or fix the path in final_summary.json."
    )

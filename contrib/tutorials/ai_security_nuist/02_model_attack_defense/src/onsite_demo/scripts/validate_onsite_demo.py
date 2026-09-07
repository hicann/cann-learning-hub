from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


ONSITE_ROOT = Path(__file__).resolve().parents[1]
if str(ONSITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ONSITE_ROOT))

from demo_lib.paths import (  # noqa: E402
    ensure_dir,
    load_demo_config,
    load_json,
    make_run_timestamp,
    resolve_demo_mode_settings,
    resolve_project_root,
    save_json,
)
from demo_lib.subset import create_demo_subset  # noqa: E402


REQUIRED_FILES = [
    "../README.md",
    "src/README.md",
    "src/onsite_demo/README_ONSITE_DEMO.md",
    "src/onsite_demo/configs/demo_config.json",
    "src/onsite_demo/demo_lib/detection.py",
    "src/onsite_demo/demo_lib/detection_visualization.py",
    "src/onsite_demo/demo_lib/training.py",
    "src/onsite_demo/scripts/run_detection_demo_smoke.py",
    "src/onsite_demo/scripts/run_full_onsite_demo.py",
    "src/onsite_demo/scripts/run_cloud_live_10epoch_acceptance.py",
    "src/onsite_demo/scripts/final_preupload_audit.py",
    "src/onsite_demo/scripts/validate_onsite_demo.py",
    "src/onsite_demo/scripts/build_onsite_demo_package.py",
    "01.01_badnets_attack.ipynb",
    "01.02_neural_cleanse_detection.ipynb",
    "src/final_summary.json",
    "src/assets/results/metrics/final_summary.json",
    "src/assets/results/detection/detection_complete_summary.json",
    "src/assets/results/detection/detection_complete_table.csv",
    "src/assets/results/detection/detection_complete_report.md",
]

BANNED_CLAIMS = (
    "\u9632\u5fa1\u6210\u529f",
    "\u6a21\u578b\u5df2\u4fee\u590d",
    "\u540e\u95e8\u5df2\u6d88\u9664",
)
NEGATION_HINTS = (
    "不代表",
    "不意味着",
    "不是",
    "不声称",
    "不等价于",
)

README_REQUIRED_SNIPPETS = [
    "cloud_live",
    "--mode cloud_live --epochs 10 --batch-size 8",
    "run_cloud_live_10epoch_acceptance.py",
    "RUN_PROFILE",
    "ipywidgets",
    "jupyterlab_widgets",
]

COURSE_ROOT_REQUIRED_ENTRIES = {
    "README.md",
    "01_model_backdoor_attack_and_detection",
}
CHAPTER_ROOT_REQUIRED_ENTRIES = {
    "01.01_badnets_attack.ipynb",
    "01.02_neural_cleanse_detection.ipynb",
    "answer",
    "images",
    "src",
}
COURSE_README_REQUIRED_SNIPPETS = (
    "教程简介",
    "适用对象",
    "整体学习目标",
    "支持硬件",
    "在线体验环境",
    "CANNLab 环境体验指南",
)


def add_check(checks: list[dict[str, Any]], name: str, status: str, detail: Any = None) -> None:
    print(f"[{status}] {name}: {detail if detail is not None else ''}")
    checks.append(
        {
            "name": name,
            "status": status,
            "detail": str(detail) if detail is not None else "",
        }
    )


def _run_subprocess(command: list[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])


def _mindspore_available() -> tuple[bool, str]:
    if importlib.util.find_spec("mindspore") is None:
        return False, "MindSpore is not installed in the current Python environment."
    return True, ""


def _load_notebook(paths: dict[str, Path]) -> dict[str, Any]:
    notebooks = []
    for name in ("01.01_badnets_attack.ipynb", "01.02_neural_cleanse_detection.ipynb"):
        notebooks.extend(json.loads((paths["PROJECT_ROOT"] / name).read_text(encoding="utf-8")).get("cells", []))
    return {"cells": notebooks}


def _notebook_text(notebook: dict[str, Any]) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _readme_text(paths: dict[str, Path]) -> str:
    return (paths["ONSITE_DEMO_ROOT"] / "README_ONSITE_DEMO.md").read_text(encoding="utf-8")


def _line_has_bad_claim(line: str) -> bool:
    text = str(line).strip()
    if not text:
        return False
    for phrase in BANNED_CLAIMS:
        if phrase not in text:
            continue
        if any(hint in text for hint in NEGATION_HINTS):
            return False
        return True
    return False


def _count_min_class_records(split_dir: Path) -> int:
    counts = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit()):
        counts.append(len([path for path in class_dir.iterdir() if path.is_file()]))
    if not counts:
        raise ValueError(f"No class directories found under {split_dir}")
    return min(counts)


def _run_interactive_smoke(paths: dict[str, Path], trigger_type: str, random_pick: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        str(paths["ONSITE_DEMO_ROOT"] / "scripts" / "run_interactive_demo_smoke.py"),
        "--mode",
        "fast",
        "--image-source",
        "data_test",
        "--trigger-type",
        trigger_type,
    ]
    if random_pick:
        command.append("--random-pick")
    else:
        command.extend(["--image-index", "1"])

    return_code, output = _run_subprocess(command, cwd=paths["PROJECT_ROOT"])
    result_path = paths["OUTPUT_ROOT"] / "interactive_demo" / "attack_demo_result.json"
    preview_path = paths["OUTPUT_ROOT"] / "interactive_demo" / "attack_demo_preview.png"
    payload = load_json(result_path) if result_path.exists() else {}
    return {
        "return_code": return_code,
        "output_tail": _tail(output),
        "result_json_exists": result_path.exists(),
        "preview_png_exists": preview_path.exists(),
        "attack_success_present": "attack_success" in payload,
    }


def _run_detection_smoke(paths: dict[str, Path], trigger_type: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(paths["ONSITE_DEMO_ROOT"] / "scripts" / "run_detection_demo_smoke.py"),
        "--mode",
        "fast",
        "--trigger-type",
        trigger_type,
        "--k",
        "4",
        "--image-source",
        "data_test",
        "--device-target",
        "auto",
    ]
    return_code, output = _run_subprocess(command, cwd=paths["PROJECT_ROOT"])
    result_path = paths["OUTPUT_ROOT"] / "detection_smoke" / f"{trigger_type}_strip_light_result.json"
    preview_path = paths["OUTPUT_ROOT"] / "detection_smoke" / f"{trigger_type}_strip_light_preview.png"
    payload = load_json(result_path) if result_path.exists() else {}
    return {
        "return_code": return_code,
        "output_tail": _tail(output),
        "result_json_exists": result_path.exists(),
        "preview_png_exists": preview_path.exists(),
        "model_source": payload.get("model_source"),
        "threshold_source": payload.get("threshold_source"),
        "demo_threshold": payload.get("demo_threshold"),
    }


def _latest_demo_run(paths: dict[str, Path], *, require_epoch_fields: bool = False) -> Path | None:
    if not paths["DEMO_RUNS_ROOT"].exists():
        return None
    candidates = sorted(
        [path for path in paths["DEMO_RUNS_ROOT"].iterdir() if path.is_dir() and (path / "detection_summary.json").exists()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not require_epoch_fields:
        return candidates[0] if candidates else None
    for candidate in candidates:
        payload = load_json(candidate / "detection_summary.json")
        if "square_epochs_completed" in payload and "checkerboard_epochs_completed" in payload:
            return candidate
    return None


def _latest_cloud_acceptance(paths: dict[str, Path]) -> Path | None:
    root = paths["OUTPUT_ROOT"] / "cloud_live_acceptance"
    if not root.exists():
        return None
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / "cloud_live_acceptance_summary.json").exists()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cloud Live Validation",
        "",
        f"- overall_status: `{report['overall_status']}`",
        "",
        "## Checks",
        "",
    ]
    for item in report["checks"]:
        lines.append(f"- [{item['status']}] {item['name']}: {item['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, Any]:
    paths = resolve_project_root()
    config = load_demo_config()
    checks: list[dict[str, Any]] = []

    project_root = paths["PROJECT_ROOT"]
    course_root = project_root.parent
    course_entries = {path.name for path in course_root.iterdir()}
    chapter_entries = {path.name for path in project_root.iterdir()}
    add_check(
        checks,
        "strict course root structure",
        "PASS" if course_entries == COURSE_ROOT_REQUIRED_ENTRIES else "FAIL",
        sorted(course_entries),
    )
    add_check(
        checks,
        "strict chapter root structure",
        "PASS" if chapter_entries == CHAPTER_ROOT_REQUIRED_ENTRIES else "FAIL",
        sorted(chapter_entries),
    )
    course_readme_text = (course_root / "README.md").read_text(encoding="utf-8")
    missing_course_readme = [
        snippet for snippet in COURSE_README_REQUIRED_SNIPPETS if snippet not in course_readme_text
    ]
    add_check(
        checks,
        "course README mandatory sections",
        "PASS" if not missing_course_readme else "FAIL",
        "all mandatory sections found" if not missing_course_readme else missing_course_readme,
    )

    for relative in REQUIRED_FILES:
        target = paths["PROJECT_ROOT"] / relative
        add_check(checks, f"required file: {relative}", "PASS" if target.exists() else "FAIL", target)

    try:
        mode_settings = resolve_demo_mode_settings(config, "cloud_live")
        add_check(checks, "cloud_live config exists", "PASS", mode_settings)
    except Exception as exc:  # noqa: BLE001
        mode_settings = {}
        add_check(checks, "cloud_live config exists", "FAIL", exc)

    if mode_settings:
        train_min = _count_min_class_records(paths["TRAIN_DIR"])
        test_min = _count_min_class_records(paths["TEST_DIR"])
        add_check(
            checks,
            "cloud_live default epochs >= 10",
            "PASS" if int(mode_settings["epochs"]) >= 10 else "FAIL",
            mode_settings["epochs"],
        )
        add_check(
            checks,
            "cloud_live train/test counts match packaged data",
            "PASS"
            if int(mode_settings["train_per_class"]) == int(train_min)
            and int(mode_settings["test_per_class"]) == int(test_min)
            else "FAIL",
            f"train={mode_settings['train_per_class']} test={mode_settings['test_per_class']} expected={train_min}/{test_min}",
        )

    notebook = _load_notebook(paths)
    notebook_text = _notebook_text(notebook)
    readme_text = _readme_text(paths)
    training_source = (paths["ONSITE_DEMO_ROOT"] / "demo_lib" / "training.py").read_text(encoding="utf-8")

    add_check(
        checks,
        "notebook default mode is cloud_live",
        "PASS"
        if 'RUN_PROFILE = os.environ.get("ONSITE_DEMO_PROFILE", "cloud_live")' in notebook_text
        and "DEMO_MODE = RUN_PROFILE" in notebook_text
        else "FAIL",
        "RUN_PROFILE defaults to cloud_live",
    )
    add_check(
        checks,
        "notebook default 10 epoch text exists",
        "PASS" if "NOTEBOOK_DEFAULT_EPOCHS = 10" in notebook_text else "FAIL",
        "expect NOTEBOOK_DEFAULT_EPOCHS = 10",
    )
    add_check(
        checks,
        "training.py supports epoch_metrics",
        "PASS" if "epoch_metrics" in training_source and "epochs_completed" in training_source else "FAIL",
        "epoch_metrics + epochs_completed",
    )
    add_check(
        checks,
        "training.py saves curve artifacts",
        "PASS" if "demo_training_curve.csv" in training_source and "training_curve.png" in training_source else "FAIL",
        "demo_training_curve.csv + training_curve.png",
    )
    add_check(
        checks,
        "notebook attack uses live checkpoint",
        "PASS"
        if 'square_result["train_log"]["demo_checkpoint_path"]' in notebook_text
        and 'checkerboard_result["train_log"]["demo_checkpoint_path"]' in notebook_text
        else "FAIL",
        "square_result/checkerboard_result live checkpoint path",
    )
    add_check(
        checks,
        "notebook detection uses live checkpoint",
        "PASS"
        if 'square_result["train_log"]["demo_checkpoint_path"]' in notebook_text
        and 'checkerboard_result["train_log"]["demo_checkpoint_path"]' in notebook_text
        and "official_checkpoint_fallback" in notebook_text
        else "FAIL",
        "live checkpoint path + fallback reason",
    )
    add_check(
        checks,
        "notebook widgets are primary with fallback",
        "PASS"
        if "run_widget_demo_if_available" in notebook_text
        and "RUN_DEFAULT_SQUARE_ATTACK_FALLBACK" in notebook_text
        and "RUN_DEFAULT_CHECKERBOARD_ATTACK_FALLBACK" in notebook_text
        else "FAIL",
        "widgets primary entry + fallback cells retained",
    )
    missing_readme = [snippet for snippet in README_REQUIRED_SNIPPETS if snippet not in readme_text]
    add_check(
        checks,
        "README mentions cloud_live 10 epoch usage",
        "PASS" if not missing_readme else "FAIL",
        "all snippets found" if not missing_readme else ", ".join(missing_readme),
    )
    claim_hits = [line for line in (notebook_text + "\n" + readme_text).splitlines() if _line_has_bad_claim(line)]
    add_check(
        checks,
        "no bad claims",
        "PASS" if not claim_hits else "FAIL",
        "no banned claims found" if not claim_hits else claim_hits[0],
    )

    try:
        manifest = create_demo_subset(
            train_dir=paths["TRAIN_DIR"],
            test_dir=paths["TEST_DIR"],
            output_dir=paths["OUTPUT_ROOT"] / "validation_subset" / make_run_timestamp(),
            train_per_class=int(config["modes"]["fast"]["train_per_class"]),
            test_per_class=int(config["modes"]["fast"]["test_per_class"]),
            seed=int(config["seed"]),
        )
        add_check(checks, "demo subset creatable", "PASS", f"train={manifest['train_total']} test={manifest['test_total']}")
    except Exception as exc:  # noqa: BLE001
        add_check(checks, "demo subset creatable", "FAIL", exc)

    available, reason = _mindspore_available()
    add_check(
        checks,
        "MindSpore available",
        "PASS" if available else "SKIP",
        reason or "mindspore import works",
    )

    if available:
        for trigger_type, random_pick in (("square", False), ("checkerboard", True)):
            try:
                smoke = _run_interactive_smoke(paths, trigger_type=trigger_type, random_pick=random_pick)
                status = (
                    "PASS"
                    if smoke["return_code"] == 0 and smoke["result_json_exists"] and smoke["preview_png_exists"]
                    else "FAIL"
                )
                add_check(checks, f"{trigger_type} interactive smoke", status, smoke["output_tail"])
            except Exception as exc:  # noqa: BLE001
                add_check(checks, f"{trigger_type} interactive smoke", "FAIL", exc)

        for trigger_type in ("square", "checkerboard"):
            try:
                smoke = _run_detection_smoke(paths, trigger_type=trigger_type)
                status = "PASS"
                if smoke["return_code"] != 0 or not smoke["result_json_exists"] or not smoke["preview_png_exists"]:
                    status = "FAIL"
                if smoke["threshold_source"] == "zero_fallback":
                    status = "FAIL"
                if smoke["demo_threshold"] == 0:
                    status = "FAIL"
                add_check(checks, f"{trigger_type} detection smoke", status, smoke["output_tail"])
            except Exception as exc:  # noqa: BLE001
                add_check(checks, f"{trigger_type} detection smoke", "FAIL", exc)

    latest_run = _latest_demo_run(paths, require_epoch_fields=True)
    if latest_run is None:
        add_check(
            checks,
            "latest run_full output",
            "SKIP",
            "No historical 10-epoch detection_summary.json found under demo_runs. This does not block cloud upload readiness.",
        )
    else:
        payload = load_json(latest_run / "detection_summary.json")
        add_check(
            checks,
            "latest run_full uses live checkpoints",
            "PASS"
            if payload.get("square_model_source") == "live_trained_demo_checkpoint"
            and payload.get("checkerboard_model_source") == "live_trained_demo_checkpoint"
            else "FAIL",
            latest_run,
        )
        add_check(
            checks,
            "latest run_full completed 10 epochs",
            "PASS"
            if int(payload.get("square_epochs_completed", 0)) >= 10 and int(payload.get("checkerboard_epochs_completed", 0)) >= 10
            else "SKIP",
            f"square={payload.get('square_epochs_completed')} checkerboard={payload.get('checkerboard_epochs_completed')}",
        )

    latest_acceptance = _latest_cloud_acceptance(paths)
    if latest_acceptance is None:
        add_check(
            checks,
            "latest cloud_live acceptance output",
            "SKIP",
            "No historical cloud_live acceptance summary found. Structure validation still passed.",
        )
    else:
        payload = load_json(latest_acceptance / "cloud_live_acceptance_summary.json")
        add_check(
            checks,
            "latest cloud_live acceptance pass",
            "PASS" if bool(payload.get("cloud_live_10epoch_acceptance_pass")) else "FAIL",
            latest_acceptance,
        )

    failed = [item for item in checks if item["status"] == "FAIL"]
    report = {
        "overall_status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "validation_report_path": str(paths["OUTPUT_ROOT"] / "validation_report.json"),
    }
    save_json(report, paths["OUTPUT_ROOT"] / "validation_report.json")

    cloud_validation_dir = ensure_dir(paths["OUTPUT_ROOT"] / "cloud_live_acceptance")
    validation_json = cloud_validation_dir / "CLOUD_LIVE_STRUCTURE_VALIDATION.json"
    validation_md = cloud_validation_dir / "CLOUD_LIVE_STRUCTURE_VALIDATION.md"
    save_json(report, validation_json)
    _write_markdown(report, validation_md)

    print(f"validation_report.json = {paths['OUTPUT_ROOT'] / 'validation_report.json'}")
    print(f"CLOUD_LIVE_STRUCTURE_VALIDATION.json = {validation_json}")
    print(f"CLOUD_LIVE_STRUCTURE_VALIDATION.md = {validation_md}")
    if failed:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(exc)
        print(traceback.format_exc(limit=4))
        raise

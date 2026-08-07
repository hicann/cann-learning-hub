from __future__ import annotations

import importlib
import json
import re
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any


ONSITE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ONSITE_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ONSITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ONSITE_ROOT))

from demo_lib.paths import ensure_dir, load_demo_config, resolve_demo_mode_settings, resolve_project_root, save_json


REQUIRED_FILES = [
    "../README.md",
    "src/README.md",
    "src/onsite_demo/README_ONSITE_DEMO.md",
    "src/onsite_demo/configs/demo_config.json",
    "01.01_badnets_attack.ipynb",
    "01.02_neural_cleanse_detection.ipynb",
    "src/onsite_demo/demo_lib/training.py",
    "src/onsite_demo/demo_lib/inference.py",
    "src/onsite_demo/demo_lib/evaluation.py",
    "src/onsite_demo/demo_lib/detection.py",
    "src/onsite_demo/demo_lib/detection_visualization.py",
    "src/onsite_demo/scripts/run_full_onsite_demo.py",
    "src/onsite_demo/scripts/run_detection_demo_smoke.py",
    "src/onsite_demo/scripts/run_cloud_live_10epoch_acceptance.py",
    "src/onsite_demo/scripts/final_preupload_audit.py",
    "src/onsite_demo/scripts/validate_onsite_demo.py",
    "src/model.py",
    "src/train.py",
    "src/poison.py",
]

FORMAL_FILES = [
    "src/assets/evidence/server_artifacts/square_main_best_epoch_10.ckpt",
    "src/assets/evidence/server_artifacts/checkerboard_main_best_epoch_10.ckpt",
    "src/assets/results/metrics/final_summary.json",
    "src/assets/results/detection/detection_complete_summary.json",
    "src/assets/results/detection/detection_complete_table.csv",
    "src/assets/results/detection/detection_complete_report.md",
]

BANNED_CLAIMS = ("防御成功", "模型已修复", "后门已消除")
NEGATION_HINTS = ("不代表", "不意味着", "不是", "不声称", "不等价于")
SECTION_TOKENS = (
    "cloud_live",
    "NOTEBOOK_DEFAULT_EPOCHS = 10",
    "plot_demo_training_curve",
    "run_stage_pipeline",
    "run_free_detection_test",
    "show_neural_cleanse_summary",
    "show_attack_detection_overview",
)


def add_check(results: dict[str, Any], name: str, value: Any) -> None:
    results[name] = value
    print(f"{name} = {value}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_notebook_text(notebook_path: Path) -> str:
    notebook = json.loads(_read_text(notebook_path))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _count_class_records(split_dir: Path) -> tuple[int, int]:
    counts: list[int] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit()):
        counts.append(len([path for path in class_dir.iterdir() if path.is_file()]))
    if not counts:
        return 0, 0
    return len(counts), min(counts)


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


def _imports_ok() -> tuple[bool, list[str]]:
    errors: list[str] = []
    modules = [
        "demo_lib.training",
        "demo_lib.inference",
        "demo_lib.evaluation",
        "demo_lib.detection",
        "demo_lib.detection_visualization",
    ]
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}: {exc}")
    return not errors, errors


def _check_candidate_zip(paths: dict[str, Path]) -> tuple[bool, str]:
    zip_path = paths["OUTPUT_ROOT"] / "cloud_live_acceptance" / "ai_model_backdoor_attack_and_detection_cloud_live.zip"
    if not zip_path.exists():
        return True, f"Candidate zip not built yet: {zip_path}"
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            required_prefixes = [
                "01_model_backdoor_attack_and_detection/src/data/train/",
                "01_model_backdoor_attack_and_detection/src/data/test/",
                "01_model_backdoor_attack_and_detection/src/",
                "01_model_backdoor_attack_and_detection/01.01_badnets_attack.ipynb",
                "01_model_backdoor_attack_and_detection/01.02_neural_cleanse_detection.ipynb",
                "01_model_backdoor_attack_and_detection/src/assets/evidence/server_artifacts/",
                "01_model_backdoor_attack_and_detection/src/assets/results/detection/",
            ]
            for prefix in required_prefixes:
                if not any(name.startswith(prefix) or f"/{prefix}" in name for name in names):
                    return False, f"Candidate zip missing required entry: {prefix}"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)
    return True, str(zip_path)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# FINAL_PREUPLOAD_AUDIT",
        "",
        f"- final_preupload_audit_pass: `{report['final_preupload_audit_pass']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.items():
        if key == "details":
            continue
        lines.append(f"- {key}: `{value}`")
    if report.get("details"):
        lines.extend(["", "## Details", ""])
        for key, value in report["details"].items():
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, Any]:
    paths = resolve_project_root()
    project_root = paths["PROJECT_ROOT"]
    output_dir = ensure_dir(paths["OUTPUT_ROOT"] / "final_preupload_audit")

    config = load_demo_config()
    notebook_path = paths["PROJECT_ROOT"] / "01.01_badnets_attack.ipynb"
    readme_path = paths["ONSITE_DEMO_ROOT"] / "README_ONSITE_DEMO.md"
    training_path = paths["ONSITE_DEMO_ROOT"] / "demo_lib" / "training.py"
    interactive_demo_path = paths["ONSITE_DEMO_ROOT"] / "demo_lib" / "interactive_demo.py"

    notebook_text = _load_notebook_text(notebook_path) + "\n" + _load_notebook_text(
        paths["PROJECT_ROOT"] / "01.02_neural_cleanse_detection.ipynb"
    )
    readme_text = _read_text(readme_path)
    training_text = _read_text(training_path)
    interactive_demo_text = _read_text(interactive_demo_path)

    required_files_present = all((project_root / relative).exists() for relative in REQUIRED_FILES)
    formal_results_present = all((project_root / relative).exists() for relative in FORMAL_FILES)

    train_class_count, train_min_per_class = _count_class_records(paths["TRAIN_DIR"])
    test_class_count, test_min_per_class = _count_class_records(paths["TEST_DIR"])
    data_subset_present = (
        paths["TRAIN_DIR"].exists()
        and paths["TEST_DIR"].exists()
        and train_class_count >= 43
        and test_class_count >= 43
        and train_min_per_class > 0
        and test_min_per_class > 0
    )

    mode_settings = resolve_demo_mode_settings(config, "cloud_live")
    cloud_live_config_ok = all(
        [
            int(mode_settings["epochs"]) >= 10,
            int(mode_settings["batch_size"]) > 0 and int(mode_settings["batch_size"]) <= 32,
            int(mode_settings["train_per_class"]) <= train_min_per_class,
            int(mode_settings["test_per_class"]) <= test_min_per_class,
            bool(mode_settings["eval_each_epoch"]),
            bool(mode_settings["save_training_curve"]),
            bool(mode_settings["prefer_live_trained_checkpoint"]),
            int(mode_settings["strip_light_k"]) >= 4,
        ]
    )

    imports_ok, import_errors = _imports_ok()
    candidate_zip_ok, candidate_zip_detail = _check_candidate_zip(paths)

    notebook_no_local_absolute_paths = not any(token in notebook_text for token in ("D:\\", "D:/", ".venv", ".venv_torch"))
    notebook_no_windows_path_hardcoding = not bool(re.search(r"[A-Za-z]:[\\/]", notebook_text))
    notebook_no_broken_text = "????" not in notebook_text
    notebook_sections_complete = all(token in notebook_text for token in SECTION_TOKENS) and all(
        token in notebook_text
        for token in (
            'square_result["train_log"]["demo_checkpoint_path"]',
            'checkerboard_result["train_log"]["demo_checkpoint_path"]',
            "formal",
            "Neural Cleanse",
            "light STRIP++",
        )
    )
    notebook_text_concise = all(
        token in notebook_text
        for token in ("#### 讲解", "#### 图像注释", "实验4结果解读", "实验5结果解读", "## 结论")
    )

    ipywidgets_primary_present = all(
        token in notebook_text or token in interactive_demo_text
        for token in (
            "import ipywidgets as widgets",
            "run_widget_demo_if_available",
            "ENABLE_WIDGET_DEMO = True",
        )
    )
    ipywidgets_fallback_present = all(
        token in notebook_text or token in interactive_demo_text
        for token in (
            "widget_fallback_markdown",
            "RUN_DEFAULT_SQUARE_ATTACK_FALLBACK",
            "RUN_DEFAULT_CHECKERBOARD_ATTACK_FALLBACK",
            "run_free_attack_test",
        )
    )

    attack_uses_live_checkpoint_path = all(
        token in notebook_text
        for token in (
            'square_result["train_log"]["demo_checkpoint_path"]',
            'checkerboard_result["train_log"]["demo_checkpoint_path"]',
            "live_trained_demo_checkpoint",
            "official_checkpoint_fallback",
        )
    )
    detection_uses_live_checkpoint_path = all(
        token in notebook_text
        for token in (
            "find_checkpoint",
            "DETECTION_MODEL_BUNDLES",
            "prefer_live_trained_checkpoint",
            "official_checkpoint_fallback",
        )
    )

    training_outputs_present = all(
        token in training_text
        for token in (
            "epoch_metrics",
            "epochs_completed",
            "demo_train_log.json",
            "demo_training_curve.csv",
            "training_curve.png",
            "demo_last.ckpt",
            "demo_eval_summary.json",
        )
    )

    bad_claim_hits = [line for line in (notebook_text + "\n" + readme_text).splitlines() if _line_has_bad_claim(line)]
    no_bad_claims = not bad_claim_hits

    report: dict[str, Any] = {
        "package_root_ok": (project_root / "src" / "train.py").exists()
        and (project_root / "src" / "onsite_demo").exists(),
        "required_files_present": required_files_present,
        "data_subset_present": data_subset_present,
        "official_attack_assets_present": formal_results_present,
        "official_detection_assets_present": formal_results_present,
        "cloud_live_config_ok": cloud_live_config_ok,
        "cloud_live_epochs_ge_10": int(mode_settings["epochs"]) >= 10,
        "notebook_no_local_absolute_paths": notebook_no_local_absolute_paths,
        "notebook_no_windows_path_hardcoding": notebook_no_windows_path_hardcoding,
        "notebook_no_broken_text": notebook_no_broken_text,
        "notebook_sections_complete": notebook_sections_complete,
        "notebook_text_concise": notebook_text_concise,
        "ipywidgets_primary_present": ipywidgets_primary_present,
        "ipywidgets_fallback_present": ipywidgets_fallback_present,
        "attack_uses_live_checkpoint_path": attack_uses_live_checkpoint_path,
        "detection_uses_live_checkpoint_path": detection_uses_live_checkpoint_path,
        "formal_results_loaded": formal_results_present,
        "training_output_logic_present": training_outputs_present,
        "imports_ok": imports_ok,
        "candidate_cloud_live_zip_openable": candidate_zip_ok,
        "no_bad_claims": no_bad_claims,
        "local_10epoch_not_required": True,
        "details": {
            "project_root": str(project_root),
            "train_class_count": train_class_count,
            "test_class_count": test_class_count,
            "train_min_per_class": train_min_per_class,
            "test_min_per_class": test_min_per_class,
            "cloud_live_mode_settings": mode_settings,
            "import_errors": import_errors,
            "candidate_zip_detail": candidate_zip_detail,
            "bad_claim_sample": bad_claim_hits[0] if bad_claim_hits else "",
        },
    }
    report["final_preupload_audit_pass"] = all(
        bool(report[key])
        for key in (
            "package_root_ok",
            "required_files_present",
            "data_subset_present",
            "official_attack_assets_present",
            "official_detection_assets_present",
            "cloud_live_config_ok",
            "cloud_live_epochs_ge_10",
            "notebook_no_local_absolute_paths",
            "notebook_no_windows_path_hardcoding",
            "notebook_no_broken_text",
            "notebook_sections_complete",
            "notebook_text_concise",
            "ipywidgets_primary_present",
            "ipywidgets_fallback_present",
            "attack_uses_live_checkpoint_path",
            "detection_uses_live_checkpoint_path",
            "formal_results_loaded",
            "training_output_logic_present",
            "imports_ok",
            "candidate_cloud_live_zip_openable",
            "no_bad_claims",
            "local_10epoch_not_required",
        )
    )

    json_path = output_dir / "FINAL_PREUPLOAD_AUDIT.json"
    md_path = output_dir / "FINAL_PREUPLOAD_AUDIT.md"
    save_json(report, json_path)
    _write_markdown(report, md_path)

    for key, value in report.items():
        if key == "details":
            continue
        print(f"{key} = {value}")

    print(f"FINAL_PREUPLOAD_AUDIT.json = {json_path}")
    print(f"FINAL_PREUPLOAD_AUDIT.md = {md_path}")

    if not report["final_preupload_audit_pass"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(exc)
        print(traceback.format_exc(limit=4))
        raise

#!/usr/bin/env python3
"""Validate the experiment-3 double-buffer schedule and NPU result."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


RESULT_POINTS = 30.0
MAX_ABS_ERROR = 1.0e-6
BUILD_TIMEOUT_SECONDS = 600
CASE_TIMEOUT_SECONDS = 120
METRIC_RE = re.compile(r"([a-z_]+)=([^\s]+)")


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run(
    command: list[str],
    cwd: Path,
    *,
    timeout: int,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {command_text(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output += f"\n[TIMEOUT after {timeout}s]\n"
        result = subprocess.CompletedProcess(command, 124, output)
    print(result.stdout)
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}")
    return result


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def prepare_build_dir(build_dir: Path, project: Path) -> None:
    allowed_roots = {
        Path("/tmp").resolve(),
        Path(tempfile.gettempdir()).resolve(),
        project.parent.resolve(),
    }
    if build_dir == project or not any(
        build_dir != root and is_relative_to(build_dir, root) for root in allowed_roots
    ):
        raise ValueError(f"refusing to recreate unsafe build directory: {build_dir}")
    if build_dir.exists():
        if build_dir.is_symlink():
            raise ValueError(f"refusing to remove symlinked build directory: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)


def locate_executable(build_dir: Path) -> Path:
    names = {"vector_add_pipeline_student", "vector_add_pipeline_student.exe"}
    matches = [
        path for path in build_dir.rglob("*") if path.is_file() and path.name in names
    ]
    if not matches:
        raise FileNotFoundError("vector_add_pipeline_student executable was not produced")
    return matches[0]


def case_command(
    executable: Path,
    case: dict[str, Any],
    *,
    iterations: int,
) -> list[str]:
    return [
        str(executable),
        "--length",
        str(case["length"]),
        "--block-dim",
        str(case["block_dim"]),
        "--tile-count",
        str(case["tile_count"]),
        "--seed",
        str(case.get("seed", 1)),
        "--warmup",
        "1",
        "--iterations",
        str(iterations),
    ]


def parse_metric(output: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith("METRIC ")]
    return dict(METRIC_RE.findall(lines[-1])) if lines else {}


def metric_is_valid(metric: dict[str, str], case: dict[str, Any]) -> bool:
    tile_length = (
        int(case["length"]) // int(case["block_dim"]) // int(case["tile_count"])
    )
    tile_bytes = tile_length * 4
    expected = {
        "buffer_num": "2",
        "queue_depth": "1",
        "schedule": "prefetch",
        "length": str(case["length"]),
        "block_dim": str(case["block_dim"]),
        "tile_count": str(case["tile_count"]),
        "tile_length": str(tile_length),
        "tile_bytes": str(tile_bytes),
        "queue_bytes": str(3 * 2 * tile_bytes),
        "queue_scope": "per_block",
        "correctness": "PASS",
        "timing_scope": "launch_plus_sync",
    }
    try:
        max_abs_error = float(metric.get("max_abs_error", "nan"))
        average_us = float(metric.get("avg_kernel_us", "nan"))
        bandwidth = float(metric.get("effective_gb_s", "nan"))
    except ValueError:
        return False
    return (
        all(metric.get(key) == value for key, value in expected.items())
        and math.isfinite(max_abs_error)
        and 0.0 <= max_abs_error <= MAX_ABS_ERROR
        and math.isfinite(average_us)
        and average_us > 0.0
        and math.isfinite(bandwidth)
        and bandwidth > 0.0
    )


def run_schedule_gate(
    project: Path,
    build_dir: Path,
    schedule_source: Path,
) -> bool:
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        print("[GATE FAIL] no host C++ compiler was found")
        return False
    executable = build_dir / "pipeline_schedule_test"
    if sys.platform.startswith("win"):
        executable = executable.with_suffix(".exe")
    try:
        run(
            [
                compiler,
                "-std=c++17",
                "-I",
                str(project),
                str(schedule_source),
                "-o",
                str(executable),
            ],
            project,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        run([str(executable)], project, timeout=CASE_TIMEOUT_SECONDS)
    except Exception as exc:
        print(f"[GATE FAIL] double-buffer schedule: {exc}")
        return False
    print("[GATE PASS] double-buffer schedule covers tile_count=1, 2 and 7")
    return True


def validate_invalid_cases(
    executable: Path,
    project: Path,
    cases: list[dict[str, Any]],
) -> bool:
    if not cases:
        print("[GATE FAIL] no invalid cases were supplied")
        return False
    all_passed = True
    for index, case in enumerate(cases, start=1):
        result = run(
            case_command(executable, case, iterations=1),
            project,
            timeout=CASE_TIMEOUT_SECONDS,
            expect_success=False,
        )
        expected_code = int(case["expected_exit_code"])
        expected_error = str(case["expected_error"])
        passed = (
            result.returncode == expected_code
            and expected_error in result.stdout
            and "METRIC " not in result.stdout
        )
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(
            f"[GATE {status}] invalid case #{index}: "
            f"expected exit={expected_code}, error contains {expected_error!r}"
        )
    return all_passed


def format_score(value: float) -> str:
    if abs(value - round(value)) < 1.0e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def emit_score(score: float) -> None:
    bounded = max(0.0, min(RESULT_POINTS, score))
    print(f"AUTO_RESULT_SCORE {format_score(bounded)}/30")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--schedule-test", type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    build_dir = args.build_dir.resolve()
    source_root = Path(__file__).resolve().parents[1]
    cases_path = (args.cases or source_root / "tests/pipeline_cases.json").resolve()
    schedule_source = (
        args.schedule_test or source_root / "tests/pipeline_schedule_test.cpp"
    ).resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    try:
        prepare_build_dir(build_dir, project)
    except Exception as exc:
        print(f"[GATE FAIL] build directory: {exc}")
        emit_score(0.0)
        return 1

    if not run_schedule_gate(project, build_dir, schedule_source):
        emit_score(0.0)
        return 1

    try:
        run(
            ["cmake", "-S", str(project), "-B", str(build_dir)],
            project,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        run(
            ["cmake", "--build", str(build_dir), "-j"],
            project,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        executable = locate_executable(build_dir)
        print("[GATE PASS] clean Ascend C build")
    except Exception as exc:
        print(f"[GATE FAIL] Ascend C build: {exc}")
        emit_score(0.0)
        return 1

    invalid_passed = validate_invalid_cases(
        executable, project, list(cases.get("invalid", []))
    )
    if not invalid_passed:
        emit_score(0.0)
        return 1

    valid_cases = list(cases.get("valid", []))
    if not valid_cases:
        print("[FAIL] no valid precision cases were supplied")
        emit_score(0.0)
        return 1

    passed_count = 0
    for index, case in enumerate(valid_cases, start=1):
        result = run(
            case_command(executable, case, iterations=5),
            project,
            timeout=CASE_TIMEOUT_SECONDS,
            expect_success=False,
        )
        passed = result.returncode == 0 and metric_is_valid(
            parse_metric(result.stdout), case
        )
        passed_count += int(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] precision case #{index}")

    score = RESULT_POINTS * passed_count / len(valid_cases)
    emit_score(score)
    return 0 if passed_count == len(valid_cases) else 1


if __name__ == "__main__":
    sys.exit(main())

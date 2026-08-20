#!/usr/bin/env python3
"""Build the vector-add exercise and award only the 30-point result score."""

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
METRIC_RE = re.compile(r"([a-z_]+)=([^\s]+)")
MAX_ABS_ERROR = 1.0e-6
BUILD_TIMEOUT_SECONDS = 600
CASE_TIMEOUT_SECONDS = 120
TIMEOUT_RETURN_CODE = 124


def command_text(command: list[str]) -> str:
    """Render a command with quoting suitable for the current host."""

    return subprocess.list2cmdline(command)


def run(
    command: list[str],
    cwd: Path,
    *,
    timeout: int,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded command and turn a timeout into a gradeable result."""

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
        result = subprocess.CompletedProcess(command, TIMEOUT_RETURN_CODE, output)

    print(result.stdout)
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}")
    return result


def is_relative_to(path: Path, root: Path) -> bool:
    """Backport Path.is_relative_to so the grader also runs on Python 3.8."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def is_strictly_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is below, but is not equal to, ``root``."""

    return path != root and is_relative_to(path, root)


def prepare_build_dir(build_dir: Path, project: Path) -> None:
    """Safely recreate a build tree only under a temporary or project parent tree."""

    allowed_roots = {
        Path("/tmp").resolve(),
        Path(tempfile.gettempdir()).resolve(),
        project.parent.resolve(),
    }
    if build_dir == project or not any(
        is_strictly_within(build_dir, root) for root in allowed_roots
    ):
        roots = ", ".join(sorted(str(root) for root in allowed_roots))
        raise ValueError(
            f"refusing to recreate unsafe build directory {build_dir}; "
            f"choose a child of one of: {roots}"
        )
    if build_dir.exists():
        if build_dir.is_symlink():
            raise ValueError(f"refusing to remove symlinked build directory: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)


def locate_executable(build_dir: Path) -> Path:
    """Find the executable emitted by the student practice project."""

    candidates = [p for p in build_dir.rglob("vector_add_student") if p.is_file()]
    if not candidates:
        candidates = [p for p in build_dir.rglob("vector_add_student.exe") if p.is_file()]
    if not candidates:
        raise FileNotFoundError("vector_add_student executable was not produced")
    return candidates[0]


def case_command(
    executable: Path,
    case: dict[str, Any],
    *,
    iterations: int,
) -> list[str]:
    """Create the common command line used by positive and negative cases."""

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
    """Parse the final machine-readable METRIC line emitted by the host program."""

    lines = [line for line in output.splitlines() if line.startswith("METRIC ")]
    if not lines:
        return {}
    return dict(METRIC_RE.findall(lines[-1]))


def metric_is_valid(metric: dict[str, str], case: dict[str, Any]) -> bool:
    """Verify the fixed experiment contract instead of trusting a PASS substring."""

    expected_tile_length = (
        int(case["length"]) // int(case["block_dim"]) // int(case["tile_count"])
    )
    expected = {
        "buffer_num": "1",
        "compute_repeat": "1",
        "length": str(case["length"]),
        "block_dim": str(case["block_dim"]),
        "tile_count": str(case["tile_count"]),
        "tile_length": str(expected_tile_length),
        "queue_bytes": str(3 * expected_tile_length * 4),
        "correctness": "PASS",
        "timing_scope": "launch_plus_sync",
    }
    try:
        max_abs_error = float(metric.get("max_abs_error", "nan"))
    except ValueError:
        return False
    return (
        all(metric.get(key) == value for key, value in expected.items())
        and math.isfinite(max_abs_error)
        and 0.0 <= max_abs_error <= MAX_ABS_ERROR
    )


def validate_invalid_cases(
    executable: Path,
    project: Path,
    invalid_cases: list[dict[str, Any]],
) -> bool:
    """Check every malformed Shape against its exact exit code and error fragment."""

    all_passed = True
    if not invalid_cases:
        print("[GATE FAIL] no invalid Shape cases were supplied")
        return False

    for index, case in enumerate(invalid_cases, start=1):
        expected_code = int(case["expected_exit_code"])
        expected_error = str(case["expected_error"])
        result = run(
            case_command(executable, case, iterations=1),
            project,
            timeout=CASE_TIMEOUT_SECONDS,
            expect_success=False,
        )
        passed = (
            result.returncode == expected_code
            and expected_error in result.stdout
            and "METRIC " not in result.stdout
        )
        all_passed &= passed
        if passed:
            print(
                f"[GATE PASS] invalid Shape #{index}: exit={expected_code}, "
                f"error contains {expected_error!r}"
            )
        else:
            print(
                f"[GATE FAIL] invalid Shape #{index}: expected exit={expected_code} "
                f"and error containing {expected_error!r}; got exit={result.returncode}"
            )
    return all_passed


def format_score(value: float) -> str:
    """Print whole points compactly and partial group points to two decimals."""

    if abs(value - round(value)) < 1.0e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def emit_score(score: float) -> None:
    """Emit the stable machine-readable result-score contract."""

    bounded = max(0.0, min(RESULT_POINTS, score))
    print(f"AUTO_RESULT_SCORE {format_score(bounded)}/30")


def function_return_uses(code: str, function_name: str, required_names: tuple[str, ...]) -> bool:
    """Check that a small student hook returns an expression using all required names."""

    match = re.search(
        rf"\b{re.escape(function_name)}\s*\([^)]*\)\s*\{{(.*?)\}}",
        code,
        flags=re.DOTALL,
    )
    if not match:
        return False
    return_match = re.search(r"\breturn\s+([^;]+);", match.group(1), flags=re.DOTALL)
    if not return_match:
        return False
    expression = return_match.group(1)
    return "*" in expression and all(
        re.search(rf"\b{re.escape(name)}\b", expression) for name in required_names
    )


def inspect_student_source(header: Path) -> bool:
    """Evaluate all three vector-lab process hooks without assigning result points."""

    if not header.is_file():
        print("[GATE FAIL] student_compute.h is missing")
        return False
    source = header.read_text(encoding="utf-8")
    code = re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.MULTILINE | re.DOTALL)
    checks = {
        "StudentBlockOffset uses blockLength and blockIdx": function_return_uses(
            code, "StudentBlockOffset", ("blockLength", "blockIdx")
        ),
        "StudentTileOffset uses progress and tileLength": function_return_uses(
            code, "StudentTileOffset", ("progress", "tileLength")
        ),
        "STUDENT_COMPUTE uses AscendC::Add": bool(
            re.search(
                r"#\s*define\s+STUDENT_COMPUTE\s*\([^)]*\)\s+AscendC::Add\s*\(",
                code,
            )
        ),
    }
    for description, passed in checks.items():
        print(f"[GATE {'PASS' if passed else 'FAIL'}] {description}")
    return all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True, help="student practice project")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help="JSON test cases; defaults to chapter src/tests")
    args = parser.parse_args()

    project = args.project.resolve()
    build_dir = args.build_dir.resolve()
    cases_path = (
        args.cases
        or (Path(__file__).resolve().parents[1] / "tests/vector_add_cases.json")
    ).resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    valid_cases = list(cases.get("valid", []))
    invalid_cases = list(cases.get("invalid", []))

    # Source inspection is a process gate; automatic points come only from NPU accuracy.
    header = project / "student_compute.h"
    source_gate = inspect_student_source(header)

    # A clean, bounded build is another process gate and never contributes result points.
    configure_command = ["cmake", "-S", str(project), "-B", str(build_dir)]
    build_command = ["cmake", "--build", str(build_dir), "-j"]
    try:
        prepare_build_dir(build_dir, project)
        run(configure_command, project, timeout=BUILD_TIMEOUT_SECONDS)
        run(build_command, project, timeout=BUILD_TIMEOUT_SECONDS)
        executable = locate_executable(build_dir)
        print("[GATE PASS] clean build")
    except Exception as exc:
        print(f"[GATE FAIL] build: {exc}")
        emit_score(0.0)
        return 1

    invalid_gate = validate_invalid_cases(executable, project, invalid_cases)
    process_gates_passed = source_gate and invalid_gate
    if not process_gates_passed:
        print("[INFO] precision cases skipped because a process gate failed")
        emit_score(0.0)
        return 1

    if not valid_cases:
        print("[FAIL] no valid precision cases were supplied")
        emit_score(0.0)
        return 1

    # The complete (possibly hidden) precision group always shares exactly 30 points.
    passed_count = 0
    points_per_case = RESULT_POINTS / len(valid_cases)
    for index, case in enumerate(valid_cases, start=1):
        result = run(
            case_command(executable, case, iterations=5),
            project,
            timeout=CASE_TIMEOUT_SECONDS,
            expect_success=False,
        )
        metric = parse_metric(result.stdout)
        passed = result.returncode == 0 and metric_is_valid(metric, case)
        if passed:
            passed_count += 1
            print(
                f"[PASS] precision case #{index} "
                f"({format_score(points_per_case)} of group-total 30 points)"
            )
        else:
            print(f"[FAIL] precision case #{index} (0 points)")

    result_score = RESULT_POINTS * passed_count / len(valid_cases)
    emit_score(result_score)
    return 0 if passed_count == len(valid_cases) else 1


if __name__ == "__main__":
    sys.exit(main())

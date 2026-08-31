#!/usr/bin/env python3
"""Build and grade Experiment 11 without using performance thresholds."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


RESULT_POINTS = 30.0
METRIC_RE = re.compile(r"([a-z_]+)=([^\s]+)")
POSITIVE_CASES: tuple[dict[str, int | str], ...] = (
    {"layout": "dense", "batch": 1, "universe": 31, "seed": 1},
    {"layout": "dense", "batch": 2, "universe": 32, "seed": 2},
    {"layout": "dense", "batch": 3, "universe": 33, "seed": 3},
    {"layout": "dense", "batch": 257, "universe": 4097, "seed": 4},
    {"layout": "bitmap", "batch": 1, "universe": 31, "seed": 5},
    {"layout": "bitmap", "batch": 2, "universe": 32, "seed": 6},
    {"layout": "bitmap", "batch": 3, "universe": 33, "seed": 7},
    {"layout": "bitmap", "batch": 32, "universe": 1000003, "seed": 8},
)
INVALID_CASES: tuple[tuple[str, ...], ...] = (
    ("--layout", "dense", "--batch", "0", "--universe", "32"),
    ("--layout", "bitmap", "--batch", "1", "--universe", "0"),
    ("--layout", "unknown", "--batch", "1", "--universe", "32"),
    ("--layout", "bitmap", "--iterations", "0"),
)


def run(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    print("$", subprocess.list2cmdline(command))
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
        result = subprocess.CompletedProcess(command, 124, output + "\n[TIMEOUT]\n")
    print(result.stdout)
    return result


def parse_metric(output: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith("METRIC ")]
    return dict(METRIC_RE.findall(lines[-1])) if lines else {}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def expected_contract(case: dict[str, int | str]) -> dict[str, str]:
    batch = int(case["batch"])
    universe = int(case["universe"])
    layout = str(case["layout"])
    if layout == "dense":
        valid_units = universe
        physical_units = align_up(universe, 32)
        element_bytes = 1
    else:
        valid_units = (universe + 31) // 32
        physical_units = align_up(valid_units, 8)
        element_bytes = 4
    return {
        "layout": layout,
        "batch": str(batch),
        "universe": str(universe),
        "valid_units_per_row": str(valid_units),
        "physical_units_per_row": str(physical_units),
        "physical_bytes": str(3 * batch * physical_units * element_bytes),
        "logical_bytes": str(3 * batch * universe),
        "buffer_num": "1",
        "correctness": "PASS",
        "guard": "PASS",
        "timing_scope": "launch_plus_sync",
    }


def source_is_complete(header: Path) -> bool:
    if not header.is_file():
        print("[GATE FAIL] student_compute.h is missing")
        return False
    source = header.read_text(encoding="utf-8")
    uncommented = re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.MULTILINE | re.DOTALL)
    checks = {
        "TODO markers removed": "TODO" not in uncommented,
        "block offset uses both factors": bool(
            re.search(r"return\s+blockFormer\s*\*\s*blockIdx\s*;", uncommented)
            or re.search(r"return\s+blockIdx\s*\*\s*blockFormer\s*;", uncommented)
        ),
        "tail branch uses block index and count": all(
            name in uncommented for name in ("blockIdx", "blockNum", "blockTail", "blockFormer")
        ) and "?" in uncommented,
        "compute uses both inputs": bool(
            re.search(r"ComputeSetAnd\s*\(\s*\(z\)\s*,\s*\(x\)\s*,\s*\(y\)", uncommented)
        ),
    }
    for name, passed in checks.items():
        print(f"[{'GATE PASS' if passed else 'GATE FAIL'}] {name}")
    return all(checks.values())


def locate_executable(build_dir: Path) -> Path:
    for name in ("bitmap_and_student", "bitmap_and_student.exe"):
        matches = [path for path in build_dir.rglob(name) if path.is_file()]
        if matches:
            return matches[0]
    raise FileNotFoundError("bitmap_and_student executable was not produced")


def case_command(executable: Path, case: dict[str, int | str]) -> list[str]:
    return [
        str(executable),
        "--layout", str(case["layout"]),
        "--batch", str(case["batch"]),
        "--universe", str(case["universe"]),
        "--seed", str(case.get("seed", 1)),
        "--warmup", "1",
        "--iterations", "1",
    ]


def emit_score(score: float) -> None:
    bounded = max(0.0, min(RESULT_POINTS, score))
    print(f"AUTO_RESULT_SCORE {bounded:.2f}/30")


def main() -> int:
    parser = argparse.ArgumentParser()
    chapter = Path(__file__).resolve().parents[2]
    parser.add_argument("--project", type=Path, default=chapter / "src" / "practice")
    parser.add_argument("--npu-arch", default="dav-2201")
    args = parser.parse_args()

    project = args.project.resolve()
    if not source_is_complete(project / "student_compute.h"):
        emit_score(0.0)
        return 1

    with tempfile.TemporaryDirectory(prefix="bitmap-and-grade-") as temp_dir:
        work = Path(temp_dir)
        shutil.copytree(project.parent / "demo", work / "demo")
        shutil.copytree(project, work / "practice")
        build = work / "build"
        configure = run(
            ["cmake", "-S", str(work / "practice"), "-B", str(build),
             f"-DNPU_ARCH={args.npu_arch}"],
            work,
            120,
        )
        if configure.returncode != 0:
            emit_score(0.0)
            return 1
        compiled = run(["cmake", "--build", str(build), "-j2"], work, 600)
        if compiled.returncode != 0:
            emit_score(0.0)
            return 1
        executable = locate_executable(build)

        invalid_ok = True
        for case_args in INVALID_CASES:
            result = run([str(executable), *case_args], work, 30)
            passed = result.returncode == 2 and "METRIC " not in result.stdout
            invalid_ok &= passed
            print(f"[{'GATE PASS' if passed else 'GATE FAIL'}] invalid arguments")
        if not invalid_ok:
            emit_score(0.0)
            return 1

        positives = POSITIVE_CASES
        passed_count = 0
        for index, case in enumerate(positives, start=1):
            result = run(case_command(executable, case), work, 120)
            metric = parse_metric(result.stdout)
            expected = expected_contract(case)
            passed = result.returncode == 0 and all(
                metric.get(key) == value for key, value in expected.items()
            )
            passed_count += int(passed)
            print(f"[{'PASS' if passed else 'FAIL'}] case {index}: {case}")

        score = RESULT_POINTS * passed_count / len(positives) if positives else 0.0
        emit_score(score)
        return 0 if passed_count == len(positives) and positives else 1


if __name__ == "__main__":
    raise SystemExit(main())

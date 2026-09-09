#!/usr/bin/env python3
"""Summarize common MSPROF CSV outputs for the YOLO single-NPU tuning lab."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def find_csvs(profile_dir: Path) -> list[Path]:
    return sorted(profile_dir.rglob("*.csv"))


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        with path.open("r", encoding="gbk", newline="") as f:
            return list(csv.DictReader(f))


def first_float(row: dict[str, str], names: tuple[str, ...]) -> float:
    for name in names:
        if name in row and row[name]:
            try:
                return float(str(row[name]).replace(",", ""))
            except ValueError:
                continue
    return 0.0


def summarize_task_time(path: Path) -> None:
    rows = read_rows(path)
    if not rows:
        return
    candidates = []
    for row in rows:
        duration = first_float(row, ("Task Duration(us)", "Duration(us)", "aicore_time(us)", "execution_time"))
        name = row.get("Op Name") or row.get("Name") or row.get("Task Type") or "unknown"
        if duration > 0:
            candidates.append((duration, name))
    if not candidates:
        return
    candidates.sort(reverse=True)
    print(f"\n[Task time] {path}")
    for duration, name in candidates[:10]:
        print(f"  {duration:12.1f} us  {name}")


def summarize_op_summary(path: Path) -> None:
    rows = read_rows(path)
    if not rows:
        return
    candidates = []
    for row in rows:
        total = first_float(row, ("Total Time(us)", "Total Time", "Duration(us)", "time(us)"))
        name = row.get("Op Name") or row.get("Name") or row.get("op_name") or "unknown"
        calls = row.get("Calls") or row.get("Call Times") or row.get("count") or ""
        if total > 0:
            candidates.append((total, calls, name))
    if not candidates:
        return
    candidates.sort(reverse=True)
    print(f"\n[Op summary] {path}")
    for total, calls, name in candidates[:10]:
        print(f"  {total:12.1f} us  calls={calls:>6}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", default="profiles/yolo_single_npu")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        raise SystemExit(f"profile directory not found: {profile_dir}")

    csvs = find_csvs(profile_dir)
    print(f"Found {len(csvs)} CSV files under {profile_dir.resolve()}")
    if not csvs:
        print("Run src/scripts/profile_msprof.sh first, then analyze the generated profile directory.")
        return

    matched = False
    for path in csvs:
        lower = path.name.lower()
        if "task" in lower and "time" in lower:
            summarize_task_time(path)
            matched = True
        elif "op" in lower and ("summary" in lower or "statistic" in lower):
            summarize_op_summary(path)
            matched = True

    if not matched:
        print("No standard task_time/op_summary CSV was detected.")
        print("Open the MSPROF timeline and focus on DataLoader gaps, Host runtime APIs, and long AI Core kernels.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def decimal_field(row: dict[str, str], field: str, csv_file: Path) -> Decimal:
    try:
        return Decimal(row[field].strip())
    except (InvalidOperation, KeyError, AttributeError) as exc:
        raise ValueError(f"Invalid {field!r} in {csv_file}") from exc


def read_mstx_device_range(
    csv_file: Path,
    message: str,
) -> tuple[Decimal, Decimal]:
    with csv_file.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("message") != message:
                continue
            return (
                decimal_field(row, "Device Start_time(us)", csv_file),
                decimal_field(row, "Device End_time(us)", csv_file),
            )
    raise ValueError(f"Cannot find MSTX range {message!r} in {csv_file}")


def find_profile_result(
    profile_root: Path,
    message: str,
) -> tuple[Path, Decimal]:
    tx_files = sorted(
        profile_root.rglob("msprof_tx_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for tx_file in tx_files:
        try:
            start, end = read_mstx_device_range(tx_file, message)
        except ValueError:
            continue
        return tx_file, end - start
    raise FileNotFoundError(f"Cannot find matching MSTX data under {profile_root}")


def format_result(
    case_name: str,
    timing_file: Path,
    duration_us: Decimal,
    baseline_duration_us: Decimal | None = None,
) -> list[str]:
    duration_ms = duration_us / 1000
    if baseline_duration_us is None:
        if case_name == "aclgraph+sk":
            improvement = "N/A (aclgraph profiling result not found)"
        else:
            improvement = "N/A (run aclgraph+sk to compare)"
    else:
        if baseline_duration_us <= 0:
            raise ValueError("aclgraph duration must be positive")
        baseline_ms = baseline_duration_us / 1000
        improvement_percent = (
            (baseline_duration_us - duration_us) / baseline_duration_us * 100
        )
        improvement = (
            f"{improvement_percent:.3f}% "
            f"(aclgraph {baseline_ms:.3f} ms -> aclgraph+sk {duration_ms:.3f} ms)"
        )

    return [
        "========== Performance Result ==========",
        f"Case name: {case_name}",
        f"Timing file: {timing_file}",
        f"Device duration: {duration_ms:.3f} ms",
        f"SK improvement: {improvement}",
        "========================================",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an SK msprof run.")
    parser.add_argument("profile_root", type=Path)
    parser.add_argument(
        "--case-name", required=True, choices=["aclgraph", "aclgraph+sk"]
    )
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--message", default="custom_sk_model_steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timing_file, duration_us = find_profile_result(args.profile_root, args.message)
    baseline_duration_us = None
    if args.case_name == "aclgraph+sk" and args.baseline_root is not None:
        try:
            _, baseline_duration_us = find_profile_result(
                args.baseline_root, args.message
            )
        except FileNotFoundError:
            pass

    print(
        "\n".join(
            format_result(
                args.case_name,
                timing_file,
                duration_us,
                baseline_duration_us,
            )
        )
    )


if __name__ == "__main__":
    main()

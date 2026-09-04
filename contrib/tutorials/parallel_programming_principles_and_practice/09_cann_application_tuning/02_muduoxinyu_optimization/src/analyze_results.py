#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Compare MuduoXinyu multi-operator and FlashAttention logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import re
import statistics
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

MARKER_A = "path=decomposed_fp32 dtype=fp32 api=multi_op fallback=0"
MARKER_B = "path=flash_attention_v4 dtype=fp16 api=aclnnIncreFlashAttentionV4 fallback=0"
FLASH_STATS_RE = re.compile(
    r"flash_attention_calls=(\d+)\s+flash_cast_calls=(\d+)\s+"
    r"flash_fallback_count=(\d+)\s+flash_sync_count=(\d+)"
)
TRACE_RE = re.compile(r"^\[DECODE_GENERATED_TOKENS\] (.*)$", re.MULTILINE)
THROUGHPUT_RE = re.compile(
    r"Sample (\d+)(?: \([^)]*\))?: (\d+) tokens in (\d+) ms, "
    r"throughput = ([0-9.eE+-]+) tokens/s"
)
ERROR_RE = re.compile(r"\b(bad_alloc|ERROR|FATAL)\b")


@dataclass(frozen=True)
class PerformanceRun:
    text: str
    samples: list[dict]
    traces: list[list[int]]
    mean: float | None
    cv: float | None


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_trace(match: re.Match) -> list[int]:
    return [int(token.strip()) for token in match.group(1).split(",")]


def traces(text: str) -> list[list[int]]:
    return [parse_trace(match) for match in TRACE_RE.finditer(text)]


def samples(text: str) -> list[dict]:
    return [
        {
            "sample": int(match.group(1)),
            "tokens": int(match.group(2)),
            "time_ms": int(match.group(3)),
            "tokens_per_s": float(match.group(4)),
        }
        for match in THROUGHPUT_RE.finditer(text)
    ]


def flash_stats(text: str) -> dict | None:
    matches = list(FLASH_STATS_RE.finditer(text))
    if not matches:
        return None
    values = map(int, matches[-1].groups())
    return dict(zip(
        ("flash_attention_calls", "flash_cast_calls",
         "flash_fallback_count", "flash_sync_count"),
        values,
    ))


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) != 3 or not all(math.isfinite(value) and value > 0 for value in values):
        return None, None
    mean = statistics.fmean(values)
    return mean, statistics.pstdev(values) / mean * 100.0


def analyze_smoke(args: argparse.Namespace, smoke_a: str, smoke_b: str) -> dict:
    """Check the functional A/B run."""
    expected_calls = args.smoke_steps * args.num_layers
    expected_stats = {
        "flash_attention_calls": expected_calls,
        "flash_cast_calls": expected_calls * 4,
        "flash_fallback_count": 0,
        "flash_sync_count": 0,
    }
    actual_stats = flash_stats(smoke_b)
    smoke_traces_a, smoke_traces_b = traces(smoke_a), traces(smoke_b)
    smoke_checks = {
        "path_a_marker": MARKER_A in smoke_a and MARKER_B not in smoke_a,
        "path_b_marker": MARKER_B in smoke_b,
        "flash_stats": actual_stats == expected_stats,
        "token_exact": bool(smoke_traces_a)
        and smoke_traces_a == smoke_traces_b,
        "no_error": not ERROR_RE.search(smoke_a + smoke_b),
    }
    return {
        "checks": smoke_checks,
        "expected_flash_stats": expected_stats,
        "flash_stats": actual_stats,
        "functional_pass": all(smoke_checks.values()),
    }


def measured_throughput(items: list[dict]) -> list[float]:
    return [item["tokens_per_s"] for item in items if item["sample"] >= 2]


def performance_run(text: str) -> PerformanceRun:
    parsed_samples = samples(text)
    mean, cv = summarize(measured_throughput(parsed_samples))
    return PerformanceRun(text, parsed_samples, traces(text), mean, cv)


def performance_checks(path_a: PerformanceRun, path_b: PerformanceRun) -> dict:
    expected_ids = [1, 2, 3, 4]
    return {
        "samples_1_to_4": [item["sample"] for item in path_a.samples] == expected_ids
        and [item["sample"] for item in path_b.samples] == expected_ids,
        "three_measured_samples": path_a.mean is not None and path_b.mean is not None,
        "token_exact": len(path_a.traces) == 4 and path_a.traces == path_b.traces,
        "no_error": not ERROR_RE.search(path_a.text + path_b.text),
    }


def percentage_delta(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    return (candidate - reference) / reference * 100.0


def analyze_performance(perf_a: str, perf_b: str) -> dict:
    """Check the measured A/B run and summarize throughput."""
    path_a = performance_run(perf_a)
    path_b = performance_run(perf_b)
    perf_checks = performance_checks(path_a, path_b)
    performance_valid = all(perf_checks.values())
    delta = percentage_delta(path_a.mean, path_b.mean)
    return {
        "checks": perf_checks,
        "path_a": {
            "samples": path_a.samples,
            "mean_tokens_per_s": path_a.mean,
            "cv_percent": path_a.cv,
        },
        "path_b": {
            "samples": path_b.samples,
            "mean_tokens_per_s": path_b.mean,
            "cv_percent": path_b.cv,
        },
        "delta_percent_b_vs_a": delta,
        "performance_valid": performance_valid,
    }


def analyze(args: argparse.Namespace) -> dict:
    smoke = analyze_smoke(args, read(args.smoke_a), read(args.smoke_b))
    perf = analyze_performance(read(args.perf_a), read(args.perf_b))
    functional_pass = smoke["functional_pass"]
    performance_valid = perf["performance_valid"]
    delta = perf["delta_percent_b_vs_a"]
    return {
        "smoke": smoke,
        "perf": perf,
        "verdict": {
            "functional_pass": functional_pass,
            "performance_valid": performance_valid,
            "overall_pass": functional_pass and performance_valid,
            "performance_beneficial": bool(
                functional_pass and performance_valid and delta is not None and delta > 0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-a", type=Path, required=True)
    parser.add_argument("--smoke-b", type=Path, required=True)
    parser.add_argument("--perf-a", type=Path, required=True)
    parser.add_argument("--perf-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=12)
    args = parser.parse_args()

    result = analyze(args)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verdict = result["verdict"]
    perf = result["perf"]
    LOGGER.info("FUNCTIONAL_PASS=%s", verdict["functional_pass"])
    LOGGER.info("PERFORMANCE_VALID=%s", verdict["performance_valid"])
    LOGGER.info("PERFORMANCE_BENEFICIAL=%s", verdict["performance_beneficial"])
    LOGGER.info("MEAN_A=%s", perf["path_a"]["mean_tokens_per_s"])
    LOGGER.info("MEAN_B=%s", perf["path_b"]["mean_tokens_per_s"])
    LOGGER.info("DELTA_PERCENT_B_VS_A=%s", perf["delta_percent_b_vs_a"])
    return 0 if verdict["overall_pass"] else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    raise SystemExit(main())

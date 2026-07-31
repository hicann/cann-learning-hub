"""Shared timing, cross-rank aggregation, and profiling helpers for chapter 07."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu


def percentile(samples: list[float], q: float) -> float:
    """Return a linearly interpolated percentile without a sample-count restriction."""
    if not samples:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(samples)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def measure_samples_ms(
    op: Callable[[], None], warmup: int, iters: int
) -> list[float]:
    """Measure synchronized wall time and retain every rank-local iteration."""
    for _ in range(warmup):
        op()
    torch.npu.synchronize()
    dist.barrier()

    samples_ms: list[float] = []
    for _ in range(iters):
        dist.barrier()
        start = time.perf_counter()
        op()
        torch.npu.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1_000)
    dist.barrier()
    return samples_ms


def summarize_across_ranks(
    name: str,
    samples_ms: list[float],
    *,
    device: torch.device,
    logical_bytes: int,
    sent_bytes_per_rank: float,
) -> dict[str, Any] | None:
    """Gather aligned iterations and summarize rank-local and slow-rank latency."""
    # HCCL support for float32 is broader than float64; millisecond samples do not
    # need double precision for the reported three decimal places.
    local = torch.tensor(samples_ms, dtype=torch.float32, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    if dist.get_rank() != 0:
        return None

    all_rank_samples = [tensor.cpu().tolist() for tensor in gathered]
    rank_stats = []
    for rank, samples in enumerate(all_rank_samples):
        median_ms = statistics.median(samples)
        rank_stats.append(
            {
                "rank": rank,
                "samples_ms": samples,
                "min_ms": min(samples),
                "median_ms": median_ms,
                "p95_ms": percentile(samples, 0.95),
                "max_ms": max(samples),
                "effective_gbps": sent_bytes_per_rank / 1_000_000 / median_ms,
            }
        )

    slow_rank_samples = [max(values) for values in zip(*all_rank_samples, strict=True)]
    rank_medians = [row["median_ms"] for row in rank_stats]
    slow_median_ms = statistics.median(slow_rank_samples)
    return {
        "name": name,
        "logical_bytes": logical_bytes,
        "sent_bytes_per_rank": sent_bytes_per_rank,
        "rank_stats": rank_stats,
        "slow_rank": {
            "samples_ms": slow_rank_samples,
            "min_ms": min(slow_rank_samples),
            "median_ms": slow_median_ms,
            "p95_ms": percentile(slow_rank_samples, 0.95),
            "max_ms": max(slow_rank_samples),
            "effective_gbps": sent_bytes_per_rank / 1_000_000 / slow_median_ms,
        },
        "rank_median_spread_ms": max(rank_medians) - min(rank_medians),
        "rank_median_pstdev_ms": statistics.pstdev(rank_medians),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"{summary['name']}: logical={summary['logical_bytes'] / 1_000_000:.3f} MB, "
        f"send={summary['sent_bytes_per_rank'] / 1_000_000:.3f} MB/rank"
    )
    for row in summary["rank_stats"]:
        print(
            f"  rank {row['rank']}: median={row['median_ms']:.3f} ms, "
            f"p95={row['p95_ms']:.3f} ms, "
            f"range={row['min_ms']:.3f}-{row['max_ms']:.3f} ms"
        )
    slow = summary["slow_rank"]
    print(
        f"  slow-rank path: median={slow['median_ms']:.3f} ms, "
        f"p95={slow['p95_ms']:.3f} ms, "
        f"range={slow['min_ms']:.3f}-{slow['max_ms']:.3f} ms, "
        f"effective={slow['effective_gbps']:.3f} GB/s, "
        f"rank-median-spread={summary['rank_median_spread_ms']:.3f} ms, "
        f"rank-median-pstdev={summary['rank_median_pstdev_ms']:.3f} ms"
    )


def profile_once(
    name: str,
    op: Callable[[], None],
    *,
    profile_dir: str | None,
    rank: int,
) -> None:
    """Optionally capture one synchronized collective on every rank."""
    if profile_dir is None:
        return
    trace_dir = Path(profile_dir) / f"rank{rank}" / name
    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = torch_npu.profiler.tensorboard_trace_handler(str(trace_dir))
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    dist.barrier()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=handler,
        record_shapes=True,
        experimental_config=experimental_config,
    ) as profiler:
        op()
        torch.npu.synchronize()
        profiler.step()
    dist.barrier()


def write_report(path: str | None, payload: dict[str, Any]) -> None:
    if path is None or dist.get_rank() != 0:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"saved raw samples and summaries to {output}")

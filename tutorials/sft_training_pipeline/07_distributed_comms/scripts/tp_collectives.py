#!/usr/bin/env python3
"""Two-rank HCCL benchmark for the TP all-reduce used in chapter 07."""

from __future__ import annotations

import argparse
import os
from functools import partial as bind

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # Registers the NPU device and HCCL backend.

from benchmark_utils import (
    measure_samples_ms,
    print_summary,
    profile_once,
    summarize_across_ranks,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Qwen3 TP all-reduce for a [B, S, H] activation."
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-json", help="Save every rank/iteration and derived statistics.")
    parser.add_argument("--profile-dir", help="Optionally capture one HCCL trace per rank.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch, args.sequence, args.hidden, args.iters) <= 0 or args.warmup < 0:
        raise ValueError("tensor dimensions and iters must be positive; warmup must be non-negative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    try:
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"This chapter script requires exactly 2 ranks, got {world_size}")

        device = torch.device(f"npu:{local_rank}")
        partial = torch.zeros(
            (args.batch, args.sequence, args.hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        all_reduce_op = bind(dist.all_reduce, partial, op=dist.ReduceOp.SUM)
        samples_ms = measure_samples_ms(
            all_reduce_op,
            args.warmup,
            args.iters,
        )

        logical_bytes = partial.numel() * partial.element_size()
        sent_bytes = 2 * (world_size - 1) / world_size * logical_bytes
        summary = summarize_across_ranks(
            "all_reduce",
            samples_ms,
            device=device,
            logical_bytes=logical_bytes,
            sent_bytes_per_rank=sent_bytes,
        )
        profile_once(
            "all_reduce",
            all_reduce_op,
            profile_dir=args.profile_dir,
            rank=dist.get_rank(),
        )

        if dist.get_rank() == 0:
            print(
                f"shape={tuple(partial.shape)}, dtype=bf16, world_size={world_size}, "
                f"warmup={args.warmup}, iters={args.iters}"
            )
            assert summary is not None
            print_summary(summary)
            write_report(
                args.output_json,
                {
                    "benchmark": "tp_collectives",
                    "world_size": world_size,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "operations": [summary],
                },
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

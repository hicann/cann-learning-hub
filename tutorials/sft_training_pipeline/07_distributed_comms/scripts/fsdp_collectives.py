#!/usr/bin/env python3
"""Two-rank HCCL benchmark for the FSDP collectives used in chapter 07."""

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
        description="Benchmark one Qwen3 layer's FSDP all-gather and reduce-scatter."
    )
    parser.add_argument("--layer-params", type=int, default=50_336_000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-json", help="Save every rank/iteration and derived statistics.")
    parser.add_argument("--profile-dir", help="Optionally capture one HCCL trace per operation and rank.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layer_params <= 0 or args.warmup < 0 or args.iters <= 0:
        raise ValueError("layer-params and iters must be positive; warmup must be non-negative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    try:
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"This chapter script requires exactly 2 ranks, got {world_size}")
        if args.layer_params % world_size:
            raise ValueError("layer-params must be divisible by world size")

        device = torch.device(f"npu:{local_rank}")
        shard_numel = args.layer_params // world_size
        local_param = torch.zeros(shard_numel, dtype=torch.bfloat16, device=device)
        full_param = torch.empty(args.layer_params, dtype=torch.bfloat16, device=device)
        full_grad = torch.zeros_like(full_param)
        reduced_grad = torch.empty_like(local_param)

        all_gather_op = bind(dist.all_gather_into_tensor, full_param, local_param)
        reduce_scatter_op = bind(dist.reduce_scatter_tensor, reduced_grad, full_grad)
        all_gather_samples = measure_samples_ms(
            all_gather_op,
            args.warmup,
            args.iters,
        )
        reduce_scatter_samples = measure_samples_ms(
            reduce_scatter_op,
            args.warmup,
            args.iters,
        )

        full_bytes = args.layer_params * local_param.element_size()
        sent_bytes = (world_size - 1) / world_size * full_bytes
        summaries = [
            summarize_across_ranks(
                "all_gather",
                all_gather_samples,
                device=device,
                logical_bytes=full_bytes,
                sent_bytes_per_rank=sent_bytes,
            ),
            summarize_across_ranks(
                "reduce_scatter",
                reduce_scatter_samples,
                device=device,
                logical_bytes=full_bytes,
                sent_bytes_per_rank=sent_bytes,
            ),
        ]

        profile_once(
            "all_gather",
            all_gather_op,
            profile_dir=args.profile_dir,
            rank=dist.get_rank(),
        )
        profile_once(
            "reduce_scatter",
            reduce_scatter_op,
            profile_dir=args.profile_dir,
            rank=dist.get_rank(),
        )

        if dist.get_rank() == 0:
            print(
                f"Qwen3 layer params={args.layer_params:,}, dtype=bf16, "
                f"world_size={world_size}, warmup={args.warmup}, iters={args.iters}"
            )
            for summary in summaries:
                assert summary is not None
                print_summary(summary)
            write_report(
                args.output_json,
                {
                    "benchmark": "fsdp_collectives",
                    "world_size": world_size,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "operations": summaries,
                },
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

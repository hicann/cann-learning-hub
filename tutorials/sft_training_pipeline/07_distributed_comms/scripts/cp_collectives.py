#!/usr/bin/env python3
"""Two-rank HCCL benchmark for the Ulysses-CP all-to-all operations."""

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
        description="Benchmark Qwen3 Ulysses-CP QKV and output all-to-all buffers."
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-json", help="Save every rank/iteration and derived statistics.")
    parser.add_argument("--profile-dir", help="Optionally capture one HCCL trace per operation and rank.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dimensions = (
        args.batch,
        args.sequence,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.iters,
    )
    if min(dimensions) <= 0 or args.warmup < 0:
        raise ValueError("tensor dimensions and iters must be positive; warmup must be non-negative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    try:
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"This chapter script requires exactly 2 ranks, got {world_size}")
        if args.sequence % world_size:
            raise ValueError("sequence must be divisible by world size")
        if args.q_heads % world_size or args.kv_heads % world_size:
            raise ValueError("Q and KV head counts must be divisible by world size")

        device = torch.device(f"npu:{local_rank}")
        local_sequence = args.sequence // world_size
        q_numel = args.batch * local_sequence * args.q_heads * args.head_dim
        kv_numel = args.batch * local_sequence * args.kv_heads * args.head_dim
        output_numel = args.batch * local_sequence * args.q_heads * args.head_dim

        q_input = torch.zeros(q_numel, dtype=torch.bfloat16, device=device)
        q_output = torch.empty_like(q_input)
        k_input = torch.zeros(kv_numel, dtype=torch.bfloat16, device=device)
        k_output = torch.empty_like(k_input)
        v_input = torch.zeros(kv_numel, dtype=torch.bfloat16, device=device)
        v_output = torch.empty_like(v_input)
        output_input = torch.zeros(output_numel, dtype=torch.bfloat16, device=device)
        output_output = torch.empty_like(output_input)

        operations = [
            (
                "pre_attention_q_all_to_all",
                q_input,
                bind(dist.all_to_all_single, q_output, q_input),
            ),
            (
                "pre_attention_k_all_to_all",
                k_input,
                bind(dist.all_to_all_single, k_output, k_input),
            ),
            (
                "pre_attention_v_all_to_all",
                v_input,
                bind(dist.all_to_all_single, v_output, v_input),
            ),
            (
                "post_attention_output_all_to_all",
                output_input,
                bind(dist.all_to_all_single, output_output, output_input),
            ),
        ]
        summaries = []
        for name, tensor, op in operations:
            samples_ms = measure_samples_ms(op, args.warmup, args.iters)
            local_bytes = tensor.numel() * tensor.element_size()
            summaries.append(
                summarize_across_ranks(
                    name,
                    samples_ms,
                    device=device,
                    logical_bytes=local_bytes,
                    sent_bytes_per_rank=(world_size - 1) / world_size * local_bytes,
                )
            )
            profile_once(name, op, profile_dir=args.profile_dir, rank=dist.get_rank())

        if dist.get_rank() == 0:
            print(
                f"B={args.batch}, S={args.sequence}, local_S={local_sequence}, "
                f"Q/KV heads={args.q_heads}/{args.kv_heads}, D={args.head_dim}, "
                f"dtype=bf16, world_size={world_size}, warmup={args.warmup}, "
                f"iters={args.iters}"
            )
            for summary in summaries:
                assert summary is not None
                print_summary(summary)
            write_report(
                args.output_json,
                {
                    "benchmark": "cp_collectives",
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

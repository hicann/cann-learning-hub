#!/usr/bin/env python3
"""Generate a labelled tree and deterministic task costs for the lab."""
from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path

from scheduler import fifo_schedule, pipeline_schedule, priority_schedule


def build_tree(num_nodes: int, rng: random.Random) -> list[int]:
    if num_nodes < 2:
        return [-1]
    # Build a tree whose node ids are deliberately not BFS ordered.
    logical_parent = [-1] + [(i - 1) // 2 for i in range(1, num_nodes)]
    permutation = list(range(1, num_nodes))
    rng.shuffle(permutation)
    order = [0] + permutation
    new_id = {old: idx for idx, old in enumerate(order)}
    parent = [-1] * num_nodes
    for old_child in range(1, num_nodes):
        parent[new_id[old_child]] = new_id[logical_parent[old_child]]
    return parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=31)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.num_nodes < 2:
        raise SystemExit("--num_nodes must be at least 2")

    rng = random.Random(args.seed)
    parent = build_tree(args.num_nodes, rng)
    costs = [rng.randint(1, 9) for _ in parent]
    payload = {
        "num_nodes": args.num_nodes,
        "seed": args.seed,
        "parent": parent,
        "cost": costs,
        "copy_in": 1.0,
        "copy_out": 1.0,
        "queue_depth": 2,
        "compute_lanes": 2,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "input.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fifo_order = fifo_schedule(parent)
    priority_order = priority_schedule(parent, costs)
    device_dir = args.output / "input"
    device_dir.mkdir(parents=True, exist_ok=True)
    (device_dir / "parent.bin").write_bytes(struct.pack(f"<{len(parent)}i", *parent))
    (device_dir / "cost.bin").write_bytes(
        b"".join(struct.pack("<e", float(value)) for value in costs)
    )
    (device_dir / "order_priority.bin").write_bytes(
        struct.pack(f"<{len(priority_order)}i", *priority_order)
    )
    (device_dir / "order_fifo.bin").write_bytes(struct.pack(f"<{len(fifo_order)}i", *fifo_order))
    for name, order in (("priority", priority_order), ("fifo", fifo_order)):
        timing = pipeline_schedule(
            order,
            costs,
            queue_depth=payload["queue_depth"],
            copy_in=payload["copy_in"],
            copy_out=payload["copy_out"],
            compute_lanes=payload["compute_lanes"],
        )
        stage_end = [0.0] * len(parent)
        for event in timing["events"]:
            stage_end[event["node"]] = event["copy_out"][1]
        (device_dir / f"ref_stage_end_{name}.bin").write_bytes(
            b"".join(struct.pack("<e", value) for value in stage_end)
        )
    (device_dir / "ref_dependency_ok.bin").write_bytes(struct.pack("<i", 1))
    (device_dir / "meta.json").write_text(
        json.dumps(
            {
                "num_nodes": len(parent),
                "queue_depth": payload["queue_depth"],
                "compute_lanes": payload["compute_lanes"],
                "copy_in": payload["copy_in"],
                "copy_out": payload["copy_out"],
                "orders": ["fifo", "priority"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated {args.num_nodes} nodes at {args.output / 'input.json'}")


if __name__ == "__main__":
    main()

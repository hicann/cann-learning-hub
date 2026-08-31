#!/usr/bin/env python3
"""Run the complete tree/queue/heap/pipeline experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scheduler import bfs_frontier, fifo_schedule, pipeline_schedule, priority_schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.data_dir / "output.json"
    payload = json.loads((args.data_dir / "input.json").read_text(encoding="utf-8"))
    parent = payload["parent"]
    cost = payload["cost"]
    bfs_order, depth, levels = bfs_frontier(parent)
    fifo_order = fifo_schedule(parent)
    priority_order = priority_schedule(parent, cost)
    result = {
        "num_nodes": payload["num_nodes"],
        "parent": parent,
        "cost": cost,
        "depth": depth,
        "levels": levels,
        "bfs_order": bfs_order,
        "fifo_order": fifo_order,
        "priority_order": priority_order,
        "fifo_pipeline": pipeline_schedule(
            fifo_order,
            cost,
            payload["queue_depth"],
            payload["copy_in"],
            payload["copy_out"],
            payload["compute_lanes"],
        ),
        "priority_pipeline": pipeline_schedule(
            priority_order,
            cost,
            payload["queue_depth"],
            payload["copy_in"],
            payload["copy_out"],
            payload["compute_lanes"],
        ),
    }
    result["priority_speedup"] = result["fifo_pipeline"]["end_to_end"] / result[
        "priority_pipeline"
    ]["end_to_end"]
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"BFS levels: {[len(level) for level in levels]}")
    print(f"FIFO end_to_end: {result['fifo_pipeline']['end_to_end']:.1f}")
    print(f"priority end_to_end: {result['priority_pipeline']['end_to_end']:.1f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

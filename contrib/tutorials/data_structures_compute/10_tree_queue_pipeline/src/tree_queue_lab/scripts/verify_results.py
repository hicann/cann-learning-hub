#!/usr/bin/env python3
"""Check dependency order and pipeline invariants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_order(order: list[int], parent: list[int]) -> None:
    if sorted(order) != list(range(len(parent))):
        raise AssertionError("schedule does not contain every node exactly once")
    position = {node: index for index, node in enumerate(order)}
    for node, p in enumerate(parent):
        if p >= 0 and position[p] >= position[node]:
            raise AssertionError(f"parent {p} is scheduled after child {node}")


def check_pipeline(pipeline: dict) -> None:
    previous_out = 0.0
    for event in pipeline["events"]:
        in_start, in_end = event["copy_in"]
        compute_start, compute_end = event["compute"]
        out_start, out_end = event["copy_out"]
        if not (in_start <= in_end <= compute_start <= compute_end <= out_start <= out_end):
            raise AssertionError(f"invalid stage order for node {event['node']}")
        if out_end < previous_out:
            raise AssertionError("pipeline completion time moved backwards")
        previous_out = out_end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    check_order(result["fifo_order"], result["parent"])
    check_order(result["priority_order"], result["parent"])
    check_pipeline(result["fifo_pipeline"])
    check_pipeline(result["priority_pipeline"])
    print("tree dependencies: PASS")
    print("FIFO pipeline: PASS")
    print("priority pipeline: PASS")


if __name__ == "__main__":
    main()


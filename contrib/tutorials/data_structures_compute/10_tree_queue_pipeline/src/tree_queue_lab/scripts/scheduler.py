#!/usr/bin/env python3
"""Reference implementation for tree frontier, heap scheduling and pipelining."""
from __future__ import annotations

import heapq
from collections import deque
from typing import Any


def children_of(parent: list[int]) -> list[list[int]]:
    children = [[] for _ in parent]
    for node, p in enumerate(parent):
        if p >= 0:
            children[p].append(node)
    return children


def bfs_frontier(parent: list[int]) -> tuple[list[int], list[int], list[list[int]]]:
    children = children_of(parent)
    root = parent.index(-1)
    depth = [-1] * len(parent)
    depth[root] = 0
    queue: deque[int] = deque([root])
    order: list[int] = []
    levels: list[list[int]] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        level = depth[node]
        while len(levels) <= level:
            levels.append([])
        levels[level].append(node)
        for child in children[node]:
            depth[child] = level + 1
            queue.append(child)
    return order, depth, levels


def subtree_work(parent: list[int], cost: list[int]) -> list[int]:
    children = children_of(parent)
    work = cost[:]
    _, depth, _ = bfs_frontier(parent)
    for node in sorted(range(len(parent)), key=lambda n: depth[n], reverse=True):
        for child in children[node]:
            work[node] += work[child]
    return work


def priority_schedule(parent: list[int], cost: list[int]) -> list[int]:
    """Pop ready tasks by larger remaining subtree work, then lower node id."""
    children = children_of(parent)
    work = subtree_work(parent, cost)
    root = parent.index(-1)
    ready: list[tuple[int, int, int]] = [(-work[root], root, root)]
    result: list[int] = []
    while ready:
        _, _, node = heapq.heappop(ready)
        result.append(node)
        for child in children[node]:
            # Children become eligible immediately after their parent completes.
            heapq.heappush(ready, (-work[child], child, child))
    return result


def fifo_schedule(parent: list[int]) -> list[int]:
    order, _, _ = bfs_frontier(parent)
    return order


def pipeline_schedule(
    order: list[int],
    cost: list[int],
    queue_depth: int = 2,
    copy_in: float = 1.0,
    copy_out: float = 1.0,
    compute_lanes: int = 2,
) -> dict[str, Any]:
    """Simulate CopyIn -> Compute -> CopyOut with reusable buffer slots."""
    if queue_depth < 1:
        raise ValueError("queue_depth must be positive")
    if compute_lanes < 1:
        raise ValueError("compute_lanes must be positive")
    copy_free = 0.0
    compute_free = [0.0] * compute_lanes
    copy_out_free = 0.0
    slots = [0.0] * queue_depth
    events: list[dict[str, Any]] = []
    for node in order:
        slot = min(range(queue_depth), key=lambda idx: slots[idx])
        in_start = max(copy_free, slots[slot])
        in_end = in_start + copy_in
        lane = min(range(compute_lanes), key=lambda idx: compute_free[idx])
        compute_start = max(compute_free[lane], in_end)
        compute_end = compute_start + float(cost[node])
        out_start = max(copy_out_free, compute_end)
        out_end = out_start + copy_out
        slots[slot] = out_end
        copy_free, compute_free[lane], copy_out_free = in_end, compute_end, out_end
        events.append(
            {
                "node": node,
                "slot": slot,
                "compute_lane": lane,
                "copy_in": [in_start, in_end],
                "compute": [compute_start, compute_end],
                "copy_out": [out_start, out_end],
            }
        )
    return {
        "queue_depth": queue_depth,
        "compute_lanes": compute_lanes,
        "events": events,
        "end_to_end": copy_out_free,
    }

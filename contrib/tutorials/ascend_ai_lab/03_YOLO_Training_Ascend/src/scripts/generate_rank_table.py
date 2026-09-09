#!/usr/bin/env python3
"""Generate a compact HCCL rank table for Ascend training examples.

Example:
    python src/scripts/generate_rank_table.py \
        --host node0:192.168.100.10,192.168.100.11,192.168.100.12,192.168.100.13 \
        --host node1:192.168.101.10,192.168.101.11,192.168.101.12,192.168.101.13 \
        --output rank_table_2node_8p.json

The device IPs should come from hccn_tool, for example:
    hccn_tool -i 0 -ip -g
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_host(value: str) -> tuple[str, list[str]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("--host must be formatted as server_id:ip0,ip1,...")
    server_id, raw_ips = value.split(":", 1)
    ips = [item.strip() for item in raw_ips.split(",") if item.strip()]
    if not server_id or not ips:
        raise argparse.ArgumentTypeError("--host must include both server_id and device IPs")
    return server_id, ips


def build_rank_table(hosts: list[tuple[str, list[str]]]) -> dict:
    rank_id = 0
    server_list = []
    for server_id, ips in hosts:
        devices = []
        for device_id, device_ip in enumerate(ips):
            devices.append(
                {
                    "device_id": str(device_id),
                    "device_ip": device_ip,
                    "rank_id": str(rank_id),
                }
            )
            rank_id += 1
        server_list.append({"server_id": server_id, "device": devices})
    return {
        "version": "1.0",
        "server_count": str(len(server_list)),
        "server_list": server_list,
        "status": "completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="append", type=parse_host, required=True)
    parser.add_argument("--output", default="rank_table.json")
    args = parser.parse_args()

    table = build_rank_table(args.host)
    output = Path(args.output)
    output.write_text(json.dumps(table, indent=2), encoding="utf-8")
    world_size = sum(len(host[1]) for host in args.host)
    print(f"rank table saved to {output.resolve()}")
    print(f"server_count={len(args.host)}, world_size={world_size}")


if __name__ == "__main__":
    main()

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

"""Generate a single-server HCCL JSON rank table (CANN 9.0 cluster info format)."""
import argparse
import json
import logging
import sys

LOGGER = logging.getLogger(__name__)


def validate_device(device):
    """Validate one device entry and return its rank id when available."""
    errors = []
    for field in ("device_id", "device_ip", "rank_id"):
        if field not in device or not str(device[field]):
            errors.append("device entry is missing non-empty %s" % field)
    rank_id = int(device["rank_id"]) if "rank_id" in device and str(device["rank_id"]) else None
    return errors, rank_id


def validate_devices(devices):
    """Validate all devices and collect their rank ids."""
    errors = []
    rank_ids = []
    for device in devices:
        device_errors, rank_id = validate_device(device)
        errors.extend(device_errors)
        if rank_id is not None:
            rank_ids.append(rank_id)
    return errors, rank_ids


def has_duplicate_device_ips(devices):
    """Return whether a device list contains duplicate or missing IP values."""
    addresses = [str(device.get("device_ip")) for device in devices]
    return len(addresses) != len(set(addresses))


def validate_server(server):
    """Validate one server entry and collect its rank ids."""
    errors = []
    if "server_id" not in server or not server["server_id"]:
        errors.append("server_list entry is missing non-empty server_id")
    devices = server.get("device", [])
    if not devices:
        errors.append("server %r has no device entries" % server.get("server_id", ""))
    device_errors, rank_ids = validate_devices(devices)
    errors.extend(device_errors)
    if has_duplicate_device_ips(devices):
        errors.append("server %r has duplicate or missing device-ip entries" % server.get("server_id", ""))
    return errors, rank_ids


def validate_table(table):
    """Validate required fields, device-ip count and rank-id continuity."""
    errors = [
        "missing required top-level field: " + field
        for field in ("status", "version", "server_count", "server_list")
        if field not in table
    ]
    server_list = table.get("server_list", [])
    if not server_list:
        errors.append("server_list must not be empty")
    rank_ids = []
    for server in server_list:
        server_errors, server_ranks = validate_server(server)
        errors.extend(server_errors)
        rank_ids.extend(server_ranks)
    if rank_ids and sorted(rank_ids) != list(range(min(rank_ids), min(rank_ids) + len(rank_ids))):
        errors.append("rank_id values must be contiguous: %s" % sorted(rank_ids))
    return errors


def build_table(server_ip, device_ips):
    """Build a one-server rank table from ordered device IP addresses."""
    devices = []
    for rank_id, device_ip in enumerate(device_ips):
        devices.append({
            "device_id": str(rank_id),
            "device_ip": device_ip,
            "rank_id": str(rank_id),
        })
    return {
        "status": "completed",
        "version": "1.0",
        "server_count": "1",
        "server_list": [{"server_id": server_ip, "device": devices}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", required=True, help="server management/network IP")
    parser.add_argument("--device-ip", nargs="+", required=True, metavar="NPU_IP",
                        help="1 to 8 HCCN device IPs, ordered by device id")
    parser.add_argument("--output", default="/tmp/hccl_spmv_rank_table.json")
    args = parser.parse_args()
    if not 1 <= len(args.device_ip) <= 8:
        parser.error("--device-ip requires between 1 and 8 addresses")
    table = build_table(args.server_ip, args.device_ip)
    errors = validate_table(table)
    if errors:
        for error in errors:
            LOGGER.error("rank table validation failed: %s", error)
        sys.exit(1)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(table, output, indent=2)
        output.write("\n")
    LOGGER.info("%s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()

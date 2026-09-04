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

"""Generate a single-server HCCL rank table for selected Ascend devices."""

import argparse
import json
import logging
import sys

LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", required=True)
    parser.add_argument("--device-id", nargs="+", required=True, type=int)
    parser.add_argument("--device-ip", nargs="+", required=True)
    parser.add_argument("--output", default="rank_table.json")
    args = parser.parse_args()
    if len(args.device_id) != len(args.device_ip):
        parser.error("--device-id and --device-ip must contain the same number of entries")
    devices = [
        {"device_id": str(device), "device_ip": ip, "rank_id": str(rank)}
        for rank, (device, ip) in enumerate(zip(args.device_id, args.device_ip))
    ]
    table = {
        "version": "1.0",
        "server_count": "1",
        "server_list": [{"server_id": args.server_ip, "device": devices}],
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(table, output, indent=2)
        output.write("\n")
    LOGGER.info("%s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()

#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import argparse
import os

import numpy as np


SAMPLE_CONFIGS = {
    "leakyrelu": {
        "m": 1,
        "k": 1,
        "n": 8192,
        "op": "leakyrelu",
        "post_process": "leakyrelu",
    },
    "mmad": {
        "m": 512,
        "k": 128,
        "n": 128,
        "op": "mmad",
        "post_process": None,
    },
    "mmad_leakyrelu": {
        "m": 512,
        "k": 128,
        "n": 128,
        "op": "mmad",
        "post_process": "leakyrelu",
    },
}


def apply_post_process(golden, post_process):
    if post_process is None:
        return golden
    if post_process == "leakyrelu":
        return np.where(golden >= 0, golden, golden * np.float16(0.001)).astype(np.float16)
    raise ValueError(f"unsupported post process: {post_process}")


def gen_golden_data(sample):
    config = SAMPLE_CONFIGS[sample]
    m = config["m"]
    n = config["n"]
    k = config["k"]

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    if config["op"] == "leakyrelu":
        # Use small binary-friendly fp16 values in an activation-like range.
        x_gm = (np.random.randint(-128, 129, [n]) / 128).astype(np.float16)
        golden = apply_post_process(x_gm, config["post_process"]).astype(np.float16)
        x_gm.tofile("./input/x_gm.bin")
        golden.tofile("./output/golden.bin")
        return

    if config["op"] != "mmad":
        raise ValueError(f"unsupported op: {config['op']}")

    # Use small binary-friendly fp16 values. B is scaled by a power of two so
    # K-direction accumulation stays in an activation-like range.
    x1_gm = (np.random.randint(-8, 9, [m, k]) / 8).astype(np.float16)
    x2_gm = (np.random.randint(-8, 9, [k, n]) / 64).astype(np.float16)
    bias_gm = (np.random.randint(-8, 9, [n]) / 64).astype(np.float16)
    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))
    golden = golden + bias_gm.astype(np.float32)
    golden = golden.astype(np.float16)
    golden = apply_post_process(golden, config["post_process"])

    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    bias_gm.tofile("./input/bias_gm.bin")
    golden.tofile("./output/golden.bin")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate input and golden data for static tensor CV samples.")
    parser.add_argument(
        "--sample",
        choices=sorted(SAMPLE_CONFIGS.keys()),
        default="mmad_leakyrelu",
        help="sample name; default: mmad_leakyrelu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen_golden_data(args.sample)

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


import os
import numpy as np


def gen_golden_data():
    m, n, k = 1920, 2048, 2048

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    rng = np.random.default_rng(20260713)
    x1_gm = (rng.integers(-8, 9, [m, k]) / 8).astype(np.float16)
    x2_gm = (rng.integers(-8, 9, [k, n]) / 64).astype(np.float16)
    bias_gm = (rng.integers(-8, 9, [n]) / 64).astype(np.float16)
    matmul_result = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))
    biased = (matmul_result + bias_gm.astype(np.float32)).astype(np.float64)
    exponent = -1.595769 * (biased + 0.044715 * biased**3)
    golden = (biased / (1.0 + np.exp(np.clip(exponent, -88.0, 88.0)))).astype(np.float32)

    x1_gm.tofile("./input/x1_gm.bin")
    # x2_gm transpose to match B matrix transpose
    x2_gm = x2_gm.transpose()
    x2_gm.tofile("./input/x2_gm.bin")
    bias_gm.tofile("./input/bias_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()

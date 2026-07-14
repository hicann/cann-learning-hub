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

np.random.seed(9)


def gen_golden_data():
    data_size = 512
    input0 = np.random.uniform(-1.0, 1.0, [data_size]).astype(np.float16)
    input1 = np.random.uniform(-1.0, 1.0, [data_size]).astype(np.float16)
    scalar = 2.0
    golden = (input1 + input0 * scalar).astype(np.float16)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    input0.tofile("./input/input0.bin")
    input1.tofile("./input/input1.bin")
    golden.tofile("./output/golden.bin")

    print(f"Generated Axpy data: input0 shape={input0.shape}, dtype={input0.dtype}")


if __name__ == "__main__":
    gen_golden_data()

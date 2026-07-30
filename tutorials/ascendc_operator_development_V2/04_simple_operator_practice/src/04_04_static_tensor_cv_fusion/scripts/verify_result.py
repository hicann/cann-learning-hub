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


import sys
import numpy as np


RELATIVE_TOL = 1e-6
ABSOLUTE_TOL = 1e-9
ERROR_TOL = 1e-4


def load_float16_data(file_path):
    return np.fromfile(file_path, dtype=np.float16).reshape(-1)


def get_mismatch_indexes(output_data, golden_data):
    close_mask = np.isclose(output_data,
                            golden_data,
                            rtol=RELATIVE_TOL,
                            atol=ABSOLUTE_TOL,
                            equal_nan=True)
    return np.where(close_mask == False)[0]


def print_mismatch_details(output_data, golden_data, mismatch_indexes):
    for index, real_index in enumerate(mismatch_indexes):
        expected = golden_data[real_index]
        actual = output_data[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, expected, actual,
            abs(actual - expected) / max(abs(expected), 1e-8)))
        if index == 100:
            break


def verify_result(output, golden):
    output_data = load_float16_data(output)
    golden_data = load_float16_data(golden)
    mismatch_indexes = get_mismatch_indexes(output_data, golden_data)
    print_mismatch_details(output_data, golden_data, mismatch_indexes)
    print("golden_data : ", golden_data)
    print("output : ", output_data)
    error_ratio = float(mismatch_indexes.size) / golden_data.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, ERROR_TOL))
    return error_ratio <= ERROR_TOL


if __name__ == '__main__':
    try:
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] result error")
        print("test pass!")
    except Exception as e:
        print(e)
        sys.exit(1)

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
import sys
import numpy as np


RELATIVE_TOL = 1e-3
ABSOLUTE_TOL = 1e-3
ERROR_TOL = 1e-3


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=np.float16).reshape(-1)
    golden = np.fromfile(golden_path, dtype=np.float16).reshape(-1)

    different_element_results = np.isclose(
        output,
        golden,
        rtol=RELATIVE_TOL,
        atol=ABSOLUTE_TOL,
        equal_nan=True,
    )
    different_element_indexes = np.where(different_element_results == False)[0]
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data if golden_data != 0 else abs(output_data - golden_data))
        )
        if index == 100:
            break

    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, ERROR_TOL))
    return error_ratio <= ERROR_TOL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Output file path")
    parser.add_argument("golden", help="Golden file path")
    args = parser.parse_args()

    try:
        res = verify_result(args.output, args.golden)
        if not res:
            raise ValueError("[ERROR] result error")
        print("test pass!")
    except Exception as exc:
        print(exc)
        sys.exit(1)

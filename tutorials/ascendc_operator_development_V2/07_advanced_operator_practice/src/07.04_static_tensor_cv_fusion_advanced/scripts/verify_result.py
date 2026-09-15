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


CLOSENESS_OPTIONS = {
    "rtol": 1e-3,
    "atol": 1e-3,
    "equal_nan": True,
}
MAX_ERROR_RATIO = 1e-3
MAX_REPORTED_ERRORS = 101


class Float32Comparison:
    def __init__(self, output_path, golden_path):
        self.actual = np.fromfile(output_path, dtype=np.float32).ravel()
        self.expected = np.fromfile(golden_path, dtype=np.float32).ravel()

    def mismatch_indexes(self):
        matched = np.isclose(self.actual, self.expected, **CLOSENESS_OPTIONS)
        return np.flatnonzero(np.logical_not(matched))

    def report_mismatches(self, indexes):
        for data_index in indexes[:MAX_REPORTED_ERRORS]:
            expected_value = self.expected[data_index]
            actual_value = self.actual[data_index]
            relative_difference = abs(actual_value - expected_value) / expected_value
            print(
                f"data index: {data_index:06d}, expected: {expected_value:-.9f}, "
                f"actual: {actual_value:-.9f}, rdiff: {relative_difference:-.6f}"
            )

    def verify(self):
        indexes = self.mismatch_indexes()
        self.report_mismatches(indexes)

        error_ratio = float(indexes.size) / self.expected.size
        print(f"error ratio: {error_ratio:.4f}, tolerance: {MAX_ERROR_RATIO:.4f}")
        return error_ratio <= MAX_ERROR_RATIO


def verify_result(output, golden):
    return Float32Comparison(output, golden).verify()


def main(arguments):
    try:
        if not verify_result(arguments[1], arguments[2]):
            raise ValueError("[ERROR] result error")
    except Exception as error:
        print(error)
        return 1

    print("test pass!")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

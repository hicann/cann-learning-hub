#!/usr/bin/env python3
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
from pathlib import Path
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--golden-dir", default='./golden', type=str, help="golden bin directory")
parser.add_argument("--op-out-dir", default='./output', type=str, help="op output directory")
args = parser.parse_args()

golden_dir, op_out_dir = Path(args.golden_dir).resolve(), Path(args.op_out_dir).resolve()

accuracy_result = {}
for golden_bin in golden_dir.rglob("*.bin"):
    bin_rel_path = golden_bin.relative_to(golden_dir)
    op_out_bin = op_out_dir / bin_rel_path

    golden_data = golden_bin.read_bytes()
    op_out_valid_data = op_out_bin.open('rb').read(len(golden_data))
    accuracy_result[str(bin_rel_path)] = (golden_data == op_out_valid_data)

print('accuracy result:')
pprint(accuracy_result)
print()
if (all(accuracy_result.values())):
    print("accuracy test success.")
else:
    print("accuracy test fail.")
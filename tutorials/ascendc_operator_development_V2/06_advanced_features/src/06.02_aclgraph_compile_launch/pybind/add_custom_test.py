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
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path.cwd()))
import ascendc_ops


def main():
    shape = (8, 2048)
    static_x = torch.rand(shape, device="npu", dtype=torch.float16)
    static_y = torch.rand(shape, device="npu", dtype=torch.float16)

    with torch.no_grad():
        ascendc_ops.ascendc_add(static_x, static_y)
    torch_npu.npu.synchronize()

    graph = torch_npu.npu.NPUGraph()
    with torch.no_grad(), torch_npu.npu.graph(graph):
        graph_output = ascendc_ops.ascendc_add(static_x, static_y)

    next_x = torch.rand(shape, dtype=torch.float16)
    next_y = torch.rand(shape, dtype=torch.float16)
    static_x.copy_(next_x.npu())
    static_y.copy_(next_y.npu())
    graph.replay()
    torch_npu.npu.synchronize()

    torch.testing.assert_close(graph_output.cpu(), next_x + next_y)
    print("Pybind11 custom operator Aclgraph replay success.")


if __name__ == "__main__":
    main()

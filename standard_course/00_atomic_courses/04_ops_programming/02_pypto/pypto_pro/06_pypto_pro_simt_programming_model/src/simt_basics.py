# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 6 章：SIMT 索引、块内计数和分歧实验；计时为 Host 墙钟。

在 Ascend 950 上运行。本轮仅做源码和静态检查，尚无设备通过或性能结论。
"""

import logging
import os
import statistics
import time

import pypto_pro.language as pl
import torch
import torch_npu

os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

LOGGER = logging.getLogger(__name__)


@pl.vector_function(mode="simt", max_threads=256)
def add_threads(x, y, out):
    i = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    if i < 1024:
        out[0, i] = x[0, i] + y[0, i]


@pl.jit()
def add_kernel(
    x: pl.Tensor[[1, 1024], pl.DT_FP32], y: pl.Tensor[[1, 1024], pl.DT_FP32], out: pl.Tensor[[1, 1024], pl.DT_FP32]
):
    with pl.section_vector():
        add_threads[256](x, y, out)


@pl.vector_function(mode="simt", max_threads=256)
def count_threads(shared, out):
    tid = pl.simt.thread_idx().x
    if tid == 0:
        shared[0, 0] = 0
    pl.simt.syncthreads()
    pl.simt.atomic_add(shared, [0, 0], 1)
    pl.simt.syncthreads()
    if tid == 0:
        out[0, pl.simt.block_idx().x] = shared[0, 0]


@pl.jit(auto_mutex=True)
def count_kernel(out: pl.Tensor[[1, 4], pl.DT_INT32]):
    tt = pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    with pl.section_vector():
        shared = group.current()
        count_threads[256](shared, out)


@pl.vector_function(mode="simt", max_threads=256)
def positive_branch(x, out):
    i = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    if i < 1024:
        value = x[0, i]
        if value > 0.0:
            value = value * 2.0
        out[0, i] = value


@pl.vector_function(mode="simt", max_threads=256)
def positive_no_branch(x, out):
    i = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    if i < 1024:
        value = x[0, i]
        out[0, i] = value + pl.simt.max(value, 0.0)


@pl.jit()
def branch_kernel(x: pl.Tensor[[1, 1024], pl.DT_FP32], out: pl.Tensor[[1, 1024], pl.DT_FP32]):
    with pl.section_vector():
        positive_branch[256](x, out)


@pl.jit()
def no_branch_kernel(x: pl.Tensor[[1, 1024], pl.DT_FP32], out: pl.Tensor[[1, 1024], pl.DT_FP32]):
    with pl.section_vector():
        positive_no_branch[256](x, out)


def time_calls(kernel, x, out):
    for _ in range(5):
        kernel[None, 4](x, out)
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(50):
            kernel[None, 4](x, out)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / 50)
    return statistics.median(samples)  # Host enqueue + device work, not pure device time.


def check_examples():
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    x = torch.randn(1, 1024, device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.full_like(x, float('nan'))
    add_kernel[None, 4](x, y, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, x + y, rtol=1e-6, atol=1e-6)
    counts = torch.full((1, 4), -1, device=device, dtype=torch.int32)
    count_kernel[None, 4](counts)
    torch.npu.synchronize()
    torch.testing.assert_close(counts, torch.full_like(counts, 256), rtol=0, atol=0)
    for pattern in ('grouped', 'alternating'):
        values = torch.arange(1024, device=device).reshape(1, 1024)
        positive = values < 512 if pattern == 'grouped' else values % 2 == 0
        data = torch.where(positive, 1.0, -1.0).to(torch.float32)
        expected = torch.where(data > 0, data * 2, data)
        for name, kernel in (('branch', branch_kernel), ('no_branch', no_branch_kernel)):
            out = torch.full_like(data, float('nan'))
            kernel[None, 4](data, out)
            torch.npu.synchronize()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            elapsed = time_calls(kernel, data, out)
            LOGGER.info('%s %s host-wall ms/call: %s', pattern, name, elapsed)
    LOGGER.info('PASS: SIMT add, block count and four divergence cases')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    check_examples()

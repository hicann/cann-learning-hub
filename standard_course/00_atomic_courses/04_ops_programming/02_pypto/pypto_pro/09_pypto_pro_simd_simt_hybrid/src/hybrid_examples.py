# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 9 章：UB 协同、两 Kernel 紧凑化后处理和稳定 Top8。

在 Ascend 950 上运行。本轮仅做源码和静态检查，尚无设备通过或性能结论。
"""

import logging
import os

import pypto_pro.language as pl
import torch
import torch_npu

os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

LOGGER = logging.getLogger(__name__)


@pl.vector_function(mode="simt", max_threads=256)
def filter_local(x, y):
    tid = pl.simt.linear_thread_idx()
    for i in pl.range(tid, 1024, 256):
        value = x[0, i]
        if value > 0.0:
            y[0, i] = value * 2.0
        else:
            y[0, i] = 0.0


@pl.jit(auto_mutex=True)
def hybrid_scale(x: pl.Tensor[[1, 1024], pl.DT_FP32], y: pl.Tensor[[1, 1024], pl.DT_FP32]):
    tt = pl.TileType(shape=[1, 1024], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])
    with pl.section_vector():
        tx = gx.current()
        ty = gy.current()
        pl.load(tx, x, [0, 0])
        filter_local[256](tx, ty)
        pl.store(y, ty, [0, 0])


@pl.vector_function(mode="simt", max_threads=256)
def compact_rows(data, scores, out, source_ids, count):
    first = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
    for row in pl.range(first, 1024, stride):
        if scores[0, row] > 0.0:
            slot = pl.simt.atomic_add(count[0, 0], 1)
            source_ids[0, slot] = pl.simt.cast(row, pl.DT_INT32)
            for col in pl.range(0, 64):
                out[slot, col] = data[row, col]


@pl.jit()
def select_kernel(
    data: pl.Tensor[[1024, 64], pl.DT_FP32],
    scores: pl.Tensor[[1, 1024], pl.DT_FP32],
    out: pl.Tensor[[1024, 64], pl.DT_FP32],
    source_ids: pl.Tensor[[1, 1024], pl.DT_INT32],
    count: pl.Tensor[[1, 1], pl.DT_INT32],
):
    with pl.section_vector():
        compact_rows[256](data, scores, out, source_ids, count)


@pl.jit(auto_mutex=True)
def center_kernel(x: pl.Tensor[[1024, 64], pl.DT_FP32], y: pl.Tensor[[1024, 64], pl.DT_FP32], count: pl.DT_INT32):
    tt = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dn = pl.TileType(
        shape=[8, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    rm = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x800, mutex_ids=[1])
    gw = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[2])
    gd = pl.make_tile_group(type=dn, addrs=0x1800, mutex_ids=[3])
    gr = pl.make_tile_group(type=rm, addrs=0x1800, mutex_ids=[4])
    with pl.section_vector():
        for task in pl.range(pl.get_block_idx(), (count + 7) // 8, pl.get_block_num()):
            row = task * 8
            valid = pl.min(8, count - row)
            tx = gx.current()
            ty = gy.current()
            tmp = gw.current()
            sd = gd.current()
            sr = gr.current()
            pl.set_validshape(tx, [valid, 64])
            pl.set_validshape(ty, [valid, 64])
            pl.set_validshape(tmp, [valid, 64])
            pl.set_validshape(sd, [valid, 1])
            pl.set_validshape(sr, [1, valid])
            pl.load(tx, x, [row, 0])
            pl.sum(sd, tx, tmp, dim=0)
            pl.div(sr, sr, 64.0)
            pl.expand_sub(ty, tx, sd, dim=0)
            pl.store(y, ty, [row, 0])


@pl.vector_function(mode="simt", max_threads=32)
def top8_threads(scores, selected):
    i = pl.simt.thread_idx().x
    rank = 0
    for j in pl.range(0, 32):
        if scores[0, j] > scores[0, i] or (scores[0, j] == scores[0, i] and j < i):
            rank = rank + 1
    if rank < 8:
        selected[0, rank] = pl.simt.cast(i, pl.DT_INT32)


@pl.jit()
def top8_kernel(scores: pl.Tensor[[1, 32], pl.DT_FP32], selected: pl.Tensor[[1, 8], pl.DT_INT32]):
    with pl.section_vector():
        top8_threads[32](scores, selected)


def check_examples():
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    x = torch.randn(1, 1024, device=device, dtype=torch.float32)
    y = torch.full_like(x, float('nan'))
    hybrid_scale[None, 1](x, y)
    torch.npu.synchronize()
    torch.testing.assert_close(y, torch.relu(x) * 2, rtol=0, atol=0)
    data = torch.randn(1024, 64, device=device, dtype=torch.float32)
    for scores in (torch.full_like(x, -1), torch.ones_like(x), x):
        packed = torch.full_like(data, float('nan'))
        ids = torch.full((1, 1024), -1, device=device, dtype=torch.int32)
        count = torch.zeros(1, 1, device=device, dtype=torch.int32)
        select_kernel[None, 4](data, scores, packed, ids, count)
        torch.npu.synchronize()  # Host reads count only after the full producer kernel.
        n = int(count.item())
        chosen = ids[0, :n].long()
        expected_ids = (scores[0] > 0).nonzero().flatten()
        torch.testing.assert_close(chosen.sort().values, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(packed[:n], data.index_select(0, chosen), rtol=0, atol=0)
        centered = torch.full_like(data, float('nan'))
        if n:
            center_kernel[None, 4](packed, centered, n)
            torch.npu.synchronize()
            golden = packed[:n] - packed[:n].mean(-1, keepdim=True)
            torch.testing.assert_close(centered[:n], golden, rtol=1e-4, atol=1e-5)
        if not torch.isnan(centered[n:]).all().item():
            raise AssertionError('untouched rows must stay NaN')
    scores = (torch.arange(32, device=device, dtype=torch.float32) % 5).reshape(1, 32)
    selected = torch.full((1, 8), -1, device=device, dtype=torch.int32)
    top8_kernel[None, 1](scores, selected)
    torch.npu.synchronize()
    golden = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :8].to(torch.int32)
    torch.testing.assert_close(selected, golden, rtol=0, atol=0)
    LOGGER.info('PASS: UB hybrid, compaction/centering, stable Top8')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    check_examples()

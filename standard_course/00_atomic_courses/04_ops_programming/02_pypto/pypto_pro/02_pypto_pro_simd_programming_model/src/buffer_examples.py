# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 2 章：方向标注、双槽预取和双输入缓冲；每个示例附独立 Golden。

在 Ascend 950 上运行。本轮仅做源码和静态检查，尚无设备通过或性能结论。
"""

import logging
import os

import pypto_pro.language as pl
import torch
import torch_npu

os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

LOGGER = logging.getLogger(__name__)


@pl.jit(auto_mutex=True)
def add_direction(
    x: pl.Tensor[[8, 256], pl.DT_FP32, pl.Input],
    y: pl.Tensor[[8, 256], pl.DT_FP32, pl.Input],
    out: pl.Tensor[[8, 256], pl.DT_FP32, pl.Output],
):
    tt = pl.TileType(shape=[8, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
    go = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        tx = gx.current()
        ty = gy.current()
        to = go.current()
        pl.load(tx, x, [0, 0])
        pl.load(ty, y, [0, 0])
        pl.add(to, tx, ty)
        pl.store(out, to, [0, 0])


@pl.jit(auto_mutex=True)
def scale_prefetch(x: pl.Tensor[[8, 256], pl.DT_FP32], out: pl.Tensor[[8, 256], pl.DT_FP32]):
    tt = pl.TileType(shape=[4, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
    with pl.section_vector():
        # 预热：进稳态循环前先搬入第 0 块
        first = group.next()
        pl.load(first, x, [0, 0])
        for task in pl.range(0, 2):
            cur = group.current()  # 本轮要计算的槽位
            if task + 1 < 2:  # 非末轮才预取
                nxt = group.next()
                pl.load(nxt, x, [(task + 1) * 4, 0])  # 预取搬运与计算重叠
            pl.mul(cur, cur, 2.0)
            pl.store(out, cur, [task * 4, 0])


@pl.jit(auto_mutex=True)
def add_double(
    x: pl.Tensor[[8, 256], pl.DT_FP32], y: pl.Tensor[[8, 256], pl.DT_FP32], out: pl.Tensor[[8, 256], pl.DT_FP32]
):
    tt = pl.TileType(shape=[4, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
    gy = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2, 3])
    go = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[4])
    with pl.section_vector():
        for task in pl.range(0, 2):
            tx = gx.next()
            ty = gy.next()
            to = go.current()
            pl.load(tx, x, [task * 4, 0])
            pl.load(ty, y, [task * 4, 0])
            pl.add(to, tx, ty)
            pl.store(out, to, [task * 4, 0])


def check_examples():
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    # Distinct row values expose wrong tile offsets; random data adds coverage.
    for x in (
        torch.arange(2048, device=device, dtype=torch.float32).reshape(8, 256),
        torch.randn(8, 256, device=device, dtype=torch.float32),
    ):
        y = torch.randn_like(x)
        for kernel, args, golden in (
            (add_direction, (x, y), x + y),
            (scale_prefetch, (x,), x * 2),
            (add_double, (x, y), x + y),
        ):
            out = torch.full_like(x, float('nan'))
            kernel[None, 1](*args, out)
            torch.npu.synchronize()
            if not torch.isfinite(out).all().item():
                raise AssertionError('output must be finite')
            torch.testing.assert_close(out, golden, rtol=1e-5, atol=1e-6)
    LOGGER.info('PASS: direction, prefetch scale, double-buffer add')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    check_examples()

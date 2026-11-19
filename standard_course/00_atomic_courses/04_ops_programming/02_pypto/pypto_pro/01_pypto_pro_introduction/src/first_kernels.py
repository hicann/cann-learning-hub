# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 1 章：Add、缩放和线性组合的完整 Kernel 与 Host 验收。

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
def add_fp16(
    x: pl.Tensor[[16, 512], pl.DT_FP16], y: pl.Tensor[[16, 512], pl.DT_FP16], out: pl.Tensor[[16, 512], pl.DT_FP16]
):
    tt = pl.TileType(shape=[16, 512], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=16384, mutex_ids=[1])
    go = pl.make_tile_group(type=tt, addrs=32768, mutex_ids=[2])
    with pl.section_vector():
        tx = gx.current()
        ty = gy.current()
        to = go.current()
        pl.load(tx, x, [0, 0])
        pl.load(ty, y, [0, 0])
        pl.add(to, tx, ty)
        pl.store(out, to, [0, 0])


@pl.jit(auto_mutex=True)
def scale(x: pl.Tensor[[8, 256], pl.DT_FP32], out: pl.Tensor[[8, 256], pl.DT_FP32], alpha: pl.DT_FP32):
    tt = pl.TileType(shape=[8, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    go = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
    with pl.section_vector():
        tx = gx.current()
        to = go.current()
        pl.load(tx, x, [0, 0])
        pl.mul(to, tx, alpha)
        pl.store(out, to, [0, 0])


@pl.jit(auto_mutex=True)
def linear_combination(
    x: pl.Tensor[[8, 256], pl.DT_FP32],
    y: pl.Tensor[[8, 256], pl.DT_FP32],
    out: pl.Tensor[[8, 256], pl.DT_FP32],
    alpha: pl.DT_FP32,
    beta: pl.DT_FP32,
):
    tt = pl.TileType(shape=[8, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=8192, mutex_ids=[1])
    go = pl.make_tile_group(type=tt, addrs=16384, mutex_ids=[2])
    with pl.section_vector():
        tx = gx.current()
        ty = gy.current()
        to = go.current()
        pl.load(tx, x, [0, 0])
        pl.load(ty, y, [0, 0])
        pl.mul(tx, tx, alpha)
        pl.mul(ty, ty, beta)
        pl.add(to, tx, ty)
        pl.store(out, to, [0, 0])


def check_examples():
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    xh = torch.randn(16, 512, device=device, dtype=torch.float16)
    yh = torch.randn_like(xh)
    out = torch.full_like(xh, float('nan'))
    add_fp16[None, 1](xh, yh, out)
    torch.npu.synchronize()
    if not torch.isfinite(out).all().item():
        raise AssertionError('output must be finite')
    torch.testing.assert_close(out, xh + yh, rtol=1e-3, atol=1e-3)
    x = torch.randn(8, 256, device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    for alpha, beta in ((0.0, 1.0), (2.0, -1.0), (-0.5, 0.25)):
        out = torch.full_like(x, float('nan'))
        scale[None, 1](x, out, alpha)
        torch.npu.synchronize()
        torch.testing.assert_close(out, alpha * x, rtol=1e-5, atol=1e-6)
        out.fill_(float('nan'))
        linear_combination[None, 1](x, y, out, alpha, beta)
        torch.npu.synchronize()
        if not torch.isfinite(out).all().item():
            raise AssertionError('output must be finite')
        torch.testing.assert_close(out, alpha * x + beta * y, rtol=1e-5, atol=1e-6)
    LOGGER.info('PASS: 第 1 章三个体验任务')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    check_examples()

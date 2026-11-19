# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 3 章：同一 RMSNorm 规格的 Tile 与 VF 实现，以及数值边界验收。

在 Ascend 950 上运行。本轮仅做源码和静态检查，尚无设备通过或性能结论。
"""

import logging
import os

import pypto_pro.language as pl
import torch
import torch_npu
from pypto_pro.language import Vf as vf

os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

LOGGER = logging.getLogger(__name__)


@pl.jit(auto_mutex=True)
def relu_affine(
    x: pl.Tensor[[8, 256], pl.DT_FP32], y: pl.Tensor[[8, 256], pl.DT_FP32], alpha: pl.DT_FP32, beta: pl.DT_FP32
):
    tt = pl.TileType(shape=[8, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
    with pl.section_vector():
        tx = gx.current()
        ty = gy.current()
        pl.load(tx, x, [0, 0])
        pl.relu(ty, tx)
        pl.mul(ty, ty, alpha)
        pl.add(ty, ty, beta)
        pl.store(y, ty, [0, 0])


@pl.jit(auto_mutex=True)
def rms_tile(x: pl.Tensor[[8, 256], pl.DT_FP32], y: pl.Tensor[[8, 256], pl.DT_FP32]):
    tt = pl.TileType(shape=[8, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    dn = pl.TileType(shape=[8, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    rm = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
    gs = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])
    gw = pl.make_tile_group(type=tt, addrs=0x6000, mutex_ids=[3])
    gd = pl.make_tile_group(type=dn, addrs=0x8000, mutex_ids=[4])
    gr = pl.make_tile_group(type=rm, addrs=0x8000, mutex_ids=[5])
    with pl.section_vector():
        tx = gx.current()
        ty = gy.current()
        sq = gs.current()
        tmp = gw.current()
        stat_dn = gd.current()
        stat_rm = gr.current()
        pl.load(tx, x, [0, 0])
        pl.mul(sq, tx, tx)  # 平方
        pl.sum(stat_dn, sq, tmp, dim=0)  # 行平方和 -> [8,1] DN
        pl.div(stat_rm, stat_rm, 256.0)  # 行主序视图上除 N
        pl.add(stat_rm, stat_rm, 1e-5)  # 加 eps
        pl.rsqrt(stat_rm, stat_rm)  # 倒数标准差
        pl.expand_mul(ty, tx, stat_dn, dim=0)  # 按行广播缩放
        pl.store(y, ty, [0, 0])


@pl.vector_function
def rms_registers(tx, ty):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    inv_n = vf.full(0.00390625, mask, dtype=pl.DT_FP32)  # 1/256；VF 域内不做标量除法
    eps = vf.full(1e-5, mask, dtype=pl.DT_FP32)
    one = vf.full(1.0, mask, dtype=pl.DT_FP32)
    for row in pl.range(0, 8):
        x0 = vf.load_align(tx, [row, 0])
        x1 = vf.load_align(tx, [row, 64])
        x2 = vf.load_align(tx, [row, 128])
        x3 = vf.load_align(tx, [row, 192])
        s0 = vf.mul(x0, x0, mask)  # 逐片段平方
        s1 = vf.mul(x1, x1, mask)
        s2 = vf.mul(x2, x2, mask)
        s3 = vf.mul(x3, x3, mask)
        s01 = vf.add(s0, s1, mask)  # 片段内累加
        s23 = vf.add(s2, s3, mask)
        sums = vf.add(s01, s23, mask)
        lane0 = vf.reduce_sum(sums, mask)  # 只有 lane 0 有效
        total = vf.full(lane0, mask)  # 广播回全部 lane
        mean = vf.mul(total, inv_n, mask)
        stabilized = vf.add(mean, eps, mask)
        std = vf.sqrt(stabilized, mask)
        rstd = vf.div(one, std, mask)
        y0 = vf.mul(x0, rstd, mask)  # 归一化
        y1 = vf.mul(x1, rstd, mask)
        y2 = vf.mul(x2, rstd, mask)
        y3 = vf.mul(x3, rstd, mask)
        vf.store_align(ty, y0, mask, [row, 0])
        vf.store_align(ty, y1, mask, [row, 64])
        vf.store_align(ty, y2, mask, [row, 128])
        vf.store_align(ty, y3, mask, [row, 192])


@pl.jit(auto_mutex=True)
def rms_vf(x: pl.Tensor[[8, 256], pl.DT_FP32], y: pl.Tensor[[8, 256], pl.DT_FP32]):
    tt = pl.TileType(shape=[8, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gx = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
    with pl.section_vector():
        tx = gx.current()
        ty = gy.current()
        pl.load(tx, x, [0, 0])
        rms_registers(tx, ty)
        pl.store(y, ty, [0, 0])


def check_examples():
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    random = torch.randn(8, 256, device=device, dtype=torch.float32)
    cases = [random, torch.zeros_like(random), torch.full_like(random, -3.0), random * 1e-3]
    for x in cases:
        golden = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
        for kernel in (rms_tile, rms_vf):
            out = torch.full_like(x, float('nan'))
            kernel[None, 1](x, out)
            torch.npu.synchronize()
            if not torch.isfinite(out).all().item():
                raise AssertionError('output must be finite')
            torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-5)
        out = torch.full_like(x, float('nan'))
        relu_affine[None, 1](x, out, 0.5, -1.0)
        torch.npu.synchronize()
        torch.testing.assert_close(out, torch.relu(x) * 0.5 - 1.0, rtol=1e-5, atol=1e-6)
    LOGGER.info('PASS: relu-affine, Tile RMSNorm, VF RMSNorm')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    check_examples()

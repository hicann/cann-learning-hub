# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 5 章：单组 MIX 的 Bias/GELU 融合与单槽、双槽交接参考。

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


@pl.vector_function
def gelu_vf(x, y):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, mask, dtype=pl.DT_FP32)
    zero = vf.full(0.0, mask, dtype=pl.DT_FP32)
    half = vf.full(0.5, mask, dtype=pl.DT_FP32)
    coeff = vf.full(0.044715, mask, dtype=pl.DT_FP32)
    scale = vf.full(0.7978845608, mask, dtype=pl.DT_FP32)
    neg2 = vf.full(-2.0, mask, dtype=pl.DT_FP32)
    for offset in pl.range(0, 64 * 128, 64):
        xx = vf.load_align(x, offset)
        x2 = vf.mul(xx, xx, mask)
        x3 = vf.mul(x2, xx, mask)
        cubic = vf.mul(x3, coeff, mask)
        inner = vf.add(xx, cubic, mask)
        u = vf.mul(inner, scale, mask)
        au = vf.abs(u, mask)
        exponent = vf.mul(au, neg2, mask)
        e = vf.exp(exponent, mask)
        numer = vf.sub(one, e, mask)
        denom = vf.add(one, e, mask)
        positive = vf.div(numer, denom, mask)
        negative = vf.neg(positive, mask)
        sign = vf.ge(u, zero, mask)
        tanh_u = vf.select(positive, negative, sign)
        factor = vf.add(one, tanh_u, mask)
        prod = vf.mul(xx, factor, mask)
        result = vf.mul(prod, half, mask)
        vf.store_align(y, result, mask, offset)


@pl.jit(auto_mutex=True)
def gemm_bias(
    a: pl.Tensor[[256, 128], pl.DT_FP16],
    b: pl.Tensor[[128, 256], pl.DT_FP16],
    bias: pl.Tensor[[1, 256], pl.DT_FP32],
    workspace: pl.Tensor[[1, 128, 128], pl.DT_FP32],
    out: pl.Tensor[[256, 256], pl.DT_FP32],
):
    a1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat), addrs=0, mutex_ids=[0]
    )
    b1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x8000,
        mutex_ids=[1],
    )
    al = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left), addrs=0, mutex_ids=[2]
    )
    br = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right), addrs=0, mutex_ids=[3]
    )
    ac = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0, mutex_ids=[4]
    )
    vx = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0, mutex_ids=[5]
    )
    vy = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[6],
    )
    vb = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000,
        mutex_ids=[7],
    )
    with pl.section_cube():
        for task in pl.range(0, 4):
            slot = task % 1
            if task >= 1:
                pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=slot + 2)
            row = (task // 2) * 128
            col = (task % 2) * 128
            ta = a1.current()
            tb = b1.current()
            left = al.current()
            right = br.current()
            result = ac.current()
            pl.load(ta, a, [row, 0])
            pl.load(tb, b, [0, col])
            pl.move(left, ta)
            pl.move(right, tb)
            pl.matmul(result, left, right)
            pl.store(workspace, result, [slot, 0, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=slot)
        # Consume final releases; no unmatched events remain at return.
        for slot in pl.range(0, 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=slot + 2)
    with pl.section_vector():
        sub = pl.get_subblock_idx()
        for task in pl.range(0, 4):
            slot = task % 1
            row = (task // 2) * 128 + sub * 64
            col = (task % 2) * 128
            tx = vx.current()
            ty = vy.current()
            bias_tile = vb.current()
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=slot)
            pl.load(tx, workspace, [slot, sub * 64, 0])
            pl.load(bias_tile, bias, [0, col])
            pl.neg(bias_tile, bias_tile)  # 仅修改 UB Tile，GM 中的 bias 不变
            pl.expand_sub(ty, tx, bias_tile, dim=1)  # tx - (-bias)
            # Bias-only result stays in ty.
            pl.store(out, ty, [row, col])
            # Conservative release: include postprocessing and output store.
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=slot + 2)


@pl.jit(auto_mutex=True)
def gemm_gelu_serial(
    a: pl.Tensor[[256, 128], pl.DT_FP16],
    b: pl.Tensor[[128, 256], pl.DT_FP16],
    bias: pl.Tensor[[1, 256], pl.DT_FP32],
    workspace: pl.Tensor[[1, 128, 128], pl.DT_FP32],
    out: pl.Tensor[[256, 256], pl.DT_FP32],
):
    a1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat), addrs=0, mutex_ids=[0]
    )
    b1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x8000,
        mutex_ids=[1],
    )
    al = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left), addrs=0, mutex_ids=[2]
    )
    br = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right), addrs=0, mutex_ids=[3]
    )
    ac = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0, mutex_ids=[4]
    )
    vx = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0, mutex_ids=[5]
    )
    vy = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[6],
    )
    vb = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000,
        mutex_ids=[7],
    )
    with pl.section_cube():
        for task in pl.range(0, 4):
            slot = task % 1
            if task >= 1:
                pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=slot + 2)
            row = (task // 2) * 128
            col = (task % 2) * 128
            ta = a1.current()
            tb = b1.current()
            left = al.current()
            right = br.current()
            result = ac.current()
            pl.load(ta, a, [row, 0])
            pl.load(tb, b, [0, col])
            pl.move(left, ta)
            pl.move(right, tb)
            pl.matmul(result, left, right)
            pl.store(workspace, result, [slot, 0, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=slot)
        # Consume final releases; no unmatched events remain at return.
        for slot in pl.range(0, 1):
            pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=slot + 2)
    with pl.section_vector():
        sub = pl.get_subblock_idx()
        for task in pl.range(0, 4):
            slot = task % 1
            row = (task // 2) * 128 + sub * 64
            col = (task % 2) * 128
            tx = vx.current()
            ty = vy.current()
            bias_tile = vb.current()
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=slot)
            pl.load(tx, workspace, [slot, sub * 64, 0])
            pl.load(bias_tile, bias, [0, col])
            pl.neg(bias_tile, bias_tile)  # 仅修改 UB Tile，GM 中的 bias 不变
            pl.expand_sub(ty, tx, bias_tile, dim=1)  # tx - (-bias)
            gelu_vf(ty, tx)
            pl.store(out, tx, [row, col])
            # Conservative release: include postprocessing and output store.
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=slot + 2)


@pl.jit(auto_mutex=True)
def gemm_gelu_double(
    a: pl.Tensor[[256, 128], pl.DT_FP16],
    b: pl.Tensor[[128, 256], pl.DT_FP16],
    bias: pl.Tensor[[1, 256], pl.DT_FP32],
    workspace: pl.Tensor[[2, 128, 128], pl.DT_FP32],
    out: pl.Tensor[[256, 256], pl.DT_FP32],
):
    a1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat), addrs=0, mutex_ids=[0]
    )
    b1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x8000,
        mutex_ids=[1],
    )
    al = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left), addrs=0, mutex_ids=[2]
    )
    br = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right), addrs=0, mutex_ids=[3]
    )
    ac = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addrs=0, mutex_ids=[4]
    )
    vx = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addrs=0, mutex_ids=[5]
    )
    vy = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids=[6],
    )
    vb = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000,
        mutex_ids=[7],
    )
    with pl.section_cube():
        for task in pl.range(0, 4):
            slot = task % 2
            if task >= 2:
                pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=slot + 2)
            row = (task // 2) * 128
            col = (task % 2) * 128
            ta = a1.current()
            tb = b1.current()
            left = al.current()
            right = br.current()
            result = ac.current()
            pl.load(ta, a, [row, 0])
            pl.load(tb, b, [0, col])
            pl.move(left, ta)
            pl.move(right, tb)
            pl.matmul(result, left, right)
            pl.store(workspace, result, [slot, 0, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=slot)
        # Consume final releases; no unmatched events remain at return.
        for slot in pl.range(0, 2):
            pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=slot + 2)
    with pl.section_vector():
        sub = pl.get_subblock_idx()
        for task in pl.range(0, 4):
            slot = task % 2
            row = (task // 2) * 128 + sub * 64
            col = (task % 2) * 128
            tx = vx.current()
            ty = vy.current()
            bias_tile = vb.current()
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=slot)
            pl.load(tx, workspace, [slot, sub * 64, 0])
            pl.load(bias_tile, bias, [0, col])
            pl.neg(bias_tile, bias_tile)  # 仅修改 UB Tile，GM 中的 bias 不变
            pl.expand_sub(ty, tx, bias_tile, dim=1)  # tx - (-bias)
            gelu_vf(ty, tx)
            pl.store(out, tx, [row, col])
            # Conservative release: include postprocessing and output store.
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=slot + 2)


def check_examples():
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    a = torch.randn(256, 128, device=device, dtype=torch.float16) * 0.25
    b = torch.randn(128, 256, device=device, dtype=torch.float16) * 0.25
    bias = torch.linspace(-4, 4, 256, device=device, dtype=torch.float32).reshape(1, 256)
    pre = a.float() @ b.float() + bias
    for kernel, depth, use_gelu in ((gemm_bias, 1, False), (gemm_gelu_serial, 1, True), (gemm_gelu_double, 2, True)):
        workspace = torch.empty(depth, 128, 128, device=device, dtype=torch.float32)
        out = torch.full((256, 256), float('nan'), device=device, dtype=torch.float32)
        kernel[None, 1](a, b, bias, workspace, out)  # One MIX block: one AIC and two AIVs.
        torch.npu.synchronize()
        golden = torch.nn.functional.gelu(pre, approximate='tanh') if use_gelu else pre
        if not torch.isfinite(out).all().item():
            raise AssertionError('output must be finite')
        torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
    LOGGER.info('PASS: Bias, serial GELU, two-slot GELU')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    check_examples()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Chapter 4 reference kernels and checks. Device execution requires Ascend 950.

Run from any directory: python /path/to/src/gemm_reference.py
The kernels are also printed in the chapter notebooks for step-by-step reading.
No device compilation or numerical validation has been performed on this host.
"""

import logging
import os

import pypto_pro.language as pl
import torch
import torch_npu

os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

LOGGER = logging.getLogger(__name__)


@pl.jit(auto_mutex=True)
def mm_single(
    a: pl.Tensor[[128, 128], pl.DT_FP16],
    b: pl.Tensor[[128, 128], pl.DT_FP16],
    out: pl.Tensor[[128, 128], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[128, 128]),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[128, 128]),
        addrs=0x08000,
        mutex_ids=[1],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[128, 128]),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[128, 128]
        ),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[128, 128], fractal=1024
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_left.current()
        br = b_right.current()
        ac = acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def mm_k_split(
    a: pl.Tensor[[128, 384], pl.DT_FP16],
    b: pl.Tensor[[384, 128], pl.DT_FP16],
    out: pl.Tensor[[128, 128], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[128, 128]),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[128, 128]),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[128, 128]),
        addrs=0x0000,
        mutex_ids=[4, 5],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[128, 128]
        ),
        addrs=0x0000,
        mutex_ids=[6, 7],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[128, 128], fractal=1024
        ),
        addrs=0x0000,
        mutex_ids=[8],
    )

    with pl.section_cube():
        ac = acc.current()
        for k in pl.range(0, 384, 128):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br)
            else:
                pl.matmul_acc(ac, ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def mm_transposed_b(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b_t: pl.Tensor[[32, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 32], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[64, 128]),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[128, 32], layout=pl.ZN
        ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[64, 128]),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[128, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[128, 32]),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[64, 32], fractal=1024
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_left.current()
        br = b_right.current()
        ac = acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b_t, [0, 0], order=[1, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])


@pl.jit(auto_mutex=True)
def mm_dynamic(
    a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
    b: pl.Tensor[[128, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    m = a.shape[0]
    n = b.shape[1]
    core_id = pl.get_block_idx()  # Pure Cube kernel: one task interval per AIC.
    core_num = pl.get_block_num()
    m_tiles = (m + 127) // 128
    n_tiles = (n + 127) // 128
    total = m_tiles * n_tiles
    tiles_per_core = (total + core_num - 1) // core_num
    start = pl.min(core_id * tiles_per_core, total)
    end = pl.min((core_id + 1) * tiles_per_core, total)

    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=0x00000,
        mutex_ids=[0, 1],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1]),
        addrs=0x10000,
        mutex_ids=[2, 3],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
        ),
        addrs=0x0000,
        mutex_ids=[5],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            valid_shape=[-1, -1],
            compact=1,
            fractal=1024,
        ),
        addrs=0x0000,
        mutex_ids=[6],
    )

    with pl.section_cube():
        for task in pl.range(start, end, 1):
            i = task // n_tiles
            j = task % n_tiles
            row = i * 128
            col = j * 128
            valid_m = pl.min(128, m - row)
            valid_n = pl.min(128, n - col)
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.current()
            br = b_right.current()
            ac = acc.current()
            pl.set_validshape(cur_a, [valid_m, 128])
            pl.set_validshape(cur_b, [128, valid_n])
            pl.set_validshape(al, [valid_m, 128])
            pl.set_validshape(br, [128, valid_n])
            pl.set_validshape(ac, [valid_m, valid_n])
            pl.load(cur_a, a, [row, 0])
            pl.load(cur_b, b, [0, col])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            pl.matmul(ac, al, br)
            pl.store(out, ac, [row, col])


def check_case(kernel, m, k, n, cores=1, *, pattern="random", transpose_b=False):
    """Run one 950 case; a NaN-filled output also detects unwritten elements."""
    if min(m, k, n, cores) <= 0:
        raise ValueError("Shapes and core count must be positive")
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    if pattern == "random":
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(k, n, device=device, dtype=torch.float16)
    elif pattern == "k_blocks":
        if k != 384:
            raise ValueError("k_blocks requires K=384")
        a = torch.ones(m, k, device=device, dtype=torch.float16)
        b = torch.ones(k, n, device=device, dtype=torch.float16)
        b[128:256, :] = 2
        b[256:384, :] = 3
    elif pattern == "ones":
        a = torch.ones(m, k, device=device, dtype=torch.float16)
        b = torch.ones(k, n, device=device, dtype=torch.float16)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    out = torch.full((m, n), float("nan"), device=device, dtype=torch.float32)
    rhs = b.t().contiguous() if transpose_b else b
    kernel[None, cores](a, rhs, out)
    torch.npu.synchronize()
    if out.shape != (m, n) or out.dtype != torch.float32:
        raise AssertionError("output shape and dtype must match the request")
    if not torch.isfinite(out).all().item():
        raise AssertionError("Output contains NaN/Inf or unwritten elements")
    golden = a.float() @ b.float()
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
    if pattern in {"ones", "k_blocks"}:
        expected = 768.0 if pattern == "k_blocks" else float(k)
        torch.testing.assert_close(out, torch.full_like(out, expected), rtol=0, atol=0)
    max_abs = (out - golden).abs().max().item()
    LOGGER.info(f"PASS M={m}, K={k}, N={n}, cores={cores}, {pattern}, max_abs={max_abs:.6g}")


def run_reference_checks():
    check_case(mm_single, 128, 128, 128)
    check_case(mm_single, 128, 128, 128, pattern="ones")
    check_case(mm_k_split, 128, 384, 128)
    check_case(mm_k_split, 128, 384, 128, pattern="k_blocks")
    check_case(mm_transposed_b, 64, 128, 32, transpose_b=True)
    for m, n, cores in ((128, 128, 4), (256, 256, 4), (512, 300, 4), (500, 300, 1), (500, 300, 4), (500, 300, 8)):
        check_case(mm_dynamic, m, 128, n, cores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_reference_checks()

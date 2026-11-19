# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 10 章：串行 GM 中转的完整 SDPA；片上流水优化另行实现。

在 Ascend 950 上运行。本轮仅做源码和静态检查，尚无设备通过或性能结论。
"""

import argparse
import itertools
import logging
import os
from functools import lru_cache

import pypto_pro.language as pl
import torch
import torch.nn.functional as F
import torch_npu

os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

LOGGER = logging.getLogger(__name__)


@pl.vector_function(mode="simt", max_threads=256)
def mask_block(tile, q0: pl.DT_INT32, k0: pl.DT_INT32, seq: pl.DT_INT32, causal: pl.DT_BOOL, fill: pl.DT_FP32):
    for i in pl.range(pl.simt.linear_thread_idx(), 64 * 128, 256):
        r = i // 128
        c = i % 128
        if q0 + r >= seq or k0 + c >= seq:
            tile[r, c] = fill
        elif causal and k0 + c > q0 + r:
            tile[r, c] = fill


@lru_cache(maxsize=2)
def build_attention(h):
    if h not in (64, 128):
        raise ValueError("h must be 64 or 128")
    scale = h**-0.5

    @pl.jit(auto_mutex=True)
    def block_attention(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, h], pl.DT_FP16],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, h], pl.DT_FP16],
        v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, h], pl.DT_FP16],
        selected: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT32],
        scores: pl.Tensor[[pl.DYNAMIC, 128, 128], pl.DT_FP32],
        probs: pl.Tensor[[pl.DYNAMIC, 128, 128], pl.DT_FP16],
        partial: pl.Tensor[[pl.DYNAMIC, 128, h], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, h], pl.DT_FP16],
        causal: pl.DT_BOOL,
    ):
        gq = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, h], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
            ),
            addrs=0,
            mutex_ids=[0],
        )
        gk = pl.make_tile_group(
            type=pl.TileType(
                shape=[h, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=0x8000,
            mutex_ids=[1],
        )
        gp = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
            ),
            addrs=0x10000,
            mutex_ids=[2],
        )
        gv = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, h], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1
            ),
            addrs=0x18000,
            mutex_ids=[3],
        )
        gql = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, h], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
            ),
            addrs=0,
            mutex_ids=[4],
        )
        gpl = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1
            ),
            addrs=0x8000,
            mutex_ids=[5],
        )
        gkr = pl.make_tile_group(
            type=pl.TileType(
                shape=[h, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
            ),
            addrs=0,
            mutex_ids=[6],
        )
        gvr = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, h], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1
            ),
            addrs=0x8000,
            mutex_ids=[7],
        )
        gsc = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
            ),
            addrs=0,
            mutex_ids=[8],
        )
        gpc = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, h], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, valid_shape=[-1, -1], compact=1
            ),
            addrs=0x10000,
            mutex_ids=[9],
        )
        gs = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0,
            mutex_ids=[10],
        )
        gph = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=0x8000,
            mutex_ids=[11],
        )
        gpv = pl.make_tile_group(
            type=pl.TileType(shape=[64, h], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0xC000,
            mutex_ids=[12],
        )
        go = pl.make_tile_group(
            type=pl.TileType(shape=[64, h], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x14000,
            mutex_ids=[13],
        )
        goh = pl.make_tile_group(
            type=pl.TileType(shape=[64, h], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=0x1C000,
            mutex_ids=[14],
        )
        gtmp = pl.make_tile_group(
            type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x20000,
            mutex_ids=[15],
        )
        gmd = pl.make_tile_group(
            type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=0x28000,
            mutex_ids=[16],
        )
        gmr = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x28000,
            mutex_ids=[16],
        )
        gnewd = pl.make_tile_group(
            type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=0x28100,
            mutex_ids=[18],
        )
        gnewr = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x28100,
            mutex_ids=[18],
        )
        galphad = pl.make_tile_group(
            type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=0x28200,
            mutex_ids=[20],
        )
        galphar = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x28200,
            mutex_ids=[20],
        )
        gld = pl.make_tile_group(
            type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=0x28300,
            mutex_ids=[22],
        )
        glr = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x28300,
            mutex_ids=[22],
        )
        gsumd = pl.make_tile_group(
            type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN),
            addrs=0x28400,
            mutex_ids=[24],
        )
        gsumr = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x28400,
            mutex_ids=[24],
        )

        # Q/K/V have zero-padded sequence storage; out retains the original S.
        seq = out.shape[1]
        nq = (seq + 127) // 128
        core = pl.get_block_idx() // pl.get_subblock_num()
        for task in pl.range(core, q.shape[0] * nq, pl.get_block_num()):
            batch = task // nq
            q0 = (task % nq) * 128
            qr = 128
            with pl.section_vector():
                sub = pl.get_subblock_idx()
                st = gs.current()
                ph = gph.current()
                pv = gpv.current()
                acc = go.current()
                oh = goh.current()
                tmp = gtmp.current()
                md = gmd.current()
                mr = gmr.current()
                nd = gnewd.current()
                nr = gnewr.current()
                ad = galphad.current()
                ar = galphar.current()
                ld = gld.current()
                lr = glr.current()
                sd = gsumd.current()
                sr = gsumr.current()
                pl.expands(mr, -1.0e30)
                pl.expands(lr, 0.0)
                pl.expands(acc, 0.0)
                pl.set_validshape(oh, [64, h])
            for j in pl.range(0, selected.shape[1]):
                # Shared selection for every batch/Q row. Original block IDs remain intact.
                k0 = pl.getval(selected, j) * 128
                kr = 128
                with pl.section_cube():
                    qt = gq.current()
                    kt = gk.current()
                    ql = gql.current()
                    kk = gkr.current()
                    sc = gsc.current()
                    pl.set_validshape(qt, [qr, h])
                    pl.set_validshape(ql, [qr, h])
                    pl.set_validshape(kt, [h, kr])
                    pl.set_validshape(kk, [h, kr])
                    pl.set_validshape(sc, [qr, kr])
                    pl.load(qt, q, [batch, q0, 0])
                    pl.load(kt, k, [batch, k0, 0], order=[2, 1])
                    pl.move(ql, qt)
                    pl.move(kk, kt)
                    pl.matmul(sc, ql, kk)
                    pl.store(scores, sc, [core, 0, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
                with pl.section_vector():
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
                    pl.load(st, scores, [core, sub * 64, 0])
                    pl.mul(st, st, scale)
                    # A finite sentinel avoids -inf-(-inf). Mask probabilities again below.
                    mask_block[256](st, q0 + sub * 64, k0, seq, causal, -1.0e30)
                    pl.maximum(nd, st, tmp, dim=0)
                    pl.maximum(nr, nr, mr)
                    pl.sub(ar, mr, nr)
                    pl.exp(ar, ar)
                    pl.expand_sub(st, st, nd, dim=0)
                    pl.exp(st, st)
                    mask_block[256](st, q0 + sub * 64, k0, seq, causal, 0.0)
                    pl.sum(sd, st, tmp, dim=0)
                    pl.mul(lr, lr, ar)
                    pl.add(lr, lr, sr)
                    pl.add(mr, nr, 0.0)
                    pl.cast(ph, st)
                    pl.store(probs, ph, [core, sub * 64, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1)
                with pl.section_cube():
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=1)
                    pt = gp.current()
                    vt = gv.current()
                    pp = gpl.current()
                    vv = gvr.current()
                    pc = gpc.current()
                    pl.set_validshape(pt, [qr, kr])
                    pl.set_validshape(pp, [qr, kr])
                    pl.set_validshape(vt, [kr, h])
                    pl.set_validshape(vv, [kr, h])
                    pl.set_validshape(pc, [qr, h])
                    pl.load(pt, probs, [core, 0, 0])
                    pl.load(vt, v, [batch, k0, 0])
                    pl.move(pp, pt)
                    pl.move(vv, vt)
                    pl.matmul(pc, pp, vv)
                    pl.store(partial, pc, [core, 0, 0])
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=2)
                with pl.section_vector():
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=2)
                    pl.load(pv, partial, [core, sub * 64, 0])
                    pl.expand_mul(acc, acc, ad, dim=0)
                    pl.add(acc, acc, pv)
                    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=3)
                with pl.section_cube():
                    # The full round has finished before any GM slot is overwritten.
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=3)
            with pl.section_vector():
                # For any nonempty row, max-subtracted exponential sum is at least 1.
                # For an empty row, acc=0 and using denominator 1 returns exactly 0.
                pl.maximum(lr, lr, 1.0)
                pl.expand_div(acc, acc, ld, dim=0)
                pl.cast(oh, acc)
                valid = pl.min(64, seq - q0 - sub * 64)
                if valid > 0:
                    pl.set_validshape(oh, [valid, h])
                    pl.store(out, oh, [batch, q0 + sub * 64, 0])

    return block_attention


def prepare_attention(q, k, v, selected=None, cores=4):
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Q/K/V must share [b,s,h]")
    b, s, h = q.shape
    if b < 1 or s < 1 or h not in (64, 128):
        raise ValueError("invalid shape")
    if cores < 1:
        raise ValueError("invalid launch size")
    if q.device.type != "npu":
        raise ValueError("contiguous FP16 Q/K/V on one device required")
    for tensor in (q, k, v):
        if tensor.dtype != torch.float16 or tensor.device != q.device or not tensor.is_contiguous():
            raise ValueError("contiguous FP16 Q/K/V on one device required")
    torch.npu.set_device(q.device)
    if not all(torch.isfinite(x).all().item() for x in (q, k, v)):
        raise ValueError("finite inputs required")
    if selected is None:
        selected = torch.arange((s + 127) // 128, device=q.device, dtype=torch.int32).reshape(1, -1)
    if selected.ndim != 2 or selected.shape[0] != 1 or selected.shape[1] < 1:
        raise ValueError("selected must be [1,K], K>=1")
    if selected.dtype != torch.int32 or selected.device != q.device or not selected.is_contiguous():
        raise ValueError("contiguous INT32 selected on input device required")
    ids = selected.cpu().flatten().tolist()
    if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= (s + 127) // 128:
        raise ValueError("selected IDs must be unique valid original block IDs")
    # Keep the PV reduction axis at a full 128: no assumption about invalid
    # fractal lanes being zero. Masking uses the original S from out.shape[1].
    pad_rows = (-s) % 128
    q_work, k_work, v_work = (F.pad(x, (0, 0, 0, pad_rows)) if pad_rows else x for x in (q, k, v))
    cores = min(cores, b * ((s + 127) // 128))
    scores = torch.zeros(cores, 128, 128, device=q.device, dtype=torch.float32)
    probs = torch.zeros(cores, 128, 128, device=q.device, dtype=torch.float16)
    partial = torch.zeros(cores, 128, h, device=q.device, dtype=torch.float32)
    out = torch.full_like(q, float('nan'))
    kernel = build_attention(h)

    def launch(causal=False):
        kernel[None, cores](q_work, k_work, v_work, selected, scores, probs, partial, out, causal)
        return out

    return launch


def run_sdpa(q, k, v, causal=False, cores=4):
    return prepare_attention(q, k, v, cores=cores)(causal)


def validate_case(s=1024, h=64, b=1, causal=False, kind='random', seed=42):
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.manual_seed(seed)
    q, k, v = (torch.randn(b, s, h, device=device, dtype=torch.float16) for _ in range(3))
    if kind == 'zero':
        q.zero_()
        k.zero_()
    elif kind == 'constant':
        q.fill_(0.5)
        k.fill_(-0.25)
        v.fill_(0.125)
    elif kind == 'large':
        q.mul_(8)
        k.mul_(8)
    out = run_sdpa(q, k, v, causal=causal)
    torch.npu.synchronize()
    golden = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
    if not torch.isfinite(out).all().item():
        raise AssertionError('output must be finite')
    error = (out.float() - golden.float()).abs()
    bad = (~torch.isclose(out, golden, rtol=1e-3, atol=1e-3)).nonzero()
    if bad.numel():
        pos = tuple(bad[0].tolist())
        LOGGER.warning('first mismatch %s actual %s expected %s', pos, out[pos].item(), golden[pos].item())
    torch.testing.assert_close(out, golden, rtol=1e-3, atol=1e-3)
    LOGGER.info('%s', dict(b=b, s=s, h=h, causal=causal, kind=kind, seed=seed, max_abs=error.max().item()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='RFC shapes, b=1/2, two seeds; may take a long time')
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ['TILE_FWK_DEVICE_ID']))
    shapes = [(1, 1024, 64)]
    seeds = [42]
    if args.full:
        shapes = list(itertools.product((1, 2), (1024, 2048, 4096, 8192), (64, 128)))
        seeds = [0, 42]
    for b, s, h in shapes:
        for seed in seeds:
            for causal in (False, True):
                validate_case(s, h, b, causal, seed=seed)
    for causal in (False, True):
        for kind in ('zero', 'constant', 'large'):
            validate_case(causal=causal, kind=kind)
        validate_case(s=3000, h=128, b=2, causal=causal)  # extra tail case

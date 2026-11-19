# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 8 章：同规格的串行、手动双槽和自动流水比较；先精度后计时。"""

import argparse
import importlib.util
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import pypto_pro.language as pl
import torch
import torch_npu

os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

LOGGER = logging.getLogger(__name__)

baseline_path = Path(__file__).resolve().parents[2] / '05_pypto_pro_simd_fused_operator/src/fused_gemm.py'
spec = importlib.util.spec_from_file_location('course_ch05_fused', baseline_path)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
gelu_vf = baseline.gelu_vf


@pl.pipeline.stage
def cube_stage(task, a, b, shared, a1, b1, al, br, ac):
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
    handoff = shared.next()
    pl.move(handoff, result, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def vector_stage(task, sub, bias, shared, out, vx, vy, vb):
    row = (task // 2) * 128 + sub * 64
    col = (task % 2) * 128
    tx = vx.current()
    ty = vy.current()
    bt = vb.current()
    handoff = shared.next()
    pl.load(bt, bias, [0, col])
    pl.neg(bt, bt)
    pl.expand_sub(ty, handoff, bt, dim=1)
    gelu_vf(ty, tx)
    pl.store(out, tx, [row, col])


@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=1))
def gemm_auto(
    a: pl.Tensor[[256, 128], pl.DT_FP16],
    b: pl.Tensor[[128, 256], pl.DT_FP16],
    bias: pl.Tensor[[1, 256], pl.DT_FP32],
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
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000,
        mutex_ids=[5],
    )
    vy = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x18000,
        mutex_ids=[6],
    )
    vb = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x20000,
        mutex_ids=[7],
    )

    shared = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0,
        mutex_ids=[8, 9],
        fwd_ids=[0, 1],
        bwd_ids=[2, 3],
    )
    sub = pl.get_subblock_idx()
    for task in pl.range(0, 4):
        with pl.section_cube():
            cube_stage(task, a, b, shared, a1, b1, al, br, ac)
        with pl.section_vector():
            vector_stage(task, sub, bias, shared, out, vx, vy, vb)


def first_mismatch(actual, expected, rtol=1e-2, atol=1e-2):
    bad = ~torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    locations = bad.nonzero()
    if locations.numel() == 0:
        return None
    pos = tuple(int(x) for x in locations[0].tolist())
    return {'index': pos, 'actual': actual[pos].item(), 'expected': expected[pos].item()}


def benchmark(profile=False):
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    a = torch.randn(256, 128, device=device, dtype=torch.float16) * 0.25
    b = torch.randn(128, 256, device=device, dtype=torch.float16) * 0.25
    bias = torch.linspace(-4, 4, 256, device=device, dtype=torch.float32).reshape(1, 256)
    golden = torch.nn.functional.gelu(a.float() @ b.float() + bias, approximate='tanh')
    records = []
    versions = (
        ('single', baseline.gemm_gelu_serial, 1),
        ('manual_double', baseline.gemm_gelu_double, 2),
        ('auto', gemm_auto, 0),
    )
    for name, kernel, depth in versions:
        ws = torch.empty(depth, 128, 128, device=device, dtype=torch.float32)
        out = torch.full((256, 256), float('nan'), device=device, dtype=torch.float32)
        args = (a, b, bias, ws, out) if depth else (a, b, bias, out)
        kernel[None, 1](*args)
        torch.npu.synchronize()
        mismatch = first_mismatch(out, golden)
        if mismatch is not None:
            raise AssertionError((name, mismatch))
        for _ in range(5):
            kernel[None, 1](*args)
        torch.npu.synchronize()
        samples = []
        for _ in range(5):
            begin = time.perf_counter()
            for _ in range(20):
                kernel[None, 1](*args)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - begin) * 1000 / 20)
        record = {
            'version': name,
            'host_wall_ms': samples,
            'median_ms': statistics.median(samples),
            'max_abs': (out - golden).abs().max().item(),
            'workspace_bytes': ws.numel() * ws.element_size(),
            'intermediate_storage': 'GM' if depth else 'UB',
        }
        records.append(record)
        LOGGER.info('%s', record)
        if profile:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
                experimental_config=getattr(torch_npu.profiler, '_ExperimentalConfig')(
                    export_type=[torch_npu.profiler.ExportType.Text],
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f'./profile_ch08/{name}'),
            ) as prof:
                for _ in range(6):
                    kernel[None, 1](*args)
                    prof.step()
                torch.npu.synchronize()
    result = {
        'shape': {'M': 256, 'K': 128, 'N': 256},
        'cores': 1,
        'seed': 42,
        'timing': 'Host wall time per call after warmup; not pure device time',
        'python': sys.version,
        'torch': torch.__version__,
        'torch_npu': getattr(torch_npu, '__version__', 'unknown'),
        'versions': records,
    }
    Path('benchmark_ch08.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true')
    benchmark(parser.parse_args().profile)

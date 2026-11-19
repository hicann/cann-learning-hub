# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 7 章：Gather、直方图和按分数/来源行仲裁的确定性 Scatter。

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
def gather_rows(data, indices, out, count: pl.DT_UINT32):
    first = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
    for row in pl.range(first, count, stride):
        source = indices[0, row]
        for col in pl.range(0, 64):
            out[row, col] = data[source, col]


@pl.jit()
def gather_512(
    data: pl.Tensor[[1024, 64], pl.DT_FP32],
    indices: pl.Tensor[[1, 512], pl.DT_INT32],
    out: pl.Tensor[[512, 64], pl.DT_FP32],
):
    with pl.section_vector():
        gather_rows[256](data, indices, out, 512)


@pl.jit()
def gather_256(
    data: pl.Tensor[[1024, 64], pl.DT_FP32],
    indices: pl.Tensor[[1, 256], pl.DT_INT32],
    out: pl.Tensor[[256, 64], pl.DT_FP32],
):
    with pl.section_vector():
        gather_rows[256](data, indices, out, 256)


@pl.vector_function(mode="simt", max_threads=256)
def histogram_threads(data, hist):
    first = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
    for i in pl.range(first, 1024, stride):
        bucket = data[0, i]
        if bucket >= 0 and bucket < 16:
            pl.simt.atomic_add(hist, [0, bucket], 1)


@pl.jit()
def histogram_kernel(data: pl.Tensor[[1, 1024], pl.DT_INT32], hist: pl.Tensor[[1, 16], pl.DT_INT32]):
    with pl.section_vector():
        histogram_threads[256](data, hist)


@pl.vector_function(mode="simt", max_threads=256)
def find_max(indices, scores, best):
    first = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
    for i in pl.range(first, 512, stride):
        target = indices[0, i]
        pl.simt.atomic_max(best, [0, target], scores[0, i])


@pl.vector_function(mode="simt", max_threads=256)
def choose_winner(indices, scores, best, winner):
    first = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
    for i in pl.range(first, 512, stride):
        target = indices[0, i]
        if scores[0, i] == best[0, target]:
            candidate = pl.simt.cast(i, pl.DT_INT32)
            pl.simt.atomic_min(winner, [0, target], candidate)


@pl.vector_function(mode="simt", max_threads=256)
def scatter_winner(data, indices, winner, out):
    first = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
    for i in pl.range(first, 512, stride):
        target = indices[0, i]
        if winner[0, target] == i:
            for col in pl.range(0, 64):
                out[target, col] = data[i, col]


@pl.jit()
def max_kernel(
    indices: pl.Tensor[[1, 512], pl.DT_INT32],
    scores: pl.Tensor[[1, 512], pl.DT_FP32],
    best: pl.Tensor[[1, 1024], pl.DT_FP32],
):
    with pl.section_vector():
        find_max[256](indices, scores, best)


@pl.jit()
def winner_kernel(
    indices: pl.Tensor[[1, 512], pl.DT_INT32],
    scores: pl.Tensor[[1, 512], pl.DT_FP32],
    best: pl.Tensor[[1, 1024], pl.DT_FP32],
    winner: pl.Tensor[[1, 1024], pl.DT_INT32],
):
    with pl.section_vector():
        choose_winner[256](indices, scores, best, winner)


@pl.jit()
def scatter_kernel(
    data: pl.Tensor[[512, 64], pl.DT_FP32],
    indices: pl.Tensor[[1, 512], pl.DT_INT32],
    winner: pl.Tensor[[1, 1024], pl.DT_INT32],
    out: pl.Tensor[[1024, 64], pl.DT_FP32],
):
    with pl.section_vector():
        scatter_winner[256](data, indices, winner, out)


def deterministic_scatter(data, indices, scores, cores=4):
    if data.shape != (512, 64) or indices.shape != (1, 512) or scores.shape != (1, 512):
        raise ValueError('unexpected input shapes')
    if not torch.isfinite(scores).all().item():
        raise ValueError('scores must be finite')
    if not ((indices >= 0) & (indices < 1024)).all().item():
        raise ValueError('indices out of range')
    best = torch.full((1, 1024), -float('inf'), device=data.device, dtype=torch.float32)
    winner = torch.full((1, 1024), 2147483647, device=data.device, dtype=torch.int32)
    out = torch.zeros(1024, 64, device=data.device, dtype=torch.float32)
    # Same current stream preserves full-kernel stage dependencies.
    max_kernel[None, cores](indices, scores, best)
    winner_kernel[None, cores](indices, scores, best, winner)
    scatter_kernel[None, cores](data, indices, winner, out)
    return out, winner


def check_examples():
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    data = torch.randn(1024, 64, device=device, dtype=torch.float32)
    indices = torch.randint(0, 1024, (1, 512), device=device, dtype=torch.int32)
    indices[0, 0] = 0
    indices[0, -1] = 1023
    # Both 1 core (multiple rounds) and 8 cores (idle threads) are exercised.
    for cores in (1, 8):
        out = torch.full((512, 64), float('nan'), device=device, dtype=torch.float32)
        gather_512[None, cores](data, indices, out)
        torch.npu.synchronize()
        expected = data.index_select(0, indices.flatten().long())
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    scores = torch.randn(1, 1024, device=device, dtype=torch.float32)
    selected = torch.topk(scores, 256, dim=1).indices.to(torch.int32)
    out = torch.full((256, 64), float('nan'), device=device, dtype=torch.float32)
    gather_256[None, 8](data, selected, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, data.index_select(0, selected.flatten().long()), rtol=0, atol=0)
    buckets = (torch.arange(1024, device=device, dtype=torch.int32) % 18 - 1).reshape(1, 1024)
    hist = torch.zeros(1, 16, device=device, dtype=torch.int32)
    histogram_kernel[None, 4](buckets, hist)
    torch.npu.synchronize()
    valid = buckets[(buckets >= 0) & (buckets < 16)]
    expected = torch.bincount(valid.long(), minlength=16).to(torch.int32).reshape(1, 16)
    torch.testing.assert_close(hist, expected, rtol=0, atol=0)
    targets = (torch.arange(512, device=device, dtype=torch.int32) % 37).reshape(1, 512)
    targets[0, -1] = 1023
    tied_scores = (torch.arange(512, device=device, dtype=torch.float32) % 5 - 3).reshape(1, 512)
    src = data[:512].contiguous()
    out, winner = deterministic_scatter(src, targets, tied_scores)
    torch.npu.synchronize()
    # Independent Host ordering: descending score, ascending source ID.
    ids = targets.flatten().cpu().tolist()
    vals = tied_scores.flatten().cpu().tolist()
    chosen = {}
    for i in sorted(range(512), key=lambda i: (-vals[i], i)):
        chosen.setdefault(ids[i], i)
    expected = torch.zeros_like(out)
    expected_winner = torch.full_like(winner, 2147483647)
    for target, i in chosen.items():
        expected[target, :] = src[i, :]
        expected_winner[0, target] = i
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(winner, expected_winner, rtol=0, atol=0)
    LOGGER.info('PASS: Gather, TopK Gather, histogram, deterministic Scatter')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    check_examples()

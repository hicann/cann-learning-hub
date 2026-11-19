# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""第 11 章：独立稳定选择 Kernel 接第 10 章注意力；包含分层验收。

在 Ascend 950 上运行。本轮仅做源码和静态检查，尚无设备通过或性能结论。
"""

import argparse
import importlib.util
import itertools
import logging
import math
import os
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

import pypto_pro.language as pl
import torch
import torch.nn.functional as F
import torch_npu

os.environ.setdefault("TILE_FWK_DEVICE_ID", "0")

LOGGER = logging.getLogger(__name__)

# Course-local dependency; no import of gym or PyPTO test code.
reference_path = Path(__file__).resolve().parents[2] / "10_pypto_pro_sdpa_practice" / "src" / "sdpa_reference.py"

spec = importlib.util.spec_from_file_location("course_sdpa_reference", reference_path)
sdpa = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sdpa
spec.loader.exec_module(sdpa)


@lru_cache(maxsize=8)
def build_selector(num_candidates, num_selected):
    if num_candidates not in (16, 32) or not 1 <= num_selected <= num_candidates:
        raise ValueError("teaching selector supports 16/32 candidates")

    @pl.vector_function(mode="simt", max_threads=32)
    def rank_blocks(scores, selected):
        i = pl.simt.thread_idx().x
        if i < num_candidates:
            rank = 0
            for j in pl.range(0, num_candidates):
                if scores[0, j] > scores[0, i] or (scores[0, j] == scores[0, i] and j < i):
                    rank = rank + 1
            if rank < num_selected:
                selected[0, rank] = pl.simt.cast(i, pl.DT_INT32)

    @pl.jit()
    def select_kernel(
        scores: pl.Tensor[[1, num_candidates], pl.DT_FP32], selected: pl.Tensor[[1, num_selected], pl.DT_INT32]
    ):
        with pl.section_vector():
            rank_blocks[32](scores, selected)

    return select_kernel


def golden_attention(q, k, v, selected_cpu, causal):
    s = q.shape[1]
    ids = selected_cpu.to(device=q.device, dtype=torch.int64).flatten()
    original = (ids[:, None] * 128 + torch.arange(128, device=q.device)).flatten()
    original = original[original < s]
    ks = k.index_select(1, original)
    vs = v.index_select(1, original)
    if not causal:
        return F.scaled_dot_product_attention(q, ks, vs, dropout_p=0.0, is_causal=False)
    visible = original[None, :] <= torch.arange(s, device=q.device)[:, None]
    empty = ~visible.any(dim=-1)
    # Define the empty-row result explicitly; never rely on backend-specific all-masked behavior.
    safe = visible.clone()
    safe[empty, 0] = True
    golden = F.scaled_dot_product_attention(q, ks, vs, attn_mask=safe, dropout_p=0.0, is_causal=False)
    golden[:, empty, :] = 0
    return golden


def median_wall_ms(fn, warmup=3, repeats=5, inner=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - begin) * 1000 / inner)
    return statistics.median(samples)


def check_case(s=2048, h=64, b=1, ratio=0.25, causal=False, kind='random', benchmark=False):
    if s not in (2048, 4096) or ratio not in (0.125, 0.25, 0.5):
        raise ValueError("use RFC S and ratio")
    num_blocks = s // 128
    num_selected = math.ceil(num_blocks * ratio)
    torch.manual_seed(42)
    device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"
    torch.npu.set_device(device)
    q, k, v = (torch.randn(b, s, h, device=device, dtype=torch.float16) for _ in range(3))
    g = torch.randn(1, num_blocks, dtype=torch.float32)
    if kind == 'ties':
        g = torch.zeros_like(g)
    elif kind == 'boundary':
        g = (torch.arange(num_blocks, dtype=torch.float32) % 3 - 1).reshape(1, num_blocks)
    elif kind == 'late':
        g = torch.arange(num_blocks, dtype=torch.float32).reshape(1, num_blocks)
    elif kind == 'uniform':
        q.zero_()
        k.zero_()
    if not torch.isfinite(g).all().item():
        raise AssertionError('scores must be finite')
    expected = torch.argsort(g, dim=1, descending=True, stable=True)[:, :num_selected].to(torch.int32)
    scores = g.to(device)
    selected = torch.full((1, num_selected), -1, device=device, dtype=torch.int32)
    selector = build_selector(num_blocks, num_selected)

    def choose():
        selector[None, 1](scores, selected)

    choose()
    torch.npu.synchronize()
    # Stop before attention if selection differs: no tolerance for an ID mismatch.
    actual = selected.cpu()
    if actual.unique().numel() != num_selected:
        raise AssertionError('selector returned duplicate IDs')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    launch = sdpa.prepare_attention(q, k, v, selected=selected)
    out = launch(causal)
    torch.npu.synchronize()
    golden = golden_attention(q, k, v, expected, causal)
    if not torch.isfinite(out).all().item():
        raise AssertionError('output must be finite')
    torch.testing.assert_close(out, golden, rtol=1e-3, atol=1e-3)
    if causal:
        first_token = int(expected.min().item()) * 128
        if torch.count_nonzero(out[:, :first_token]).item() != 0:
            raise AssertionError('causal output above the diagonal must be zero')
    record = dict(
        b=b,
        s=s,
        h=h,
        ratio=ratio,
        causal=causal,
        kind=kind,
        selected=actual.flatten().tolist(),
        max_abs=(out.float() - golden.float()).abs().max().item(),
    )
    if benchmark:
        dense = sdpa.prepare_attention(q, k, v)
        dense_out = dense(causal)
        torch.npu.synchronize()
        dense_golden = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
        torch.testing.assert_close(dense_out, dense_golden, rtol=1e-3, atol=1e-3)

        def total():
            choose()
            launch(causal)  # Same stream: consumers run after the full selector.

        record['select_wall_ms'] = median_wall_ms(choose)
        record['attention_wall_ms'] = median_wall_ms(lambda: launch(causal))
        record['total_wall_ms'] = median_wall_ms(total)
        record['dense_wall_ms'] = median_wall_ms(lambda: dense(causal))
        record['dense_over_gated'] = record['dense_wall_ms'] / record['total_wall_ms']
    LOGGER.info('%s', record)
    return record


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument(
        '--benchmark', action='store_true', help='Host wall clock after correctness; use profiler for device time'
    )
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ['TILE_FWK_DEVICE_ID']))
    shapes = [(1, 2048, 64)] if not args.full else list(itertools.product((1, 2), (2048, 4096), (64, 128)))
    records = []
    for b, s, h in shapes:
        for ratio in (0.125, 0.25, 0.5):
            for causal in (False, True):
                records.append(check_case(s, h, b, ratio, causal, benchmark=args.benchmark))
    for kind in ('ties', 'boundary', 'late', 'uniform'):
        for causal in (False, True):
            records.append(check_case(causal=causal, kind=kind))
    import json

    Path('gated_kv_results.json').write_text(
        json.dumps(
            {'torch': torch.__version__, 'torch_npu': getattr(torch_npu, '__version__', 'unknown'), 'records': records},
            indent=2,
        )
    )

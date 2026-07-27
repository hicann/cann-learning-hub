# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Tuple
import logging
import argparse
import torch

try:
    import torch_npu
except ModuleNotFoundError:
    pass

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt
import asc.lib.host as host

logging.basicConfig(level=logging.INFO)

USE_CORE_NUM = 24
IS_TRANS_A = False
IS_TRANS_B = False
ENABLE_BIAS = True


@asc.jit(matmul_cube_only=True, always_compile=True)
def matmul_cube_only_kernel(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress,
                             bias: asc.GlobalAddress, tiling: asc.adv.TCubeTiling,
                             workspace: asc.GlobalAddress):
    if asc.ascend_is_aic():
        offset_a, offset_b, offset_c, offset_bias, tail_m, tail_n = calc_offsets(
            tiling, IS_TRANS_A, IS_TRANS_B)

        a_global = asc.GlobalTensor()
        b_global = asc.GlobalTensor()
        c_global = asc.GlobalTensor()
        bias_global = asc.GlobalTensor()
        a_global.set_global_buffer(a + offset_a)
        b_global.set_global_buffer(b + offset_b)
        c_global.set_global_buffer(c + offset_c)
        bias_global.set_global_buffer(bias + offset_bias)

        pipe = asc.TPipe()
        matmul = asc.adv.Matmul(
            a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, a_global.dtype, IS_TRANS_A),
            b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, b_global.dtype, IS_TRANS_B),
            c=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, c_global.dtype),
            bias=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, bias_global.dtype),
        )
        asc.adv.register_matmul(pipe, workspace, matmul, tiling)

        if asc.get_block_idx() < tiling.used_core_num:
            matmul.set_tensor_a(a_global, IS_TRANS_A)
            matmul.set_tensor_b(b_global, IS_TRANS_B)
            if tiling.is_bias:
                matmul.set_bias(bias_global)
            matmul.set_tail(tail_m, tail_n, tiling.k_a)
            matmul.iterate_all(c_global)
            matmul.end()

        asc.pipe_barrier(asc.PipeID.PIPE_ALL)


@asc.jit
def calc_offsets(tiling: asc.adv.TCubeTiling, is_trans_a: bool = False,
                 is_trans_b: bool = False) -> Tuple[int, int, int, int, int, int]:
    block_idx = asc.get_block_idx()
    m_single_blocks = tiling.m.ceildiv(tiling.single_core_m)
    n_index = block_idx // m_single_blocks
    m_index = block_idx % m_single_blocks
    offset_a = m_index * tiling.k_a * tiling.single_core_m
    if is_trans_a:
        offset_a = m_index * tiling.single_core_m
    offset_b = n_index * tiling.single_core_n
    if is_trans_b:
        offset_b = n_index * tiling.k_b * tiling.single_core_n
    offset_c = m_index * tiling.n * tiling.single_core_m + n_index * tiling.single_core_n
    offset_bias = n_index * tiling.single_core_n
    tail_m = tiling.m - m_index * tiling.single_core_m
    if tail_m <= 0 or tail_m >= tiling.single_core_m:
        tail_m = tiling.single_core_m
    tail_n = tiling.n - n_index * tiling.single_core_n
    if tail_n <= 0 or tail_n >= tiling.single_core_n:
        tail_n = tiling.single_core_n
    return offset_a, offset_b, offset_c, offset_bias, tail_m, tail_n


def matmul_cube_only_launch(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor,
                             tiling: asc.adv.TCubeTiling, device) -> torch.Tensor:
    size_m, size_k = a.shape
    _, size_n = b.shape
    c = torch.zeros((size_m, size_n), dtype=torch.float32, device=device)
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    matmul_cube_only_kernel[tiling.used_core_num, rt.current_stream()](a, b, c, bias, tiling, workspace)
    return c


def generate_tiling(m, n, k) -> asc.adv.TCubeTiling:
    matmul_tiling = host.MultiCoreMatmulTiling(host.get_ascendc_platform())
    matmul_tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, IS_TRANS_A)
    matmul_tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, IS_TRANS_B)
    matmul_tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    matmul_tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    matmul_tiling.set_dim(USE_CORE_NUM)
    matmul_tiling.set_org_shape(m, n, k)
    matmul_tiling.set_shape(m, n, k)
    matmul_tiling.enable_bias(ENABLE_BIAS)
    matmul_tiling.set_buffer_space(-1, -1, -1)

    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    return tiling


def matmul_cube_only_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    m, k, n = 128, 64, 30720
    a = torch.randint(-5, 5, (m, k), device=device).to(torch.float16)
    b = torch.randint(-5, 5, (k, n), device=device).to(torch.float16)
    bias = torch.randint(-5, 5, (1, n), device=device).to(torch.float32)

    if ENABLE_BIAS:
        golden = (torch.matmul(a.to(torch.float32), b.to(torch.float32)) + bias).to(torch.float32)
    else:
        golden = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    tiling = generate_tiling(m, n, k)
    c = matmul_cube_only_launch(a, b, bias, tiling, device)
    assert torch.allclose(c, golden, rtol=1e-3, atol=1e-3)
    logging.info(f"[INFO] Cube Only模式验证通过! M={m}, N={n}, K={k}, 核数={USE_CORE_NUM}, Bias={ENABLE_BIAS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="NPU", help="backend to run")
    parser.add_argument("-v", type=str, default=None, help="platform to run")
    args = parser.parse_args()
    backend = args.r
    platform = args.v
    if backend not in config.Backend.__members__:
        raise ValueError(f"Unsupported Backend! Supported: {list(config.Backend.__members__.keys())}")
    backend = config.Backend(backend)
    if platform is not None:
        platform_values = [p.value for p in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info("[INFO] start process sample matmul_cube_only.")
    matmul_cube_only_custom(backend, platform)
    logging.info("[INFO] Sample matmul_cube_only run success.")

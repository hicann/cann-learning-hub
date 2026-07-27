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

USE_CORE_NUM = 48
M_DIM = 256
N_DIM = 512
K_DIM = 256


@asc.jit(always_compile=True)
def matmul_mix_bias_kernel(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress,
                            bias: asc.GlobalAddress, tiling: asc.adv.TCubeTiling,
                            workspace: asc.GlobalAddress):
    offset_a, offset_b, offset_c, offset_bias, tail_m, tail_n = calc_offsets(tiling)

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
        a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, a_global.dtype, False),
        b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, b_global.dtype, False),
        c=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, c_global.dtype),
        bias=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, bias_global.dtype),
    )
    asc.adv.register_matmul(pipe, workspace, matmul, tiling)

    if asc.get_block_idx() < tiling.used_core_num:
        matmul.set_tensor_a(a_global, False)
        matmul.set_tensor_b(b_global, False)
        if tiling.is_bias:
            matmul.set_bias(bias_global)
        matmul.set_tail(tail_m, tail_n)
        matmul.iterate_all(c_global)
        matmul.end()

    asc.pipe_barrier(asc.PipeID.PIPE_ALL)


@asc.jit
def calc_offsets(tiling: asc.adv.TCubeTiling) -> Tuple[int, int, int, int, int, int]:
    block_idx = asc.get_block_idx()
    m_single_blocks = tiling.m.ceildiv(tiling.single_core_m)
    m_index = block_idx % m_single_blocks
    n_index = block_idx // m_single_blocks
    offset_a = m_index * tiling.k_a * tiling.single_core_m
    offset_b = n_index * tiling.single_core_n
    offset_c = m_index * tiling.n * tiling.single_core_m + n_index * tiling.single_core_n
    offset_bias = n_index * tiling.single_core_n
    tail_m = tiling.m - m_index * tiling.single_core_m
    if tail_m <= 0 or tail_m >= tiling.single_core_m:
        tail_m = tiling.single_core_m
    tail_n = tiling.n - n_index * tiling.single_core_n
    if tail_n <= 0 or tail_n >= tiling.single_core_n:
        tail_n = tiling.single_core_n
    return offset_a, offset_b, offset_c, offset_bias, tail_m, tail_n


def generate_tiling(m, n, k) -> asc.adv.TCubeTiling:
    matmul_tiling = host.MultiCoreMatmulTiling(host.get_ascendc_platform())
    matmul_tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, False)
    matmul_tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, False)
    matmul_tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    matmul_tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    matmul_tiling.set_dim(USE_CORE_NUM)
    matmul_tiling.set_org_shape(m, n, k)
    matmul_tiling.set_shape(m, n, k)
    matmul_tiling.enable_bias(True)
    matmul_tiling.set_buffer_space(-1, -1, -1)

    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    return tiling


def matmul_mix_bias_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"

    a = torch.randint(-5, 5, (M_DIM, K_DIM), device=device).to(torch.float16)
    b = torch.randint(-5, 5, (K_DIM, N_DIM), device=device).to(torch.float16)
    bias = torch.randint(-5, 5, (1, N_DIM), device=device).to(torch.float32)
    c = torch.zeros((M_DIM, N_DIM), dtype=torch.float32, device=device)

    golden = (torch.matmul(a.to(torch.float32), b.to(torch.float32)) + bias).to(torch.float32)

    tiling = generate_tiling(M_DIM, N_DIM, K_DIM)
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    matmul_mix_bias_kernel[USE_CORE_NUM // 2, rt.current_stream()](a, b, c, bias, tiling, workspace)

    assert torch.allclose(c, golden, rtol=1e-3, atol=1e-3)
    logging.info(f"[INFO] MIX+Bias验证通过! M={M_DIM}, N={N_DIM}, K={K_DIM}")


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
    logging.info("[INFO] start process sample matmul_mix_bias.")
    matmul_mix_bias_custom(backend, platform)
    logging.info("[INFO] Sample matmul_mix_bias run success.")

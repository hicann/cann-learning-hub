# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.

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


@asc.jit
def calc_offsets(tiling: asc.adv.TCubeTiling) -> Tuple[int, int, int, int]:
    block_idx = asc.get_block_idx()
    m_single_blocks = tiling.m.ceildiv(tiling.single_core_m)
    m_index = block_idx % m_single_blocks
    n_index = block_idx // m_single_blocks
    offset_a = m_index * tiling.k_a * tiling.single_core_m
    offset_b = n_index * tiling.single_core_n
    offset_c = m_index * tiling.n * tiling.single_core_m + n_index * tiling.single_core_n
    offset_bias = n_index * tiling.single_core_n
    return offset_a, offset_b, offset_c, offset_bias


@asc.jit(always_compile=True)
def matmul_activation_kernel(
    a: asc.GlobalAddress,
    b: asc.GlobalAddress,
    c: asc.GlobalAddress,
    bias: asc.GlobalAddress,
    alpha: float,
    tiling: asc.adv.TCubeTiling,
    workspace: asc.GlobalAddress,
):
    offset_a, offset_b, offset_c, offset_bias = calc_offsets(tiling)

    a_global = asc.GlobalTensor()
    b_global = asc.GlobalTensor()
    c_global = asc.GlobalTensor()
    bias_global = asc.GlobalTensor()
    a_global.set_global_buffer(a + offset_a)
    b_global.set_global_buffer(b + offset_b)
    c_global.set_global_buffer(c + offset_c)
    bias_global.set_global_buffer(bias + offset_bias)

    size = tiling.base_m * tiling.base_n * c.dtype.sizeof()
    pipe = asc.TPipe()
    out_queue = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(que=out_queue, num=1, len=size)

    matmul = asc.adv.Matmul(
        a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, a_global.dtype),
        b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, b_global.dtype),
        c=asc.adv.MatmulType(asc.TPosition.VECCALC, asc.CubeFormat.ND, c_global.dtype),
        bias=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, bias_global.dtype),
    )
    asc.adv.register_matmul(pipe, workspace, matmul, tiling)
    matmul.set_tensor_a(a_global)
    matmul.set_tensor_b(b_global)
    matmul.set_bias(bias_global)

    with matmul.iterate() as count:
        out_local = out_queue.alloc_tensor(c.dtype)
        matmul.get_tensor_c(out_local, en_sequential_write=True)
        asc.leaky_relu(
            out_local,
            out_local,
            alpha,
            count=tiling.base_m * tiling.base_n,
        )
        out_queue.enque(out_local)
        out_local = out_queue.deque(c.dtype)

        round_m = tiling.single_core_m // tiling.base_m
        start_offset = (
            count % round_m * tiling.base_m * tiling.n
            + count // round_m * tiling.base_n
        )
        params = asc.DataCopyParams(
            block_count=tiling.base_m,
            block_len=(
                tiling.base_n * c.dtype.sizeof()
                // asc.property(asc.DEFAULT_C0_SIZE)
            ),
            src_stride=0,
            dst_stride=(
                (tiling.n - tiling.base_n) * c.dtype.sizeof()
                // asc.property(asc.DEFAULT_C0_SIZE)
            ),
        )
        asc.data_copy(c_global[start_offset:], out_local, repeat_params=params)
        out_queue.free_tensor(out_local)

    matmul.end()
    asc.pipe_barrier(asc.PipeID.PIPE_ALL)


def matmul_activation_launch(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    alpha: float,
    tiling: asc.adv.TCubeTiling,
    device,
) -> torch.Tensor:
    size_m, _ = a.shape
    _, size_n = b.shape
    c = torch.zeros((size_m, size_n), dtype=torch.float32, device=device)
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    matmul_activation_kernel[tiling.used_core_num // 2, rt.current_stream()](
        a, b, c, bias, alpha, tiling, workspace
    )
    return c


def generate_tiling(m, n, k):
    matmul_tiling = host.MultiCoreMatmulTiling(host.get_ascendc_platform())
    matmul_tiling.set_a_type(
        host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, False
    )
    matmul_tiling.set_b_type(
        host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, False
    )
    matmul_tiling.set_c_type(
        host.TPosition.VECCALC, host.CubeFormat.ND, host.DataType.DT_FLOAT
    )
    matmul_tiling.set_bias_type(
        host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT
    )

    matmul_tiling.set_dim(2)
    matmul_tiling.set_org_shape(m, n, k)
    matmul_tiling.set_shape(m, n, k)
    matmul_tiling.enable_bias(True)
    matmul_tiling.set_traverse(host.MatrixTraverse.FIRSTM)
    matmul_tiling.set_fix_split(256, 128, -1)
    matmul_tiling.set_buffer_space(-1, -1, -1)

    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    return tiling


def matmul_activation_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    m, k, n = 1024, 256, 640
    a = torch.randint(-5, 5, (m, k), device=device).to(torch.float16)
    b = torch.randint(-5, 5, (k, n), device=device).to(torch.float16)
    bias = torch.randint(-5, 5, (1, n), device=device).to(torch.float32)
    alpha = 0.001

    tiling = generate_tiling(m, n, k)
    actual = matmul_activation_launch(a, b, bias, alpha, tiling, device)

    matmul = (
        torch.matmul(a.to(torch.float32), b.to(torch.float32)) + bias
    ).to(torch.float32)
    expected = torch.where(matmul >= 0, matmul, matmul * alpha)
    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="NPU", help="backend to run")
    parser.add_argument("-v", type=str, default=None, help="platform to run")
    args = parser.parse_args()
    backend = args.r
    platform = args.v
    if backend not in config.Backend.__members__:
        raise ValueError("Unsupported Backend! Supported: ['Model', 'NPU']")
    backend = config.Backend(backend)
    if platform is not None:
        platform_values = [item.value for item in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info("[INFO] start process sample matmul_activation.")
    matmul_activation_custom(backend, platform)
    logging.info("[INFO] Sample matmul_activation run success.")

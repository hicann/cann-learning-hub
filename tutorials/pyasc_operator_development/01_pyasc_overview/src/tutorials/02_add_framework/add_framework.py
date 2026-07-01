# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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

BUFFER_NUM = 2  # BUFFER_NUM should be 1 or 2
USE_CORE_NUM = 8
TILE_NUM = 8


logging.basicConfig(level=logging.INFO)


@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, block_length: int,
                tile_length: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    y_gm.set_global_buffer(y + offset)
    z_gm.set_global_buffer(z + offset)
    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, BUFFER_NUM)
    in_queue_y = asc.TQue(asc.TPosition.VECIN, BUFFER_NUM)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, BUFFER_NUM)
    pipe.init_buffer(in_queue_x, BUFFER_NUM, tile_length * x.dtype.sizeof())
    pipe.init_buffer(in_queue_y, BUFFER_NUM, tile_length * y.dtype.sizeof())
    pipe.init_buffer(out_queue_z, BUFFER_NUM, tile_length * z.dtype.sizeof())
    for i in range(TILE_NUM * BUFFER_NUM):
        copy_in(i, x_gm, y_gm, in_queue_x, in_queue_y, tile_length)
        compute(z_gm, in_queue_x, in_queue_y, out_queue_z, tile_length)
        copy_out(i, z_gm, out_queue_z, tile_length)


@asc.jit
def copy_in(i: int, x_gm: asc.GlobalAddress, y_gm: asc.GlobalAddress, in_queue_x: asc.TQue, in_queue_y: asc.TQue,
            tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    y_local = in_queue_y.alloc_tensor(y_gm.dtype)
    asc.data_copy(x_local, x_gm[i * tile_length:], tile_length)
    asc.data_copy(y_local, y_gm[i * tile_length:], tile_length)
    in_queue_x.enque(x_local)
    in_queue_y.enque(y_local)


@asc.jit
def compute(z_gm: asc.GlobalTensor, in_queue_x: asc.TQue, in_queue_y: asc.TQue, out_queue_z: asc.TQue,
            tile_length: asc.ConstExpr[int]):
    # "z_gm" is passed here to obtain dtype
    x_local = in_queue_x.deque(z_gm.dtype)
    y_local = in_queue_y.deque(z_gm.dtype)
    z_local = out_queue_z.alloc_tensor(z_gm.dtype)
    asc.add(z_local, x_local, y_local, tile_length)
    out_queue_z.enque(z_local)
    in_queue_x.free_tensor(x_local)
    in_queue_y.free_tensor(y_local)


@asc.jit
def copy_out(i: int, z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, tile_length: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm[i * tile_length:], z_local, tile_length)
    out_queue_z.free_tensor(z_local)


def vadd_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    total_length = z.numel()
    block_length = (total_length + USE_CORE_NUM - 1) // USE_CORE_NUM
    tile_length = block_length // TILE_NUM // BUFFER_NUM
    vadd_kernel[USE_CORE_NUM, rt.current_stream()](x, y, z, block_length, tile_length)
    return z


def vadd_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    size = 8 * 2048
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    z = vadd_launch(x, y)
    assert torch.allclose(z, x + y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend to run")
    parser.add_argument("-v", type=str, default=None, help="platform to run")
    args = parser.parse_args()
    backend = args.r
    platform = args.v
    if backend not in config.Backend.__members__:
        raise ValueError("Unsupported Backend! Supported: ['Model', 'NPU']")
    backend = config.Backend(backend)
    if platform is not None:
        platform_values = [platform.value for platform in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info("[INFO] start process sample add_framework.")
    vadd_custom(backend, platform)
    logging.info("[INFO] Sample add_framework run success.")

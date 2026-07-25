# Copyright (c) 2025 Huawei Technologies Co., Ltd.

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

BUFFER_NUM = 2
USE_CORE_NUM = 8
TILE_NUM = 8

logging.basicConfig(level=logging.INFO)


@asc.jit(always_compile=True)
def vrelu_kernel(x: asc.GlobalAddress, z: asc.GlobalAddress, block_length: int,
                 tile_length: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    z_gm.set_global_buffer(z + offset)
    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, BUFFER_NUM)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, BUFFER_NUM)
    pipe.init_buffer(in_queue_x, BUFFER_NUM, tile_length * x.dtype.sizeof())
    pipe.init_buffer(out_queue_z, BUFFER_NUM, tile_length * z.dtype.sizeof())

    for i in range(TILE_NUM * BUFFER_NUM):
        copy_in(i, x_gm, in_queue_x, tile_length)
        compute(z_gm, in_queue_x, out_queue_z, tile_length)
        copy_out(i, z_gm, out_queue_z, tile_length)


@asc.jit
def copy_in(i: int, x_gm: asc.GlobalAddress, in_queue_x: asc.TQue, tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    asc.data_copy(x_local, x_gm[i * tile_length:], tile_length)
    in_queue_x.enque(x_local)


@asc.jit
def compute(z_gm: asc.GlobalTensor, in_queue_x: asc.TQue, out_queue_z: asc.TQue, tile_length: asc.ConstExpr[int]):
    x_local = in_queue_x.deque(z_gm.dtype)
    z_local = out_queue_z.alloc_tensor(z_gm.dtype)
    asc.relu(z_local, x_local, tile_length)
    out_queue_z.enque(z_local)
    in_queue_x.free_tensor(x_local)


@asc.jit
def copy_out(i: int, z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, tile_length: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm[i * tile_length:], z_local, tile_length)
    out_queue_z.free_tensor(z_local)


def vrelu_launch(x: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    total_length = z.numel()
    block_length = (total_length + USE_CORE_NUM - 1) // USE_CORE_NUM
    tile_length = block_length // TILE_NUM // BUFFER_NUM
    vrelu_kernel[USE_CORE_NUM, rt.current_stream()](x, z, block_length, tile_length)
    return z


def vrelu_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    size = 8 * 2048
    x = torch.rand(size, dtype=torch.float32, device=device) - 0.5
    z = vrelu_launch(x)
    assert torch.allclose(z, torch.relu(x))


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
        platform_values = [platform.value for platform in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info("[INFO] start process sample relu_framework.")
    vrelu_custom(backend, platform)
    logging.info("[INFO] Sample relu_framework run success.")

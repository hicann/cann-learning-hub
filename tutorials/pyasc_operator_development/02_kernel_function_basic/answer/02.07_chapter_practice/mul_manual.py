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

USE_CORE_NUM = 8
BUFFER_NUM = 2
TILE_NUM = 8

logging.basicConfig(level=logging.INFO)


@asc.jit
def vmul_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, block_length: int):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset, block_length)
    y_gm.set_global_buffer(y + offset, block_length)
    z_gm.set_global_buffer(z + offset, block_length)

    tile_length = block_length // TILE_NUM // BUFFER_NUM
    data_type = x.dtype
    buffer_size = tile_length * BUFFER_NUM * data_type.sizeof()

    x_local = asc.LocalTensor(data_type, asc.TPosition.VECIN, 0, tile_length * BUFFER_NUM)
    y_local = asc.LocalTensor(data_type, asc.TPosition.VECIN, buffer_size, tile_length * BUFFER_NUM)
    z_local = asc.LocalTensor(data_type, asc.TPosition.VECOUT, buffer_size + buffer_size, tile_length * BUFFER_NUM)

    for i in range(TILE_NUM * BUFFER_NUM):
        buf_id = i % BUFFER_NUM

        asc.data_copy(x_local[buf_id * tile_length:], x_gm[i * tile_length:], tile_length)
        asc.data_copy(y_local[buf_id * tile_length:], y_gm[i * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.MTE2_V, buf_id)
        asc.wait_flag(asc.HardEvent.MTE2_V, buf_id)

        asc.mul(z_local[buf_id * tile_length:], x_local[buf_id * tile_length:],
                y_local[buf_id * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.V_MTE3, buf_id)
        asc.wait_flag(asc.HardEvent.V_MTE3, buf_id)

        asc.data_copy(z_gm[i * tile_length:], z_local[buf_id * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.MTE3_MTE2, buf_id)
        asc.wait_flag(asc.HardEvent.MTE3_MTE2, buf_id)


def vmul_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    total_length = z.numel()
    block_length = total_length // USE_CORE_NUM
    vmul_kernel[USE_CORE_NUM, rt.current_stream()](x, y, z, block_length)
    return z


def vmul_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    size = 8 * 2048
    x = torch.rand(size, dtype=torch.float32, device=device)
    y = torch.rand(size, dtype=torch.float32, device=device)
    z = vmul_launch(x, y)
    assert torch.allclose(z, x * y)


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
    logging.info("[INFO] start process sample mul.")
    vmul_custom(backend, platform)
    logging.info("[INFO] Sample mul run success.")

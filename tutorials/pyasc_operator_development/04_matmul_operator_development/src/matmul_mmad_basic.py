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

logging.basicConfig(level=logging.INFO)

M_DIM = 16
N_DIM = 16
K_DIM = 32
K_BLOCK = 16
K_ITER = K_DIM // K_BLOCK


@asc.jit(always_compile=True)
def matmul_mmad_kernel(a: asc.GlobalAddress, b: asc.GlobalAddress, c: asc.GlobalAddress):
    a_gm = asc.GlobalTensor()
    b_gm = asc.GlobalTensor()
    c_gm = asc.GlobalTensor()
    a_gm.set_global_buffer(a)
    b_gm.set_global_buffer(b)
    c_gm.set_global_buffer(c)

    pipe = asc.TPipe()

    a_l1 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.A1, addr=0, tile_size=256)
    b_l1 = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.B1, addr=0, tile_size=256)
    a_l0a = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.A2, addr=0, tile_size=256)
    b_l0b = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.B2, addr=0, tile_size=256)
    c_l0c = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.CO1, addr=0, tile_size=512)

    for i in range(K_ITER):
        nd2nz_a = asc.Nd2NzParams(
            1, K_BLOCK, M_DIM, 0, M_DIM, 16, 1, 0
        )
        asc.data_copy(a_l1, a_gm + i * M_DIM * K_BLOCK, intri_params=nd2nz_a)

        nd2nz_b = asc.Nd2NzParams(
            1, N_DIM, K_BLOCK, 0, K_BLOCK, 16, 1, 0
        )
        asc.data_copy(b_l1, b_gm + i * K_BLOCK * N_DIM, intri_params=nd2nz_b)

        event_id = pipe.fetch_event_id(event=asc.HardEvent.MTE2_MTE1)
        asc.set_flag(event=asc.HardEvent.MTE2_MTE1, event_id=event_id)
        asc.wait_flag(event=asc.HardEvent.MTE2_MTE1, event_id=event_id)

        load_params = asc.LoadData2DParams(
            start_index=0, repeat_times=1, src_stride=0, sid=0,
            dst_gap=0, if_transpose=False, addr_mode=0
        )
        asc.load_data(a_l0a, a_l1, load_params)
        asc.load_data(b_l0b, b_l1, load_params)

        event_id = pipe.fetch_event_id(event=asc.HardEvent.MTE1_M)
        asc.set_flag(event=asc.HardEvent.MTE1_M, event_id=event_id)
        asc.wait_flag(event=asc.HardEvent.MTE1_M, event_id=event_id)

        mmad_params = asc.MmadParams(m=M_DIM, n=N_DIM, k=K_BLOCK,
                                     cmatrix_init_val=(i == 0))
        asc.mmad(c_l0c, a_l0a, b_l0b, mmad_params)

        if i < K_ITER - 1:
            event_id = pipe.fetch_event_id(event=asc.HardEvent.MTE1_MTE2)
            asc.set_flag(event=asc.HardEvent.MTE1_MTE2, event_id=event_id)
            asc.wait_flag(event=asc.HardEvent.MTE1_MTE2, event_id=event_id)

    event_id = pipe.fetch_event_id(event=asc.HardEvent.M_FIX)
    asc.set_flag(event=asc.HardEvent.M_FIX, event_id=event_id)
    asc.wait_flag(event=asc.HardEvent.M_FIX, event_id=event_id)

    fixpipe_params = asc.FixpipeParamsV220(
        n_size=N_DIM, m_size=M_DIM, src_stride=0,
        dst_stride=N_DIM * 4 // 32,
        quant_pre=asc.QuantModes.NoQuant, deq_scalar=0,
        nd_num=1, src_nd_stride=0, dst_nd_stride=0,
        relu_en=False, unit_flag=0, is_channel_split=False
    )
    asc.fixpipe(c_gm, c_l0c, fixpipe_params)

    asc.pipe_barrier(asc.PipeID.PIPE_ALL)


def matmul_mmad_custom(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"

    a = torch.randint(-5, 5, (M_DIM, K_DIM), device=device).to(torch.float16)
    b = torch.randint(-5, 5, (K_DIM, N_DIM), device=device).to(torch.float16)
    c = torch.zeros((M_DIM, N_DIM), dtype=torch.float32, device=device)

    matmul_mmad_kernel[1, rt.current_stream()](a, b, c)

    golden = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    assert torch.allclose(c, golden, rtol=1e-3, atol=1e-3)
    logging.info(f"[INFO] 基础mmad验证通过! M={M_DIM}, N={N_DIM}, K={K_DIM}, K_BLOCK={K_BLOCK}, K_ITER={K_ITER}")


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
    logging.info("[INFO] start process sample matmul_mmad_basic.")
    matmul_mmad_custom(backend, platform)
    logging.info("[INFO] Sample matmul_mmad_basic run success.")

#!/usr/bin/env python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import argparse
import os
import shutil
import tempfile
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Callable
from multiprocessing.managers import ListProxy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

OP_EXE_DIR = Path('.').resolve()
INPUT_DIR = "input"
GOLDEN_DIR = "golden"
OUTPUT_DIR = "output"


class IOCategory(Enum):
    INPUTS = "inputs"
    OUTPUTS = "outputs"


def remove_previous_case() -> None:
    for io_dir in [INPUT_DIR, GOLDEN_DIR, OUTPUT_DIR]:
        op_io_dir = OP_EXE_DIR / io_dir
        if op_io_dir.exists():
            shutil.rmtree(op_io_dir)


def gen_output_dir(world_size: int) -> None:
    for chip_id in range(world_size):
        output_dir = OP_EXE_DIR / OUTPUT_DIR / f'chip_{chip_id}'
        output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_tensor_dir(chip_id: int, category: IOCategory) -> Path:
    io_dir = (
        INPUT_DIR
        if category == IOCategory.INPUTS
        else GOLDEN_DIR
    )
    return OP_EXE_DIR / io_dir / f"chip_{chip_id}"


def save_tensor_to_bin(
    tensor: torch.Tensor,
    tensor_name: str,
    chip_id: int,
    category: IOCategory,
) -> None:
    out_dir = _resolve_tensor_dir(chip_id, category)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = tensor.detach().cpu().contiguous().view(torch.int8).numpy()
    raw.tofile(out_dir / f"{tensor_name}_{chip_id}.bin")


_BFLOAT16_NEEDS_FP32_FOR_NPY: bool | None = None


def _numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16

        return bfloat16
    except ModuleNotFoundError:
        try:
            import tensorflow as tf

            return tf.bfloat16.as_numpy_dtype
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "bfloat16 需要 ml-dtypes 或 tensorflow 支持，请安装 "
                "`pip3 install ml-dtypes` 或 `pip3 install tensorflow`。"
            ) from exc


def _numpy_float8_e5m2():
    try:
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "float8_e5m2 需要 ml-dtypes 支持，请安装 `pip3 install ml-dtypes`。"
        ) from exc


def _numpy_float8_e8m0():
    try:
        from en_dtypes import float8_e8m0

        return float8_e8m0
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "float8_e8m0 需要 en-dtypes 支持，请安装 `pip3 install en-dtypes`。"
        ) from exc


def numpy_to_torch_tensor(np_array: np.ndarray | None) -> torch.Tensor | None:
    if np_array is None:
        return None

    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        return torch.from_numpy(np_array.view(np.int16)).view(torch.bfloat16)

    if np_dtype == "float8_e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError(
                f"当前 PyTorch 版本 {torch.__version__} 不支持 torch.float8_e5m2。"
            )
        return torch.from_numpy(np_array.view(np.uint8)).view(torch.float8_e5m2)

    if np_dtype == "float8_e8m0":
        return torch.from_numpy(np_array.view(np.uint8)).view(torch.uint8)

    return torch.from_numpy(np_array)


def torch_to_numpy_tensor(torch_tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(torch_tensor, torch.Tensor):
        raise TypeError(f"只支持 torch.Tensor，实际得到 {type(torch_tensor)}")

    if torch_tensor.dtype == torch.bfloat16:
        return torch_tensor.view(torch.int16).cpu().numpy().view(_numpy_bfloat16())

    if torch_tensor.dtype == torch.float8_e5m2:
        return torch_tensor.view(torch.uint8).cpu().numpy().view(_numpy_float8_e5m2())

    return torch_tensor.detach().cpu().numpy()


def _round_mantissa(values: np.ndarray, round_mode: str = "rint") -> np.ndarray:
    if round_mode in ("rint", "even"):
        return np.rint(values)
    if round_mode in ("round", "nearest"):
        return np.sign(values) * np.floor(np.abs(values) + np.array([0.5], dtype=values.dtype))
    if round_mode == "floor":
        return np.floor(values)
    if round_mode == "ceil":
        return np.ceil(values)
    if round_mode == "trunc":
        return np.trunc(values)
    raise ValueError(f"不支持的 round_mode: {round_mode}")


def _reshape_to_blocks(
    fp_array: np.ndarray, axis: int, block_size: int
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    fp_array = np.expand_dims(fp_array, axis=axis + 1)
    original_shape = fp_array.shape

    pad = [[0, 0] for _ in range(fp_array.ndim)]
    remainder = original_shape[axis] % block_size
    if remainder:
        pad[axis][1] = block_size - remainder
        fp_array = np.pad(fp_array, pad, mode="constant")

    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] //= block_size
    return fp_array.reshape(reshape), original_shape, padded_shape


def _undo_reshape_to_blocks(
    fp_array: np.ndarray,
    axis: int,
    original_shape: tuple[int, ...],
    padded_shape: tuple[int, ...],
) -> np.ndarray:
    fp_array = fp_array.reshape(padded_shape)
    if padded_shape != original_shape:
        slices = tuple(slice(0, n) for n in original_shape)
        fp_array = fp_array[slices]
    return np.squeeze(fp_array, axis=axis + 1)


def _calculate_shared_exp(fp_array: np.ndarray, scale_axis: int) -> np.ndarray:
    fp32_exp_bias = 127
    fp32_min_normal = 2 ** (-fp32_exp_bias + 1)
    element_emax = 15  # for float8_e5m2

    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
    share_exp = np.floor(
        np.log2(fp_abs_max.astype(np.float32) + fp32_min_normal * (fp_abs_max == 0))
    ) - element_emax
    share_exp[fp_abs_max == 0] = -float("inf")
    return share_exp


def _quantize_to_float8_e5m2(
    fp_array: np.ndarray, share_exp: np.ndarray, round_mode: str
) -> np.ndarray:
    exp_bits = 5
    mantissa_bits = 2

    scaled = fp_array / (2 ** share_exp)
    private_exp = np.floor(
        np.log2(np.abs(scaled.astype(np.float32)) + (scaled == 0))
    ).astype(fp_array.dtype, copy=False)

    min_exp = -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)

    scaled = scaled / (2 ** private_exp) * (2 ** mantissa_bits)
    scaled = _round_mantissa(scaled, round_mode)
    return scaled / (2 ** mantissa_bits) * (2 ** private_exp)


def _pad_to_even_length(array: np.ndarray, axis: int) -> np.ndarray:
    if array.shape[axis] % 2 == 0:
        return array

    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, 1)
    return np.pad(array, pad_width, mode="constant", constant_values=2 ** -127)


def mx_quantize_e5m2(
    fp_array: np.ndarray,
    axis: int = -1,
    block_size: int = 32,
    round_mode: str = "rint",
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fp_array, np.ndarray):
        raise TypeError(f"输入必须是 numpy.ndarray，实际得到 {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise TypeError(f"不支持的输入类型: {fp_array.dtype.name}")

    axis = fp_array.ndim + axis if axis < 0 else axis
    fp_array, original_shape, padded_shape = _reshape_to_blocks(fp_array, axis, block_size)

    share_exp = _calculate_shared_exp(fp_array, scale_axis=axis + 1)
    scale_emax = 2 ** 7 - 1
    share_exp[share_exp > scale_emax] = np.nan
    share_exp[share_exp < -scale_emax] = -scale_emax

    elements = _quantize_to_float8_e5m2(fp_array, share_exp, round_mode)
    elements = _undo_reshape_to_blocks(elements, axis, original_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)

    if elements.dtype.name == "bfloat16":
        elements = elements.astype(np.float32, copy=False)

    scales = (2 ** share_exp)
    scales = _pad_to_even_length(scales, axis=axis).astype(_numpy_float8_e8m0(), copy=False)
    elements = elements.astype(_numpy_float8_e5m2(), copy=False)
    return scales, elements


def quantize_tokens_mx(
    quant_mode: int, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if quant_mode != 4:
        raise ValueError(f"仅支持 quant_mode=4，当前为 {quant_mode}")

    scales_np, tokens_np = mx_quantize_e5m2(torch_to_numpy_tensor(tokens))
    quantized_tokens = numpy_to_torch_tensor(tokens_np).view(torch.int8)
    dynamic_scales = numpy_to_torch_tensor(scales_np).view(torch.int8)
    return quantized_tokens, dynamic_scales


def gen_dispatch_input_bin_file(
    tokens_per_chip: list[torch.Tensor],
    dst_expert_indices_per_chip: list[torch.Tensor],
    world_size: int,
    saver: Callable[[torch.Tensor, str, int, IOCategory], None],
) -> None:
    for chip_id in range(world_size):
        saver(tokens_per_chip[chip_id], "x", chip_id, IOCategory.INPUTS)
        saver(
            dst_expert_indices_per_chip[chip_id],
            "expert_ids",
            chip_id,
            IOCategory.INPUTS,
        )


def gen_dispatch_output_bin_file(
    recv_tokens: list[torch.Tensor],
    recv_dynamics_scales: list[torch.Tensor],
    recv_src_info: list[torch.Tensor],
    recv_prefix_count: list[torch.Tensor],
    recv_info_per_expert: list[torch.Tensor],
    world_size: int,
    do_quant: bool,
    saver: Callable[[torch.Tensor, str, int, IOCategory], None],
) -> None:
    for chip_id in range(world_size):
        saver(recv_tokens[chip_id], "expand_x", chip_id, IOCategory.OUTPUTS)
        if do_quant:
            saver(
                recv_dynamics_scales[chip_id],
                "dynamic_scales",
                chip_id,
                IOCategory.OUTPUTS,
            )
        saver(
            recv_src_info[chip_id],
            "assist_info_for_combine",
            chip_id,
            IOCategory.OUTPUTS,
        )
        saver(
            recv_info_per_expert[chip_id],
            "expert_token_nums",
            chip_id,
            IOCategory.OUTPUTS,
        )
        saver(
            recv_prefix_count[chip_id],
            "ep_recv_count",
            chip_id,
            IOCategory.OUTPUTS,
        )


def gen_combine_input_bin_file(
    recv_tokens: list[torch.Tensor],
    topk_weights_per_chip: list[torch.Tensor],
    h: int,
    world_size: int,
    a_moe: int,
    saver: Callable[[torch.Tensor, str, int, IOCategory], None],
) -> None:
    def pad_expand_x(expand_x: torch.Tensor, ep_id: int) -> torch.Tensor:
        target_rows = a_moe
        pad_rows = target_rows - expand_x.size(0)
        if pad_rows <= 0:
            return expand_x
        padding = torch.zeros((pad_rows, h), dtype=expand_x.dtype)
        return torch.cat([expand_x, padding], dim=0)

    for chip_id in range(world_size):
        saver(
            pad_expand_x(recv_tokens[chip_id], chip_id),
            "expand_x",
            chip_id,
            IOCategory.INPUTS,
        )
        saver(
            topk_weights_per_chip[chip_id],
            "expert_scales",
            chip_id,
            IOCategory.INPUTS,
        )


def gen_combine_output_bin_file(
    tokens_per_chip: list[torch.Tensor],
    world_size: int,
    saver: Callable[[torch.Tensor, str, int, IOCategory], None],
) -> None:
    for chip_id in range(world_size):
        saver(tokens_per_chip[chip_id], "x", chip_id, IOCategory.OUTPUTS)


def gen_tokens_per_chip(
    bs: int, h: int, token_dtype: torch.dtype, world_size: int, bound: int = 10
) -> list[torch.Tensor]:
    return [
        torch.empty((bs, h), dtype=token_dtype).uniform_(-bound, bound)
        for _ in range(world_size)
    ]


def gen_dst_expert_indices_per_chip(
    bs: int, k: int, world_size: int, num_experts: int
) -> list[torch.Tensor]:
    def gen_balanced_dst_expert_indices(round_idx: int) -> torch.Tensor:
        return torch.arange(round_idx, round_idx + num_experts, dtype=torch.int32) % num_experts

    flat = torch.cat(
        [
            gen_balanced_dst_expert_indices(round_idx)
            for round_idx in range(bs * k * world_size // num_experts + 1)
        ]
    )[: bs * k * world_size]
    return list(flat.view(-1, k).chunk(world_size))


def gen_topk_weights_per_chip(bs: int, k: int, world_size: int) -> list[torch.Tensor]:
    return [
        torch.empty((bs, k), dtype=torch.float32).uniform_(-1, 1)
        for _ in range(world_size)
    ]


def dispatch(
    rank: int,
    init_method: str,
    tokens_per_chip: list[torch.Tensor],
    dst_expert_indices_per_chip: list[torch.Tensor],
    world_size: int,
    num_experts_total: int,
    quant_mode: int,
    expert_recv_info_type: int,
    dispatch_outputs: ListProxy,
) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init_method)
    print(f"[dispatch][rank={rank}] process group ready", flush=True)

    local_tokens_raw = tokens_per_chip[rank]
    dst_expert_indices_raw = dst_expert_indices_per_chip[rank]

    feat_dim = local_tokens_raw.size(1)
    topk = dst_expert_indices_raw.size(1)

    local_tokens = local_tokens_raw.repeat_interleave(repeats=topk, dim=0)
    dst_expert_indices = dst_expert_indices_raw.flatten()

    dst_expert_indices, etk_indices = dst_expert_indices.sort(stable=True)
    etk_indices = etk_indices.to(torch.int32)
    local_tokens_in_etk_order = local_tokens[etk_indices]

    if quant_mode > 0:
        local_tokens_to_send, local_dynamic_scales = quantize_tokens_mx(
            quant_mode, local_tokens_in_etk_order
        )
    else:
        local_tokens_to_send = local_tokens_in_etk_order
        local_dynamic_scales = local_tokens_in_etk_order.new_empty(
            (local_tokens_in_etk_order.size(0), 1), dtype=torch.float32
        )

    local_tk_idx_in_etk_order = torch.arange(
        dst_expert_indices_raw.numel(), dtype=torch.int32
    )[etk_indices]
    local_src_info_to_send = torch.stack(
        [
            torch.full_like(local_tk_idx_in_etk_order, rank),
            local_tk_idx_in_etk_order // topk,
            local_tk_idx_in_etk_order % topk,
        ],
        dim=1,
    )

    experts_per_rank = num_experts_total // world_size
    dst_local_expert_indices_to_send = dst_expert_indices % experts_per_rank

    send_token_count = torch.bincount(
        dst_expert_indices // experts_per_rank, minlength=world_size
    )
    recv_token_count = torch.empty_like(send_token_count)
    dist.all_to_all_single(output=recv_token_count, input=send_token_count)

    total_recv_token_count = int(recv_token_count.sum().item())
    send_splits = send_token_count.tolist()
    recv_splits = recv_token_count.tolist()

    recv_tokens = local_tokens_to_send.new_empty((total_recv_token_count, feat_dim))
    dynamic_scales_ele_per_token = ((feat_dim + 31) // 32) if quant_mode == 4 else 1
    recv_dynamic_scales = local_dynamic_scales.new_empty(
        (total_recv_token_count, dynamic_scales_ele_per_token)
    )
    recv_src_info = local_src_info_to_send.new_empty((total_recv_token_count, 3))
    recv_dst_local_expert_indices = dst_local_expert_indices_to_send.new_empty(
        (total_recv_token_count,)
    )

    dist.all_to_all_single(
        output=recv_tokens,
        input=local_tokens_to_send,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    dist.all_to_all_single(
        output=recv_dynamic_scales,
        input=local_dynamic_scales,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    dist.all_to_all_single(
        output=recv_src_info,
        input=local_src_info_to_send,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    dist.all_to_all_single(
        output=recv_dst_local_expert_indices,
        input=dst_local_expert_indices_to_send,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )

    recv_dst_local_expert_indices, output_indices = recv_dst_local_expert_indices.sort(
        stable=True
    )
    recv_tokens = recv_tokens[output_indices]
    recv_dynamic_scales = recv_dynamic_scales[output_indices]
    recv_src_info = recv_src_info[output_indices]

    recv_prefix_count = torch.bincount(
        recv_src_info[:, 0] + recv_dst_local_expert_indices * world_size,
        minlength=experts_per_rank * world_size,
    ).cumsum(dim=0).to(torch.int32)

    recv_token_count_per_expert = torch.bincount(
        recv_dst_local_expert_indices, minlength=experts_per_rank
    )
    recv_info_per_expert = (
        recv_token_count_per_expert
        if expert_recv_info_type == 1
        else recv_token_count_per_expert.cumsum(dim=0)
    )

    dispatch_outputs[rank].extend(
        [
            recv_tokens,
            recv_dynamic_scales,
            recv_src_info,
            recv_prefix_count,
            recv_info_per_expert,
        ]
    )

    dist.destroy_process_group()
    print(f"[dispatch][rank={rank}] process group destroyed", flush=True)

def comm_quant_round_trip(
        comm_quant_mode: int, tensor: torch.Tensor, token_dtype: torch.dtype,  
) -> torch.Tensor:

    def comm_quant(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quant_dtype = torch.int8 if comm_quant_mode == 2 else torch.int16
        quant_dtype_max = 127.0 if comm_quant_mode == 2 else 2047.0

        blocks = tensor.view(-1, element_per_block).to(torch.float32)
        blocks_max, _ = blocks.abs().max(dim=-1, keepdim=True)
        blocks_scales = blocks_max / quant_dtype_max
        blocks_scales[blocks_scales==0.0] = 1.0 # block_scales = 0 的时候改成 1，避免除 0
        blocks_quant = (blocks / blocks_scales).to(torch.float16).round().to(quant_dtype)

        return blocks_quant, blocks_scales.to(token_dtype)

    def comm_dequant(blocks_quant: torch.Tensor, blocks_scales: torch.Tensor) -> torch.Tensor:
        blocks_scales = blocks_scales.to(torch.float32)
        blocks_quant = blocks_quant.to(torch.float16).to(torch.float32)
        blocks_dequant = (blocks_quant * blocks_scales).to(token_dtype)
        tensor_dequant = blocks_dequant.view(-1, h + token_pad_length)

        return tensor_dequant
    
    if comm_quant_mode == 0:
        return tensor
    
    h = tensor.size(1)

    block_size = 32  # 核函数侧，一个 block 有 32 字节
    computation_dtype = torch.float32

    computation_dtype_size = torch.finfo(computation_dtype).bits // 8   # 用于量化计算的数据类型的字节数
    element_per_block =  block_size // computation_dtype_size # element_per_block 值为 8

    token_pad_length = (element_per_block - (tensor.size(1) % element_per_block)) % element_per_block
    padded_tensor = torch.cat([tensor, tensor.new_zeros(tensor.size(0), token_pad_length)], dim=1)

    padded_tensor_after_round_trip = comm_dequant(*comm_quant(padded_tensor))
    return padded_tensor_after_round_trip[:, :tensor.size(1)]

def combine(
    rank: int,
    init_method: str,
    token_dtype: torch.dtype,
    combine_inputs: list[list[torch.Tensor]],
    topk_weights_per_chip: list[torch.Tensor],
    world_size: int,
    combine_outputs: ListProxy,
) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init_method)
    print(f"[combine][rank={rank}] process group ready", flush=True)

    local_send_tokens = combine_inputs[rank][0]
    local_send_src_info = combine_inputs[rank][2]

    rank_first_indices = local_send_src_info[:, 0].argsort(stable=True)
    local_send_tokens = local_send_tokens[rank_first_indices]
    local_send_transfer_idx = local_send_src_info[rank_first_indices][:, 1:].contiguous()

    send_token_count = torch.bincount(local_send_src_info[:, 0], minlength=world_size)
    recv_token_count = torch.empty_like(send_token_count)
    dist.all_to_all_single(output=recv_token_count, input=send_token_count)

    total_recv_token_count = int(recv_token_count.sum().item())
    send_splits = send_token_count.tolist()
    recv_splits = recv_token_count.tolist()

    recv_tokens = local_send_tokens.new_empty(
        (total_recv_token_count, local_send_tokens.size(1))
    )
    recv_transfer_idx = local_send_transfer_idx.new_empty((total_recv_token_count, 2))

    dist.all_to_all_single(
        output=recv_tokens,
        input=local_send_tokens,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    dist.all_to_all_single(
        output=recv_transfer_idx,
        input=local_send_transfer_idx,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )

    topk = topk_weights_per_chip[0].size(1)
    transfer_idx_1d = recv_transfer_idx[:, 0] * topk + recv_transfer_idx[:, 1]
    transfer_order_indices = transfer_idx_1d.argsort(stable=True)

    recv_tokens = recv_tokens[transfer_order_indices]
    recv_tokens = comm_quant_round_trip(2, recv_tokens, token_dtype)
    local_topk_weights = topk_weights_per_chip[rank].flatten()
    scaled_recv_tokens = recv_tokens.to(torch.float32) * local_topk_weights.to(
        torch.float32
    ).unsqueeze(-1)
    combined_tokens = torch.stack(
        [chunk.sum(dim=0) for chunk in scaled_recv_tokens.split(topk)]
    )

    combine_outputs[rank].append(combined_tokens.to(token_dtype))

    dist.destroy_process_group()
    print(f"[combine][rank={rank}] process group destroyed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dispatch/combine test cases"
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--chip-num-per-server", type=int, default=2)
    parser.add_argument("--moe-expert-num", type=int, default=None)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--h", type=int, default=7168)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--token-dtype-choice", type=int, choices=[0, 1], default=1)
    parser.add_argument("--quant-mode", type=int, choices=[0, 4], default=4)
    parser.add_argument("--expert-recv-info-type", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def _spawn_distributed(
    fn,
    fn_args: tuple,
    world_size: int,
    temp_dir: str,
    init_name: str,
) -> None:
    init_file = os.path.join(temp_dir, init_name)
    init_method = f"file://{init_file.replace(os.sep, '/')}"
    mp.spawn(fn, args=(init_method, *fn_args), nprocs=world_size, join=True)


def main() -> None:
    args = parse_args()

    random_seed = args.random_seed
    quant_mode = args.quant_mode
    expert_recv_info_type = args.expert_recv_info_type
    do_quant = quant_mode > 0

    server_num = 1
    chip_num_per_server = args.chip_num_per_server
    shared_expert_num = 0
    shared_expert_rank_num = 0
    moe_expert_num = (
        chip_num_per_server * 4 if args.moe_expert_num is None else args.moe_expert_num
    )

    world_size = server_num * chip_num_per_server
    rank_num_per_shared_expert = (
        0 if shared_expert_num == 0 else shared_expert_rank_num // shared_expert_num
    )
    moe_rank_num = world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    num_experts = moe_expert_num

    bs = args.bs
    h = args.h
    k = args.k
    global_bs = bs * world_size
    max_shared_group_num = (
        ceil(world_size / rank_num_per_shared_expert)
        if rank_num_per_shared_expert > 0
        else 0
    )
    a_moe = global_bs * min(local_moe_expert_num, k)

    token_dtype = torch.bfloat16 if args.token_dtype_choice == 0 else torch.float16

    print(f"{random_seed=}")
    print(f"{chip_num_per_server=}")
    print(f"{moe_expert_num=}")
    print(f"{bs=}")
    print(f"{h=}")
    print(f"{k=}")
    print(f"{token_dtype=}")
    print(f"{quant_mode=}")
    print(f"{expert_recv_info_type=}")

    torch.manual_seed(random_seed)
    bound = 10

    tokens_per_chip = gen_tokens_per_chip(bs, h, token_dtype, world_size, bound)
    dst_expert_indices_per_chip = gen_dst_expert_indices_per_chip(
        bs, k, world_size, num_experts
    )
    topk_weights_per_chip = gen_topk_weights_per_chip(bs, k, world_size)

    manager = mp.Manager()
    dispatch_outputs = manager.list([manager.list() for _ in range(world_size)])
    combine_outputs = manager.list([manager.list() for _ in range(world_size)])

    temp_dir = tempfile.mkdtemp(prefix="dispatch_combine_")
    try:
        _spawn_distributed(
            dispatch,
            (
                tokens_per_chip,
                dst_expert_indices_per_chip,
                world_size,
                num_experts,
                quant_mode,
                expert_recv_info_type,
                dispatch_outputs,
            ),
            world_size,
            temp_dir,
            "dispatch_init",
        )

        combine_inputs = [
            [torch.empty_like(output[0], dtype=token_dtype).uniform_(-bound, bound)]
            + list(output[1:])
            for output in dispatch_outputs
        ]

        _spawn_distributed(
            combine,
            (
                token_dtype,
                combine_inputs,
                topk_weights_per_chip,
                world_size,
                combine_outputs,
            ),
            world_size,
            temp_dir,
            "combine_init",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    remove_previous_case()
    gen_output_dir(world_size)
    saver = lambda tensor, tensor_name, chip_id, category: save_tensor_to_bin(
        tensor=tensor,
        tensor_name=tensor_name,
        chip_id=chip_id,
        category=category,
    )

    gen_dispatch_input_bin_file(
        tokens_per_chip,
        dst_expert_indices_per_chip,
        world_size,
        saver,
    )
    gen_dispatch_output_bin_file(
        [output[0] for output in dispatch_outputs],
        [output[1] for output in dispatch_outputs],
        [output[2] for output in dispatch_outputs],
        [output[3] for output in dispatch_outputs],
        [output[4] for output in dispatch_outputs],
        world_size,
        do_quant,
        saver,
    )
    gen_combine_input_bin_file(
        [item[0] for item in combine_inputs],
        topk_weights_per_chip,
        h,
        world_size,
        a_moe,
        saver,
    )
    gen_combine_output_bin_file(
        [output[0] for output in combine_outputs],
        world_size,
        saver,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

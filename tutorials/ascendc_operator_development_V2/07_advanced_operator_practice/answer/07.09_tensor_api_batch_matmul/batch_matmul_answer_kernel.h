/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_MATMUL_ANSWER_KERNEL_H
#define BATCH_MATMUL_ANSWER_KERNEL_H

#include "c_api/asc_simd.h"
#include "tensor_api/tensor.h"

template <uint32_t Value, uint32_t Align>
struct ceil_align {
    static constexpr uint32_t value = (Value + Align - 1) / Align * Align;
};

template <
    typename T, uint32_t B, uint32_t M, uint32_t K, uint32_t N,
    uint32_t L1_BATCH_SIZE, uint32_t L0_BATCH_SIZE>
__cube__ __global__ void batch_matmul_kernel(__gm__ T *a, __gm__ T *b, __gm__ T *c, __gm__ T *bias)
{
    asc_init();
    using namespace asc::te;

    // 各级缓冲的单 batch 分形尺寸：L1 A/B 的 C0 为 32/sizeof(T)，L0C/BiasTable 为 float 类型
    constexpr uint32_t C0 = 32 / sizeof(T);
    constexpr uint32_t L1_A_SIZE = ceil_align<M, 16>::value * ceil_align<K, C0>::value;
    constexpr uint32_t L1_B_SIZE = ceil_align<K, 16>::value * ceil_align<N, C0>::value;
    constexpr uint32_t L0_A_SIZE = L1_A_SIZE;
    constexpr uint32_t L0_B_SIZE = ceil_align<K, C0>::value * ceil_align<N, 16>::value;
    constexpr uint32_t L0_C_SIZE = ceil_align<M, 16>::value * ceil_align<N, 16>::value;

    // ---- [Answer-1] batch 维多核切分：每核负责一段连续 batch，余数分给编号靠前的核 ----
    uint32_t block_index = block_idx;
    uint32_t single_core_b = B / block_num;
    uint32_t single_core_res_b = B % block_num;
    uint32_t actual_single_core_b = single_core_b + (single_core_res_b > block_index ? 1 : 0);
    uint32_t batch_index_start = single_core_b * block_index + min(block_index, single_core_res_b);
    uint32_t batch_index_end = actual_single_core_b + batch_index_start;

    // ---- [Answer-2] GM Tensor：batch 维作为第一维的 3D ND 布局，无需手工基址偏移 ----
    auto gm_a = make_tensor(make_mem_ptr(a), make_frame_layout<nd_ext_layout_ptn>(B, M, K));
    auto gm_b = make_tensor(make_mem_ptr(b), make_frame_layout<nd_ext_layout_ptn>(B, K, N));
    auto gm_c = make_tensor(make_mem_ptr(c), make_frame_layout<nd_ext_layout_ptn>(B, M, N));
    auto gm_bias = make_tensor(make_mem_ptr(bias), make_frame_layout<nd_ext_layout_ptn>(B, 1, N));

    // L1/L0/BiasTable 缓冲区（脚手架，无需修改）
    __cbuf__ T l1_a_buf[L1_BATCH_SIZE * L1_A_SIZE];
    __cbuf__ T l1_b_buf[L1_BATCH_SIZE * L1_B_SIZE];
    __cbuf__ T l1_bias_buf[L1_BATCH_SIZE * N];
    __ca__ T l0_a_buf[L0_BATCH_SIZE * L0_A_SIZE];
    __cb__ T l0_b_buf[L0_BATCH_SIZE * L0_B_SIZE];
    __cc__ float l0_c_buf[L0_BATCH_SIZE * L0_C_SIZE];
    __biasbuf__ float l0_bias_buf[L0_BATCH_SIZE * N];

    // ---- [Answer-3] L1/L0 缓冲 Tensor：batch 维为第一维的 3D 布局 ----
    // L1A/L0A 用 nz_layout_ptn，L1B/L0B 用 zn_layout_ptn（Cube 分形要求）；
    // Bias 在 L1 与 BiasTable 均为 nd_ext_layout_ptn，BiasTable 位于 __biasbuf__
    auto l1_a_tensor = make_tensor(make_mem_ptr(l1_a_buf), make_frame_layout<nz_layout_ptn, T>(L1_BATCH_SIZE, M, K));
    auto l1_b_tensor = make_tensor(make_mem_ptr(l1_b_buf), make_frame_layout<nz_layout_ptn, T>(L1_BATCH_SIZE, K, N));
    auto l1_bias_tensor = make_tensor(make_mem_ptr(l1_bias_buf), make_frame_layout<nd_ext_layout_ptn, T>(L1_BATCH_SIZE, 1, N));
    auto l0_a_tensor = make_tensor(make_mem_ptr(l0_a_buf), make_frame_layout<nz_layout_ptn, T>(L0_BATCH_SIZE, M, K));
    auto l0_b_tensor = make_tensor(make_mem_ptr(l0_b_buf), make_frame_layout<zn_layout_ptn, T>(L0_BATCH_SIZE, K, N));
    auto l0_c_tensor = make_tensor(make_mem_ptr(l0_c_buf), make_frame_layout<nz_layout_ptn>(L0_BATCH_SIZE, M, N));
    auto l0_bias_tensor = make_tensor(make_mem_ptr(l0_bias_buf), make_frame_layout<nd_ext_layout_ptn>(L0_BATCH_SIZE, 1, N));

    auto copy_gm_to_l1_atom = make_copy(copy_gm_to_l1{}, gm_to_l1_trait_default{});
    auto copy_l1_to_l0a_atom = make_copy(copy_l1_to_l0a{}, l1_to_l0a_trait_default{});
    auto copy_l1_to_l0b_atom = make_copy(copy_l1_to_l0b{}, l1_to_l0b_trait_default{});
    auto copy_l1_to_bt_atom = make_copy(copy_l1_to_biastable{}, l1_to_biastable_trait_default{});
    auto copy_l0c_to_gm_atom = make_copy(copy_l0c_to_gm{}, l0c_to_gm_trait_default{});
    auto mmad_atom = make_mmad(mmad_operation{}, mmad_trait_default{});

    // 事件同步（脚手架，无需修改）：MTE1/MTE2 管 L1 批量搬运，M/MTE1 管 L0 批量加载，FIX/M 管 L0C 写出
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID0);
    asc_sync_notify(PIPE_FIX, PIPE_M, EVENT_ID0);

    // ---- L1 batch 外层循环：每次从 GM 搬入 L1_BATCH_SIZE 个 batch ----
    for (uint32_t l1_batch_index = batch_index_start; l1_batch_index < batch_index_end;
         l1_batch_index += L1_BATCH_SIZE) {
        asc_sync_wait(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        uint32_t l1_batch_size = min(L1_BATCH_SIZE, batch_index_end - l1_batch_index);

        // ---- [Answer-4] GM -> L1：3D slice 一次搬入整批 A/B/Bias ----
        // coord 的第一维是 batch 起始索引，内层 coord(0, 0) 为矩阵内坐标；
        // shape 第一维为本批 batch 数，内层 shape 为单个矩阵的 (行, 列)
        copy(
            copy_gm_to_l1_atom, l1_a_tensor,
            gm_a.slice(make_coord(l1_batch_index, make_coord(0, 0)), make_shape(l1_batch_size, make_shape(M, K))));
        copy(
            copy_gm_to_l1_atom, l1_b_tensor,
            gm_b.slice(make_coord(l1_batch_index, make_coord(0, 0)), make_shape(l1_batch_size, make_shape(K, N))));
        copy(
            copy_gm_to_l1_atom, l1_bias_tensor,
            gm_bias.slice(
                make_coord(l1_batch_index, make_coord(0, 0)), make_shape(l1_batch_size, make_shape(1, N))));

        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // ---- L0 batch 内层循环：每次从 L1 加载 L0_BATCH_SIZE 个 batch 到 L0A/L0B/BiasTable ----
        for (uint32_t l0_batch_index = 0; l0_batch_index < l1_batch_size; l0_batch_index += L0_BATCH_SIZE) {
            asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID0);
            uint32_t l0_batch_size = min(L0_BATCH_SIZE, l1_batch_size - l0_batch_index);

            // ---- [Answer-5] L1 -> L0A/L0B/BiasTable：从 L1 批量张量中切出 L0 批次 ----
            copy(
                copy_l1_to_l0a_atom, l0_a_tensor,
                l1_a_tensor.slice(
                    make_coord(l0_batch_index, make_coord(0, 0)), make_shape(l0_batch_size, make_shape(M, K))));
            copy(
                copy_l1_to_l0b_atom, l0_b_tensor,
                l1_b_tensor.slice(
                    make_coord(l0_batch_index, make_coord(0, 0)), make_shape(l0_batch_size, make_shape(K, N))));
            copy(
                copy_l1_to_bt_atom, l0_bias_tensor,
                l1_bias_tensor.slice(
                    make_coord(l0_batch_index, make_coord(0, 0)), make_shape(l0_batch_size, make_shape(1, N))));

            asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
            asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);
            asc_sync_wait(PIPE_FIX, PIPE_M, EVENT_ID0);

            // ---- [Answer-6] mmad：硬件不支持 batch 维矩阵乘，逐 batch 执行带 Bias 的 5 参数 mmad ----
            // init_with_zero = true：每个 batch 的输出独立计算，无需跨 K 累加
            for (uint32_t l0c_batch_index = 0; l0c_batch_index < l0_batch_size; l0c_batch_index++) {
                mmad(
                    mmad_atom.with(mmad_params{
                        static_cast<uint16_t>(M), static_cast<uint16_t>(N), static_cast<uint16_t>(K),
                        unit_flag_mode::disable, true}),
                    l0_c_tensor.slice(make_coord(l0c_batch_index, make_coord(0, 0)), make_shape(1, make_shape(M, N))),
                    l0_a_tensor.slice(make_coord(l0c_batch_index, make_coord(0, 0)), make_shape(1, make_shape(M, K))),
                    l0_b_tensor.slice(make_coord(l0c_batch_index, make_coord(0, 0)), make_shape(1, make_shape(K, N))),
                    l0_bias_tensor.slice(
                        make_coord(l0c_batch_index, make_coord(0, 0)), make_shape(1, make_shape(1, N))));
            }

            asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
            asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID0);
            asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);

            // ---- [Answer-7] L0C -> GM：一次搬出整批 L0_BATCH_SIZE 个输出矩阵 ----
            // GM 目标位置 = L1 批起点 + L0 批内偏移
            copy(
                copy_l0c_to_gm_atom,
                gm_c.slice(
                    make_coord(l1_batch_index + l0_batch_index, make_coord(0, 0)),
                    make_shape(l0_batch_size, make_shape(M, N))),
                l0_c_tensor);

            asc_sync_notify(PIPE_FIX, PIPE_M, EVENT_ID0);
        }
        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    }
    asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID0);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_FIX, PIPE_M, EVENT_ID0);
    asc_sync_pipe(PIPE_ALL);
}

#endif

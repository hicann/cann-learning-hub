/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_MATMUL_PRACTICE_KERNEL_H
#define BATCH_MATMUL_PRACTICE_KERNEL_H

#include "c_api/asc_simd.h"
#include "tensor_api/tensor.h"

// 注意：本模板包含未补全的 TODO 占位，需补全全部 TODO 后方可编译运行。

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

    uint32_t block_index = block_idx;

    // TODO(1): 计算 batch 维多核切分索引
    // 提示：B 个 batch 均分给 block_num 个核，余数 B % 核数 逐个分给编号靠前的核
    //   single_core_b = B / block_num                                   // 每核基础 batch 数
    //   single_core_res_b = B % block_num                           // 未整除的余数
    //   actual_single_core_b = single_core_b + (余数分给当前核 ? 1 : 0)  // 当前核实际 batch 数
    //   batch_index_start = single_core_b * block_index + min(block_index, single_core_res_b)  // 起始 batch
    //   batch_index_end = actual_single_core_b + batch_index_start       // 结束 batch（不含）
    uint32_t batch_index_start = 0;  // TODO: 替换为你的计算
    uint32_t batch_index_end = 0;    // TODO: 替换为你的计算

    // TODO(2): 构造 4 个 GM Tensor（batch 维作为第一维的 3D ND 布局，无需手工基址偏移）
    // 提示：A 为 (B, M, K)，B 为 (B, K, N)，C 为 (B, M, N)，Bias 为 (B, 1, N)，均使用 nd_ext_layout_ptn
    //   A 已给出作为示例，将 B/C/Bias 从 2D 布局升级为带 batch 维的 3D 布局
    auto gm_a = make_tensor(make_mem_ptr(a), make_frame_layout<nd_ext_layout_ptn>(B, M, K));
    auto gm_b = make_tensor(make_mem_ptr(b), make_frame_layout<nd_ext_layout_ptn>(K, N));    // TODO(2): 改为 3D
    auto gm_c = make_tensor(make_mem_ptr(c), make_frame_layout<nd_ext_layout_ptn>(M, N));    // TODO(2): 改为 3D
    auto gm_bias = make_tensor(make_mem_ptr(bias), make_frame_layout<nd_ext_layout_ptn>(1, N)); // TODO(2): 改为 3D

    // TODO(3): L1/L0 缓冲 Tensor（脚手架，已给全，无需修改）
    // 说明：L1A/L0A 用 nz_layout_ptn、L1B/L0B 用 zn_layout_ptn（Cube 核分形要求）；
    //   L0C 为 float 类型（FP32 累加）；Bias 在 L1 与 BiasTable 均为 nd_ext_layout_ptn；
    //   各张量第一维均为 batch 维（L1_BATCH_SIZE / L0_BATCH_SIZE）
    __cbuf__ T l1_a_buf[L1_BATCH_SIZE * L1_A_SIZE];
    __cbuf__ T l1_b_buf[L1_BATCH_SIZE * L1_B_SIZE];
    __cbuf__ T l1_bias_buf[L1_BATCH_SIZE * N];
    __ca__ T l0_a_buf[L0_BATCH_SIZE * L0_A_SIZE];
    __cb__ T l0_b_buf[L0_BATCH_SIZE * L0_B_SIZE];
    __cc__ float l0_c_buf[L0_BATCH_SIZE * L0_C_SIZE];
    __biasbuf__ float l0_bias_buf[L0_BATCH_SIZE * N];

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

        // TODO(4): 将本批 A/B/Bias 从 GM 搬入 L1（用 3D slice 从 GM 张量切出整批数据）
        // 提示：A 已给出作为示例；coord 第一维为 l1_batch_index，内层为 make_coord(0, 0)；
        //   shape 第一维为 l1_batch_size，内层为单个矩阵的 make_shape(M, K) / (K, N) / (1, N)
        copy(
            copy_gm_to_l1_atom, l1_a_tensor,
            gm_a.slice(make_coord(l1_batch_index, make_coord(0, 0)), make_shape(l1_batch_size, make_shape(M, K))));
        copy(copy_gm_to_l1_atom, l1_b_tensor, gm_b);
        copy(copy_gm_to_l1_atom, l1_bias_tensor, gm_bias);

        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // ---- L0 batch 内层循环：每次从 L1 加载 L0_BATCH_SIZE 个 batch 到 L0A/L0B/BiasTable ----
        for (uint32_t l0_batch_index = 0; l0_batch_index < l1_batch_size; l0_batch_index += L0_BATCH_SIZE) {
            asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID0);
            uint32_t l0_batch_size = min(L0_BATCH_SIZE, l1_batch_size - l0_batch_index);

            // TODO(5): 将本 L0 批次的 A/B/Bias 从 L1 加载到 L0A/L0B/BiasTable
            // 提示：从 L1 批量张量中切出 L0 批次（A 已给出作为示例）；
            //   coord 第一维为 l0_batch_index，shape 第一维为 l0_batch_size；
            //   A/B 走 copy_l1_to_l0a/l0b，Bias 走 copy_l1_to_biastable
            copy(
                copy_l1_to_l0a_atom, l0_a_tensor,
                l1_a_tensor.slice(
                    make_coord(l0_batch_index, make_coord(0, 0)), make_shape(l0_batch_size, make_shape(M, K))));
            copy(copy_l1_to_l0b_atom, l0_b_tensor, l1_b_tensor);
            copy(copy_l1_to_bt_atom, l0_bias_tensor, l1_bias_tensor);

            asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
            asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);
            asc_sync_wait(PIPE_FIX, PIPE_M, EVENT_ID0);

            // TODO(6): 逐 batch 执行带 Bias 的 5 参数 mmad
            // 提示：硬件不支持 batch 维矩阵乘，需在 batch 维循环（l0c_batch_index 从 0 到 l0_batch_size）；
            //   每次从 L0 张量切出第 l0c_batch_index 个 batch（coord 第一维为 l0c_batch_index，shape 第一维为 1）；
            //   init_with_zero 设为 true（每个 batch 输出独立计算，无需跨 K 累加）
            mmad(
                mmad_atom.with(mmad_params{
                    static_cast<uint16_t>(M), static_cast<uint16_t>(N), static_cast<uint16_t>(K),
                    unit_flag_mode::disable, true}),
                l0_c_tensor, l0_a_tensor, l0_b_tensor, l0_bias_tensor);

            asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
            asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID0);
            asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);

            // TODO(7): 将整批 L0_BATCH_SIZE 个 L0C 结果搬出到 GM
            // 提示：GM 目标位置的 batch 起点 = l1_batch_index + l0_batch_index，
            //   shape 第一维为 l0_batch_size，内层为 make_shape(M, N)；源为整个 l0_c_tensor
            copy(copy_l0c_to_gm_atom, gm_c, l0_c_tensor);

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

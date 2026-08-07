#include "c_api/asc_simd.h"

namespace Custom {
union DataUnion {
    constexpr __aicore__ DataUnion() : f(0.0f) {}
    constexpr __aicore__ DataUnion(uint32_t val) : i(val) {}
    float f;
    uint32_t i;
};
constexpr DataUnion fp32_min_value(0x00800000u);
}

constexpr uint32_t LOOP_COUNT = 4096;
constexpr uint32_t WIDTH = 512;
constexpr uint32_t SINGLE_VF_HEIGHT = 16;
constexpr uint32_t SINGLE_VF_DATA_LEN = SINGLE_VF_HEIGHT * WIDTH;
constexpr uint32_t SINGLE_CORE_DATA_LEN = SINGLE_VF_DATA_LEN * LOOP_COUNT;

__simd_vf__ inline void softmax_vf(__ubuf__ float* dst_ub, __ubuf__ float* src_ub, __ubuf__ float* exp_ub)
{
    constexpr uint16_t one_repeat_cnt = asc_get_vf_len() / sizeof(float);
    uint16_t repeat_times = (WIDTH + one_repeat_cnt - 1) / one_repeat_cnt;
    vector_bool mask_full = asc_create_mask_b32(PAT_ALL);

    vector_float src_reg;
    vector_float max_reg;
    vector_float exp_reg;
    vector_float sum_reg;
    vector_float div_reg;

    vector_float src_reg1;
    vector_float max_reg1;
    vector_float exp_reg1;
    vector_float sum_reg1;
    vector_float div_reg1;

    uint16_t halfA = SINGLE_VF_HEIGHT >> 1;

    // 第一部分: ReduceMax -> Sub -> Exp
    for (uint16_t i = 0; i < halfA; i++) {
        asc_duplicate_scalar(max_reg, Custom::fp32_min_value.f, mask_full);
        asc_duplicate_scalar(max_reg1, Custom::fp32_min_value.f, mask_full);
        for (uint16_t j = 0; j < repeat_times; j++) {
            asc_loadalign(src_reg, src_ub + i * WIDTH + j * one_repeat_cnt);
            asc_loadalign(src_reg1, src_ub + i * WIDTH + j * one_repeat_cnt + halfA * WIDTH);
            asc_max(max_reg, max_reg, src_reg, mask_full);
            asc_max(max_reg1, max_reg1, src_reg1, mask_full);
        }
        asc_reduce_max(max_reg, max_reg, mask_full);
        asc_reduce_max(max_reg1, max_reg1, mask_full);
        asc_duplicate(max_reg, max_reg, mask_full);
        asc_duplicate(max_reg1, max_reg1, mask_full);
        for (uint16_t j = 0; j < repeat_times; j++) {
            asc_loadalign(src_reg, src_ub + i * WIDTH + j * one_repeat_cnt);
            asc_loadalign(src_reg1, src_ub + i * WIDTH + j * one_repeat_cnt + halfA * WIDTH);

            asc_exp_sub(exp_reg, src_reg, max_reg, mask_full);
            asc_exp_sub(exp_reg1, src_reg1, max_reg1, mask_full);
            asc_storealign(exp_ub + i * WIDTH + j * one_repeat_cnt, exp_reg, mask_full);
            asc_storealign(exp_ub + i * WIDTH + j * one_repeat_cnt + halfA * WIDTH, exp_reg1, mask_full);
        }
    }

    // 同步：前置步骤中写入exp_ub的操作完成后才能启动后续步骤。
    asc_mem_bar(VST_VLD);

    // 第二部分: ReduceSum -> Div
    for (uint16_t i = 0; i < halfA; i++) {
        asc_duplicate_scalar(sum_reg, 0.0, mask_full);
        asc_duplicate_scalar(sum_reg1, 0.0, mask_full);
        for (uint16_t j = 0; j < repeat_times; j++) {
            asc_loadalign(src_reg, exp_ub + i * WIDTH + j * one_repeat_cnt);
            asc_loadalign(src_reg1, exp_ub + i * WIDTH + j * one_repeat_cnt + halfA * WIDTH);
            asc_add(sum_reg, sum_reg, src_reg, mask_full);
            asc_add(sum_reg1, sum_reg1, src_reg1, mask_full);
        }
        asc_reduce_sum(sum_reg, sum_reg, mask_full);
        asc_reduce_sum(sum_reg1, sum_reg1, mask_full);
        asc_duplicate(sum_reg, sum_reg, mask_full);
        asc_duplicate(sum_reg1, sum_reg1, mask_full);
        for (uint16_t j = 0; j < repeat_times; j++) {
            asc_loadalign(max_reg, exp_ub + i * WIDTH + j * one_repeat_cnt);
            asc_loadalign(max_reg1, exp_ub + i * WIDTH + j * one_repeat_cnt + halfA * WIDTH);
            asc_div(div_reg, max_reg, sum_reg, mask_full);
            asc_div(div_reg1, max_reg1, sum_reg1, mask_full);
            asc_storealign(dst_ub + i * WIDTH + j * one_repeat_cnt, div_reg, mask_full);
            asc_storealign(dst_ub + i * WIDTH + j * one_repeat_cnt + halfA * WIDTH, div_reg1, mask_full);
        }
    }
}

// ======================= 核函数 V1 =========================
__global__ __vector__ void softmax_custom(__gm__ uint8_t* x, __gm__ uint8_t* y)
{
    asc_init();

    // 申请UB内存
    __ubuf__ float src0_ub[SINGLE_VF_DATA_LEN];
    __ubuf__ float dst0_ub[SINGLE_VF_DATA_LEN];
    __ubuf__ float exp0_ub[SINGLE_VF_DATA_LEN];

    __ubuf__ float src1_ub[SINGLE_VF_DATA_LEN];
    __ubuf__ float dst1_ub[SINGLE_VF_DATA_LEN];
    __ubuf__ float exp1_ub[SINGLE_VF_DATA_LEN];

    uint8_t mutex_id_src0 = 1;   // src0_ub对应的资源锁
    uint8_t mutex_id_dst0 = 2;   // dst0_ub对应的资源锁

    uint8_t mutex_id_src1 = 3;   // src1_ub对应的资源锁
    uint8_t mutex_id_dst1 = 4;   // dst1_ub对应的资源锁

    __gm__ float* x_gm = reinterpret_cast<__gm__ float*>(x) + block_idx * SINGLE_CORE_DATA_LEN;
    __gm__ float* y_gm = reinterpret_cast<__gm__ float*>(y) + block_idx * SINGLE_CORE_DATA_LEN;

    for (uint32_t i = 0; i < LOOP_COUNT; i++) {
        __ubuf__ float* src_ub;
        __ubuf__ float* dst_ub;
        __ubuf__ float* exp_ub;
        uint8_t mutex_id_src;   // src_ub对应的资源锁
        uint8_t mutex_id_dst;   // dst_ub对应的资源锁

        if(i % 2 == 0) {
            src_ub = src0_ub;
            dst_ub = dst0_ub;
            exp_ub = exp0_ub;
            mutex_id_src = mutex_id_src0;
            mutex_id_dst = mutex_id_dst0;
        } else {
            src_ub = src1_ub;
            dst_ub = dst1_ub;
            exp_ub = exp1_ub;
            mutex_id_src = mutex_id_src1;
            mutex_id_dst = mutex_id_dst1;
        }

        // 搬入操作需要独占src_ub
        asc_lock(PIPE_MTE2, mutex_id_src);
        asc_copy_gm2ub_align(src_ub, &x_gm[i * SINGLE_VF_DATA_LEN], 1, SINGLE_VF_DATA_LEN * sizeof(float),
            0, 0, true, 4, 0, 0);
        asc_unlock(PIPE_MTE2, mutex_id_src);

        // 计算操作需要独占src_ub和dst_ub
        asc_lock(PIPE_V, mutex_id_src);
        asc_lock(PIPE_V, mutex_id_dst);
        softmax_vf(dst_ub, src_ub, exp_ub);
        asc_unlock(PIPE_V, mutex_id_src);
        asc_unlock(PIPE_V, mutex_id_dst);

        // 搬出操作需要独占dst_ub
        asc_lock(PIPE_MTE3, mutex_id_dst);
        asc_copy_ub2gm_align(&y_gm[i * SINGLE_VF_DATA_LEN], dst_ub, 1, SINGLE_VF_DATA_LEN * sizeof(float),
            4, 0, 0);
        asc_unlock(PIPE_MTE3, mutex_id_dst);
    }
}

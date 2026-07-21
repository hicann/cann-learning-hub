#include "c_api/asc_simd.h"

constexpr uint32_t LOOP_COUNT = 512;
constexpr uint32_t SINGLE_VF_DATA_LEN = 12288;

__simd_vf__ inline void add_vf(__ubuf__ float* dst_ub, __ubuf__ float* src0_ub, __ubuf__ float* src1_ub)
{
    constexpr uint16_t one_repeat_cnt = asc_get_vf_len() / sizeof(float);
    uint16_t repeat_times = (SINGLE_VF_DATA_LEN + one_repeat_cnt - 1) / one_repeat_cnt;
    vector_bool mask_full = asc_create_mask_b32(PAT_ALL);

    vector_float src0_reg;
    vector_float src1_reg;
    vector_float dst_reg;

    for (uint16_t i = 0; i < repeat_times; i++) {
        asc_loadalign(src0_reg, src0_ub + i * one_repeat_cnt);
        asc_loadalign(src1_reg, src1_ub + i * one_repeat_cnt);
        asc_add(dst_reg, src0_reg, src1_reg, mask_full);
        asc_storealign(dst_ub + i * one_repeat_cnt, dst_reg, mask_full);
    }
}

// ======================= 核函数 =========================
__global__ __vector__ void add_custom(__gm__ uint8_t* src0, __gm__ uint8_t* src1, __gm__ uint8_t* dst)
{
    asc_init();

    // 申请UB内存
    __ubuf__ float src0_ub[SINGLE_VF_DATA_LEN];
    __ubuf__ float src1_ub[SINGLE_VF_DATA_LEN];
    __ubuf__ float dst_ub[SINGLE_VF_DATA_LEN];

    __gm__ float* src0_gm = reinterpret_cast<__gm__ float*>(src0);
    __gm__ float* src1_gm = reinterpret_cast<__gm__ float*>(src1);
    __gm__ float* dst_gm = reinterpret_cast<__gm__ float*>(dst);

    uint8_t mutex_id_src = 1;
    uint8_t mutex_id_dst = 2;

    for (uint32_t i = 0; i < LOOP_COUNT; i++) {
        // 上一个循环的数据搬出操作完成后才能启动新的数据搬入操作
        asc_lock(PIPE_MTE2, mutex_id_src);
        asc_set_copy_pad_val(float(0.0));
        asc_copy_gm2ub_align(src0_ub, &src0_gm[i * SINGLE_VF_DATA_LEN], 1, static_cast<uint16_t>(SINGLE_VF_DATA_LEN * sizeof(float)), 0, 0, false, 0, 0, 0);
        asc_copy_gm2ub_align(src1_ub, &src1_gm[i * SINGLE_VF_DATA_LEN], 1, static_cast<uint16_t>(SINGLE_VF_DATA_LEN * sizeof(float)), 0, 0, false, 0, 0, 0);
        asc_unlock(PIPE_MTE2, mutex_id_src);

        // 数据搬入操作完成后才能启动计算
        asc_lock(PIPE_V, mutex_id_src);
        asc_lock(PIPE_V, mutex_id_dst);
        // 调用VF函数执行计算
        add_vf(dst_ub, src0_ub, src1_ub);
        asc_unlock(PIPE_V, mutex_id_src);
        asc_unlock(PIPE_V, mutex_id_dst);

        // 计算完成后才能启动数据搬出
        asc_lock(PIPE_MTE3, mutex_id_dst);
        asc_copy_ub2gm_align(&dst_gm[i * SINGLE_VF_DATA_LEN], dst_ub, 1, static_cast<uint16_t>(SINGLE_VF_DATA_LEN * sizeof(float)), 0, 0, 0);
        asc_unlock(PIPE_MTE3, mutex_id_dst);
    }
}

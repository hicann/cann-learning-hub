#include "c_api/asc_simd.h"

constexpr float COEFF_LINEAR = -1.595769f;
constexpr float COEFF_CUBIC = -0.071405f;

constexpr uint32_t LOOP_NUM = 1024;
constexpr uint32_t SINGLE_VF_LENGTH = 12288;
constexpr uint32_t SINGLE_CORE_LENGTH = LOOP_NUM * SINGLE_VF_LENGTH;

__aicore__ __inline__ constexpr uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__simd_vf__ inline static void gelu_vf(__ubuf__ float* x_ub, __ubuf__ float* y_ub)
{
    // single_repeat_length 表示一次向量重复能处理多少个 float 元素。
    constexpr uint32_t single_repeat_length = asc_get_vf_len() / sizeof(float);
    uint32_t repeat_num = div_ceil(SINGLE_VF_LENGTH, single_repeat_length);

    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    vector_float x0_reg;
    vector_float y0_reg;
    vector_float tmp0_reg;

    vector_float x1_reg;
    vector_float y1_reg;
    vector_float tmp1_reg;

    for (uint32_t i = 0; i < repeat_num / 2; ++i) {
        // 从 UB 中取一段数据到寄存器。
        asc_loadalign(x0_reg, x_ub + i * single_repeat_length);
        asc_loadalign(x1_reg, x_ub + (i + repeat_num / 2) * single_repeat_length);

        asc_mul(y0_reg, x0_reg, x0_reg, mask);                      // x^2
        asc_mul(y0_reg, y0_reg, x0_reg, mask);                      // x^3
        asc_mul_scalar(y0_reg, y0_reg, COEFF_CUBIC, mask);          // -0.071405 * x^3
        asc_mul_scalar(tmp0_reg, x0_reg, COEFF_LINEAR, mask);       // -1.595769 * x
        asc_add(y0_reg, tmp0_reg, y0_reg, mask);                    // -1.595769 * x - 0.071405 * x^3
        asc_exp(y0_reg, y0_reg, mask);                              // exp(...)
        asc_add_scalar(y0_reg, y0_reg, 1.0f, mask);                 // 1 + exp(...)
        asc_div(y0_reg, x0_reg, y0_reg, mask);                      // x / (1 + exp(...))

        asc_mul(y1_reg, x1_reg, x1_reg, mask);                      // x^2
        asc_mul(y1_reg, y1_reg, x1_reg, mask);                      // x^3
        asc_mul_scalar(y1_reg, y1_reg, COEFF_CUBIC, mask);          // -0.071405 * x^3
        asc_mul_scalar(tmp1_reg, x1_reg, COEFF_LINEAR, mask);       // -1.595769 * x
        asc_add(y1_reg, tmp1_reg, y1_reg, mask);                    // -1.595769 * x - 0.071405 * x^3
        asc_exp(y1_reg, y1_reg, mask);                              // exp(...)
        asc_add_scalar(y1_reg, y1_reg, 1.0f, mask);                 // 1 + exp(...)
        asc_div(y1_reg, x1_reg, y1_reg, mask);                      // x / (1 + exp(...))

        asc_storealign(y_ub + i * single_repeat_length, y0_reg, mask);
        asc_storealign(y_ub + (i + repeat_num / 2) * single_repeat_length, y1_reg, mask);
    }
}

__global__ __vector__ void gelu_custom(__gm__ uint8_t* x, __gm__ uint8_t* y)
{
    asc_init();

    // 每个核只负责自己对应的连续数据段，起始位置由 block_idx 决定。
    __gm__ float* x_gm = reinterpret_cast<__gm__ float*>(x) + block_idx * SINGLE_CORE_LENGTH;
    __gm__ float* y_gm = reinterpret_cast<__gm__ float*>(y) + block_idx * SINGLE_CORE_LENGTH;

    // 申请两份完整的UB内存
    __ubuf__ float x0_ub[SINGLE_VF_LENGTH];
    __ubuf__ float y0_ub[SINGLE_VF_LENGTH];
    __ubuf__ float x1_ub[SINGLE_VF_LENGTH];
    __ubuf__ float y1_ub[SINGLE_VF_LENGTH];

    constexpr uint8_t x0_mutex_id = 1;
    constexpr uint8_t x1_mutex_id = 2;
    constexpr uint8_t y0_mutex_id = 3;
    constexpr uint8_t y1_mutex_id = 4;

    constexpr uint32_t copy_bytes = SINGLE_VF_LENGTH * sizeof(float);

    for (uint32_t i = 0; i < LOOP_NUM; i++) {
        __ubuf__ float* x_ub;
        __ubuf__ float* y_ub;

        uint8_t x_mutex_id;
        uint8_t y_mutex_id;

        if ((i & 1U) == 0U) {
            x_ub = x0_ub;
            y_ub = y0_ub;
            x_mutex_id = x0_mutex_id;
            y_mutex_id = y0_mutex_id;
        } else {
            x_ub = x1_ub;
            y_ub = y1_ub;
            x_mutex_id = x1_mutex_id;
            y_mutex_id = y1_mutex_id;
        }

        asc_lock(PIPE_MTE2, x_mutex_id);
        asc_copy_gm2ub_align(x_ub, x_gm + i * SINGLE_VF_LENGTH, 1, copy_bytes, 0, 0, false, 4, 0, 0);
        asc_unlock(PIPE_MTE2, x_mutex_id);

        // 在寄存器中完成 GELU 近似计算。
        asc_lock(PIPE_V, x_mutex_id);
        asc_lock(PIPE_V, y_mutex_id);
        gelu_vf(x_ub, y_ub);
        asc_unlock(PIPE_V, x_mutex_id);
        asc_unlock(PIPE_V, y_mutex_id);

        asc_lock(PIPE_MTE3, y_mutex_id);
        asc_copy_ub2gm_align(y_gm + i * SINGLE_VF_LENGTH, y_ub, 1, copy_bytes, 4, 0, 0);
        asc_unlock(PIPE_MTE3, y_mutex_id);
    }
}

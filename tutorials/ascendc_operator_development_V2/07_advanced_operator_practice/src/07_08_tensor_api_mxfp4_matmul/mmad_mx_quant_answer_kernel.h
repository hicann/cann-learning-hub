#ifndef MMAD_MX_QUANT_ANSWER_KERNEL_H
#define MMAD_MX_QUANT_ANSWER_KERNEL_H

#include "c_api/asc_simd.h"
#include "tensor_api/tensor.h"
#include "utils/std/cmath.h"

__aicore__ __inline__ constexpr uint32_t align_even(uint32_t a) { return (a + 1) / 2 * 2; }

constexpr uint32_t SCALE_CEIL_NUMBER = 32;
constexpr uint32_t SCALE_ALIGN_NUMBER = 2;
constexpr uint32_t C0_ELEMENT_SCALE = 2;
constexpr uint32_t C0_ELEMENT_L0C = 16;
constexpr uint32_t C0_ELEMENT_B4 = 64;
constexpr uint32_t CUBE_BLOCK = 16;

template <
    uint32_t M_, uint32_t N_, uint32_t K_, uint32_t single_core_m_, uint32_t single_core_n_, uint32_t single_core_k_,
    uint32_t base_m_, uint32_t base_n_, uint32_t base_k_, uint32_t step_k_, uint32_t scale_factor_k_>
struct kernel_trait {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    static constexpr uint32_t single_core_m = single_core_m_;
    static constexpr uint32_t single_core_n = single_core_n_;
    static constexpr uint32_t single_core_k = single_core_k_;

    static constexpr uint32_t base_m = base_m_;
    static constexpr uint32_t base_n = base_n_;
    static constexpr uint32_t base_k = base_k_;

    static constexpr uint32_t step_k = step_k_;
    static constexpr uint32_t scale_factor_k = scale_factor_k_;
};

constexpr asc::te::mmad_trait MX_MMAD_TRAIT =
    asc::te::mmad_trait{0, false, false, true, asc::te::mmad_type::mx};
struct mmad_trait_mx {
    using trait_type = asc::te::mmad_trait;
    static constexpr const trait_type value = MX_MMAD_TRAIT;
};

template <typename Trait>
class kernel_mmadmx_quant {
public:
    __aicore__ inline kernel_mmadmx_quant() {}

    // [Answer-1] process 签名添加量化参数 quant_scale 和 quant_offset，输出类型改为 int8_t
    __aicore__ inline void process(
        __gm__ fp4x2_e1m2_t* a, __gm__ fp4x2_e1m2_t* b, __gm__ fp8_e8m0_t* as, __gm__ fp8_e8m0_t* bs,
        __gm__ int8_t* c, __gm__ uint64_t* quant_scale, __gm__ uint64_t* quant_offset)
    {
        init_compute_params();

        // Init GM tensor
        auto gm_tensor_a = asc::te::make_tensor(
            asc::te::make_mem_ptr(a), asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(Trait::M, Trait::K));
        auto gm_tensor_b = asc::te::make_tensor(
            asc::te::make_mem_ptr(b), asc::te::make_frame_layout<asc::te::dn_ext_layout_ptn>(Trait::K, Trait::N));

        auto gm_tensor_as = asc::te::make_tensor(
            asc::te::make_mem_ptr(as),
            asc::te::make_frame_layout<asc::te::scalea_nd_layout_ptn>(Trait::M, scale_k));
        auto gm_tensor_bs = asc::te::make_tensor(
            asc::te::make_mem_ptr(bs),
            asc::te::make_frame_layout<asc::te::scaleb_dn_layout_ptn>(scale_k, Trait::N));
        auto gm_tensor_c = asc::te::make_tensor(
            asc::te::make_mem_ptr(c), asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(Trait::M, Trait::N));

        // Init GM quant tensors (scale and offset, shape [1, N])
        auto gm_quant_scale = asc::te::make_tensor(
            asc::te::make_mem_ptr(quant_scale),
            asc::te::make_frame_layout<asc::te::nd_layout_ptn>(1, Trait::N));
        auto gm_quant_offset = asc::te::make_tensor(
            asc::te::make_mem_ptr(quant_offset),
            asc::te::make_frame_layout<asc::te::nd_layout_ptn>(1, Trait::N));

        // Slice single core tensor
        auto gm_single_tensor_a = gm_tensor_a.slice(
            asc::te::make_coord(m_iter_idx * Trait::single_core_m, 0),
            asc::te::make_shape(actual_single_core_m, Trait::single_core_k));
        auto gm_single_tensor_b = gm_tensor_b.slice(
            asc::te::make_coord(0, n_iter_idx * Trait::single_core_n),
            asc::te::make_shape(Trait::single_core_k, actual_single_core_n));
        auto gm_single_tensor_as = gm_tensor_as.slice(
            asc::te::make_coord(m_iter_idx * Trait::single_core_m, 0),
            asc::te::make_shape(actual_single_core_m, scale_k));
        auto gm_single_tensor_bs = gm_tensor_bs.slice(
            asc::te::make_coord(0, n_iter_idx * Trait::single_core_n),
            asc::te::make_shape(scale_k, actual_single_core_n));
        auto gm_single_tensor_c = gm_tensor_c.slice(
            asc::te::make_coord(m_iter_idx * Trait::single_core_m, n_iter_idx * Trait::single_core_n),
            asc::te::make_shape(actual_single_core_m, actual_single_core_n));

        process_loop(gm_single_tensor_a, gm_single_tensor_b, gm_single_tensor_as, gm_single_tensor_bs, gm_single_tensor_c,
                    gm_quant_scale, gm_quant_offset);
    }

private:
    static constexpr uint32_t L1_DATA_INDEX = 0;   // A/B L1 ping/pong slots
    static constexpr uint32_t L1_SCALE_INDEX = 2;  // As/Bs L1 ping/pong slots
    static constexpr uint32_t L0_INDEX = 4;       // L0 A/B ping/pong slots
    static constexpr uint32_t L0C_INDEX = 6;      // L0C single buffer slot
    static constexpr uint32_t QUANT_INDEX = 7;    // Quant L1 buffer slot

    __aicore__ inline uint32_t get_data_slot(uint32_t ping_pong_idx) const { return L1_DATA_INDEX + ping_pong_idx; }

    __aicore__ inline uint32_t get_scale_slot(uint32_t ping_pong_idx) const { return L1_SCALE_INDEX + ping_pong_idx; }

    __aicore__ inline uint32_t get_l0_slot(uint32_t ping_pong_idx) const { return L0_INDEX + ping_pong_idx; }

    // Main loop of matrix multiplication, including L1 prefetching, L1 to L0 copy, Mmad compute, and L0 to GM copy
    template <typename TensorA, typename TensorB, typename TensorAs, typename TensorBs, typename TensorC,
              typename TensorQuantScale, typename TensorQuantOffset>
    __aicore__ inline void process_loop(
        const TensorA& a, const TensorB& b, const TensorAs& as, const TensorBs& bs, TensorC& c,
        const TensorQuantScale& quant_scale, const TensorQuantOffset& quant_offset)
    {
        auto l1_layout_a = asc::te::make_frame_layout<asc::te::nz_layout_ptn, C0_ELEMENT_B4>(
            Trait::base_m, Trait::base_k * Trait::step_k);
        auto l1_layout_b = asc::te::make_frame_layout<asc::te::zn_layout_ptn, C0_ELEMENT_B4>(
            Trait::base_k * Trait::step_k, Trait::base_n);
        auto l1_layout_as = asc::te::make_frame_layout<asc::te::zz_layout_ptn, C0_ELEMENT_SCALE>(
            Trait::base_m, base_scale_k * Trait::step_k * Trait::scale_factor_k);
        auto l1_layout_bs = asc::te::make_frame_layout<asc::te::nn_layout_ptn, C0_ELEMENT_SCALE>(
            base_scale_k * Trait::step_k * Trait::scale_factor_k, Trait::base_n);

        // Malloc L1 and L0 buffers
        __cbuf__ fp4x2_e1m2_t l1_buf_a_ping[l1_size_a / 2];
        __cbuf__ fp4x2_e1m2_t l1_buf_a_pong[l1_size_a / 2];
        __cbuf__ fp4x2_e1m2_t l1_buf_b_ping[l1_size_b / 2];
        __cbuf__ fp4x2_e1m2_t l1_buf_b_pong[l1_size_b / 2];
        __cbuf__ fp8_e8m0_t l1_buf_as_ping[l1_size_as];
        __cbuf__ fp8_e8m0_t l1_buf_as_pong[l1_size_as];
        __cbuf__ fp8_e8m0_t l1_buf_bs_ping[l1_size_bs];
        __cbuf__ fp8_e8m0_t l1_buf_bs_pong[l1_size_bs];
        __ca__ fp4x2_e1m2_t l0_buf_a_ping[l0_size_a / 2];
        __ca__ fp4x2_e1m2_t l0_buf_a_pong[l0_size_a / 2];
        __cb__ fp4x2_e1m2_t l0_buf_b_ping[l0_size_b / 2];
        __cb__ fp4x2_e1m2_t l0_buf_b_pong[l0_size_b / 2];
        __cc__ float l0_buf_c[l0_size_c];

        // [Answer-2] 量化参数的 L1 缓冲区
        __cbuf__ uint64_t l1_buf_quant_scale[Trait::base_n];
        __cbuf__ uint64_t l1_buf_quant_offset[Trait::base_n];

        auto l0_ptr_a_ping = asc::te::make_mem_ptr(l0_buf_a_ping);
        auto l0_ptr_a_pong = asc::te::make_mem_ptr(l0_buf_a_pong);
        auto l0_ptr_b_ping = asc::te::make_mem_ptr(l0_buf_b_ping);
        auto l0_ptr_b_pong = asc::te::make_mem_ptr(l0_buf_b_pong);
        auto l0_ptr_as_ping = asc::te::make_mem_ptr<asc::te::location::l0scalea, fp8_e8m0_t>(
            reinterpret_cast<uint64_t>(l0_buf_a_ping) / 16);
        auto l0_ptr_as_pong = asc::te::make_mem_ptr<asc::te::location::l0scalea, fp8_e8m0_t>(
            reinterpret_cast<uint64_t>(l0_buf_a_pong) / 16);
        auto l0_ptr_bs_ping = asc::te::make_mem_ptr<asc::te::location::l0scaleb, fp8_e8m0_t>(
            reinterpret_cast<uint64_t>(l0_buf_b_ping) / 16);
        auto l0_ptr_bs_pong = asc::te::make_mem_ptr<asc::te::location::l0scaleb, fp8_e8m0_t>(
            reinterpret_cast<uint64_t>(l0_buf_b_pong) / 16);

        auto l1_tensor_a_ping = asc::te::make_tensor(asc::te::make_mem_ptr(l1_buf_a_ping), l1_layout_a);
        auto l1_tensor_a_pong = asc::te::make_tensor(asc::te::make_mem_ptr(l1_buf_a_pong), l1_layout_a);
        auto l1_tensor_b_ping = asc::te::make_tensor(asc::te::make_mem_ptr(l1_buf_b_ping), l1_layout_b);
        auto l1_tensor_b_pong = asc::te::make_tensor(asc::te::make_mem_ptr(l1_buf_b_pong), l1_layout_b);
        auto l1_tensor_as_ping = asc::te::make_tensor(asc::te::make_mem_ptr(l1_buf_as_ping), l1_layout_as);
        auto l1_tensor_as_pong = asc::te::make_tensor(asc::te::make_mem_ptr(l1_buf_as_pong), l1_layout_as);
        auto l1_tensor_bs_ping = asc::te::make_tensor(asc::te::make_mem_ptr(l1_buf_bs_ping), l1_layout_bs);
        auto l1_tensor_bs_pong = asc::te::make_tensor(asc::te::make_mem_ptr(l1_buf_bs_pong), l1_layout_bs);

        // L1 quant tensors
        auto l1_quant_scale = asc::te::make_tensor(
            asc::te::make_mem_ptr(l1_buf_quant_scale),
            asc::te::make_frame_layout<asc::te::nd_layout_ptn, uint64_t>(1, Trait::base_n));
        auto l1_quant_offset = asc::te::make_tensor(
            asc::te::make_mem_ptr(l1_buf_quant_offset),
            asc::te::make_frame_layout<asc::te::nd_layout_ptn, uint64_t>(1, Trait::base_n));
        (void)l1_quant_offset; // 当前 4 参数 Copy 仅使用 scale，offset 预留待硬件 API 扩展

        // ============================================================
        // Mutex slot-lock synchronization: no preset needed (first asc_lock acquires a free slot).
        // ============================================================

        // ============================================================
        // N/M outer loops + K main loop
        // ============================================================
        for (uint32_t n_block_idx = 0; n_block_idx < n_loop_count; n_block_idx++) {
            uint16_t cur_n = (n_block_idx + 1 == n_loop_count) ? tail_n : Trait::base_n;
            for (uint32_t m_block_idx = 0; m_block_idx < m_loop_count; m_block_idx++) {
                uint16_t cur_m = (m_block_idx + 1 == m_loop_count) ? tail_m : Trait::base_m;
                // Reset the data_copy_in progress for each (m_block_idx, n_block_idx) tile.
                uint32_t data_next_k_chunk_idx = 0;  // K-direction chunk index for the next A/B data_copy_in.
                uint32_t scale_next_k_chunk_idx = 0; // K-direction chunk index for the next As/Bs data_copy_in.
                uint8_t data_copy_in_idx = 0;       // L1 buffer index written by the next A/B data_copy_in (0=Ping, 1=Pong).
                uint8_t scale_copy_in_idx = 0; // L1 buffer index written by the next As/Bs data_copy_in (0=Ping, 1=Pong).

                // Acquire L0C for Cube write (waits for Fixpipe to release it from the previous tile).
                asc_lock(PIPE_M, L0C_INDEX);

                // ---- Copy in the first chunk of A/B and As/Bs ----
                uint32_t data_write_slot = get_data_slot(data_copy_in_idx);
                asc_lock(PIPE_MTE2, data_write_slot); // A/B Ping writable.

                constexpr uint32_t step_cur_k = data_chunk_step * Trait::base_k;
                // A: GM -> L1
                asc::te::copy(
                    gm_to_l1_atom, l1_tensor_a_ping,
                    a.slice(
                        asc::te::make_coord(m_block_idx * Trait::base_m, data_next_k_chunk_idx * Trait::base_k),
                        asc::te::make_shape(cur_m, step_cur_k)));
                // B: GM -> L1
                asc::te::copy(
                    gm_to_l1_atom, l1_tensor_b_ping,
                    b.slice(
                        asc::te::make_coord(data_next_k_chunk_idx * Trait::base_k, n_block_idx * Trait::base_n),
                        asc::te::make_shape(step_cur_k, cur_n)));

                asc_unlock(PIPE_MTE2, data_write_slot); // A/B Ping written.
                data_next_k_chunk_idx += data_chunk_step;
                data_copy_in_idx ^= 1; // Write to Pong next time.

                uint32_t scale_write_slot = get_scale_slot(scale_copy_in_idx);
                asc_lock(PIPE_MTE2, scale_write_slot); // As/Bs Ping writable.

                constexpr uint32_t step_cur_scale_k =
                    align_even(AscendC::Std::ceil_div(scale_chunk_step * Trait::base_k, SCALE_CEIL_NUMBER));
                // scale_a: GM -> L1
                asc::te::copy(
                    gm_to_l1_atom, l1_tensor_as_ping,
                    as.slice(
                        asc::te::make_coord(m_block_idx * Trait::base_m, scale_next_k_chunk_idx * base_scale_k),
                        asc::te::make_shape(cur_m, step_cur_scale_k)));
                // scale_b: GM -> L1
                asc::te::copy(
                    gm_to_l1_atom, l1_tensor_bs_ping,
                    bs.slice(
                        asc::te::make_coord(scale_next_k_chunk_idx * base_scale_k, n_block_idx * Trait::base_n),
                        asc::te::make_shape(step_cur_scale_k, cur_n)));

                asc_unlock(PIPE_MTE2, scale_write_slot); // As/Bs Ping written.
                scale_next_k_chunk_idx += scale_chunk_step;
                scale_copy_in_idx ^= 1;

                auto l0_layout_c = asc::te::make_frame_layout<asc::te::nz_layout_ptn, C0_ELEMENT_L0C>(cur_m, cur_n);
                auto l0_tensor_c = asc::te::make_tensor(asc::te::make_mem_ptr(l0_buf_c), l0_layout_c);

                // ---- K-direction main loop ----
                for (uint32_t k_block_idx = 0; k_block_idx < k_loop_count; k_block_idx++) {
                    constexpr uint16_t cur_k = Trait::base_k;
                    // Determine the L1 read buffer (Ping/Pong) for the current k_block_idx.
                    uint32_t data_read_idx = (k_block_idx / data_chunk_step) % 2;
                    uint32_t scale_read_idx = (k_block_idx / scale_chunk_step) % 2;
                    uint32_t k_offset_in_data_chunk = k_block_idx % data_chunk_step;
                    uint32_t k_offset_in_scale_chunk = k_block_idx % scale_chunk_step;

                    const auto& l1_read_buf_a = (data_read_idx == 0) ? l1_tensor_a_ping : l1_tensor_a_pong;
                    const auto& l1_read_buf_b = (data_read_idx == 0) ? l1_tensor_b_ping : l1_tensor_b_pong;
                    const auto& l1_read_buf_as = (scale_read_idx == 0) ? l1_tensor_as_ping : l1_tensor_as_pong;
                    const auto& l1_read_buf_bs = (scale_read_idx == 0) ? l1_tensor_bs_ping : l1_tensor_bs_pong;

                    auto l0_layout_a = asc::te::make_frame_layout<asc::te::nz_layout_ptn, C0_ELEMENT_B4>(cur_m, cur_k);
                    auto l0_layout_b = asc::te::make_frame_layout<asc::te::zn_layout_ptn, C0_ELEMENT_B4>(cur_k, cur_n);
                    auto l0_tensor_a_ping = asc::te::make_tensor(l0_ptr_a_ping, l0_layout_a);
                    auto l0_tensor_a_pong = asc::te::make_tensor(l0_ptr_a_pong, l0_layout_a);
                    auto l0_tensor_b_ping = asc::te::make_tensor(l0_ptr_b_ping, l0_layout_b);
                    auto l0_tensor_b_pong = asc::te::make_tensor(l0_ptr_b_pong, l0_layout_b);

                    constexpr uint32_t cur_scale_k = align_even(AscendC::Std::ceil_div(cur_k, SCALE_CEIL_NUMBER));
                    auto l0_layout_as =
                        asc::te::make_frame_layout<asc::te::zz_layout_ptn, C0_ELEMENT_SCALE>(cur_m, cur_scale_k);
                    auto l0_layout_bs =
                        asc::te::make_frame_layout<asc::te::nn_layout_ptn, C0_ELEMENT_SCALE>(cur_scale_k, cur_n);
                    auto l0_tensor_as_ping = asc::te::make_tensor(l0_ptr_as_ping, l0_layout_as);
                    auto l0_tensor_as_pong = asc::te::make_tensor(l0_ptr_as_pong, l0_layout_as);
                    auto l0_tensor_bs_ping = asc::te::make_tensor(l0_ptr_bs_ping, l0_layout_bs);
                    auto l0_tensor_bs_pong = asc::te::make_tensor(l0_ptr_bs_pong, l0_layout_bs);

                    // Select the L0 double buffer.
                    const auto& l0_tensor_a = (mte1_db_flag == 0) ? l0_tensor_a_ping : l0_tensor_a_pong;
                    const auto& l0_tensor_b = (mte1_db_flag == 0) ? l0_tensor_b_ping : l0_tensor_b_pong;
                    const auto& l0_tensor_as = (mte1_db_flag == 0) ? l0_tensor_as_ping : l0_tensor_as_pong;
                    const auto& l0_tensor_bs = (mte1_db_flag == 0) ? l0_tensor_bs_ping : l0_tensor_bs_pong;

                    // ---- Reverse synchronization: wait for the previous Compute to release the L0 buffer ----
                    uint32_t l0_slot = get_l0_slot(mte1_db_flag);
                    asc_lock(PIPE_MTE1, l0_slot);

                    // ---- Forward synchronization ----
                    // Acquire the L1 chunk for reading at chunk start (waits for data_copy_in to finish writing it).
                    if (k_offset_in_data_chunk == 0) {
                        asc_lock(PIPE_MTE1, get_data_slot(data_read_idx));
                    }
                    if (k_offset_in_scale_chunk == 0) {
                        asc_lock(PIPE_MTE1, get_scale_slot(scale_read_idx));
                    }

                    // ---- data_load: L1 -> L0 ----
                    // A:L1 -> L0A
                    asc::te::copy(
                        l1_to_l0a_atom, l0_tensor_a,
                        l1_read_buf_a.slice(
                            asc::te::make_coord(0, k_offset_in_data_chunk * Trait::base_k),
                            asc::te::make_shape(cur_m, cur_k)));
                    // B:L1 -> L0B
                    asc::te::copy(
                        l1_to_l0b_atom, l0_tensor_b,
                        l1_read_buf_b.slice(
                            asc::te::make_coord(k_offset_in_data_chunk * Trait::base_k, 0),
                            asc::te::make_shape(cur_k, cur_n)));

                    // scale_a: L1 -> L0AScale
                    asc::te::copy(
                        l1_to_l0scalea_atom, l0_tensor_as,
                        l1_read_buf_as.slice(
                            asc::te::make_coord(0, k_offset_in_scale_chunk * base_scale_k),
                            asc::te::make_shape(cur_m, cur_scale_k)));
                    // scale_b: L1 -> L0BScale
                    asc::te::copy(
                        l1_to_l0scaleb_atom, l0_tensor_bs,
                        l1_read_buf_bs.slice(
                            asc::te::make_coord(k_offset_in_scale_chunk * base_scale_k, 0),
                            asc::te::make_shape(cur_scale_k, cur_n)));

                    // ---- Reverse synchronization ----
                    // The current L1 chunk has been consumed; release it so data_copy_in can overwrite.
                    if (((k_offset_in_data_chunk + 1) == data_chunk_step) || (k_block_idx + 1 == k_loop_count)) {
                        asc_unlock(PIPE_MTE1, get_data_slot(data_read_idx));
                    }
                    if (((k_offset_in_scale_chunk + 1) == scale_chunk_step) || (k_block_idx + 1 == k_loop_count)) {
                        asc_unlock(PIPE_MTE1, get_scale_slot(scale_read_idx));
                    }

                    // L0 written, ready for Cube.
                    asc_unlock(PIPE_MTE1, l0_slot);

                    // ---- Compute: Mmad matrix multiply-accumulate ----
                    asc_lock(PIPE_M, l0_slot);
                    asc::te::mmad_params params{cur_m, cur_n, cur_k, asc::te::unit_flag_mode::disable, true};
                    params.init_with_zero = (k_block_idx == 0);

                    asc::te::mmad(mmad_atom.with(params), l0_tensor_c, l0_tensor_a, l0_tensor_b);
                    // M_MTE1 reverse synchronization: release the L0 buffer for the next data_load.
                    asc_unlock(PIPE_M, l0_slot);
                    mte1_db_flag ^= 1;

                    // ---- Copy in the next L1 chunk so data_copy_in overlaps with compute in the pipeline ----
                    // Trigger conditions:
                    //   (1) k_block_idx == 0: when computing the first K block, A1/B1 Pong has not been used and can be
                    //   copied in directly. (2) The last base_k of the current L1 chunk has been consumed, so the buffer
                    //   can be overwritten.
                    if (((k_block_idx == 0) || ((k_offset_in_data_chunk + 1) == data_chunk_step)) &&
                        data_next_k_chunk_idx < k_loop_count) {
                        const auto& l1_write_buf_a = (data_copy_in_idx == 0) ? l1_tensor_a_ping : l1_tensor_a_pong;
                        const auto& l1_write_buf_b = (data_copy_in_idx == 0) ? l1_tensor_b_ping : l1_tensor_b_pong;
                        uint32_t data_write_slot = get_data_slot(data_copy_in_idx);
                        asc_lock(PIPE_MTE2, data_write_slot);
                        // A: GM -> L1
                        asc::te::copy(
                            gm_to_l1_atom, l1_write_buf_a,
                            a.slice(
                                asc::te::make_coord(m_block_idx * Trait::base_m, data_next_k_chunk_idx * Trait::base_k),
                                asc::te::make_shape(cur_m, step_cur_k)));
                        // B: GM -> L1
                        asc::te::copy(
                            gm_to_l1_atom, l1_write_buf_b,
                            b.slice(
                                asc::te::make_coord(data_next_k_chunk_idx * Trait::base_k, n_block_idx * Trait::base_n),
                                asc::te::make_shape(step_cur_k, cur_n)));

                        asc_unlock(PIPE_MTE2, data_write_slot);
                        data_next_k_chunk_idx += data_chunk_step;
                        data_copy_in_idx ^= 1;
                    }
                    if (((k_block_idx == 0) || ((k_offset_in_scale_chunk + 1) == scale_chunk_step)) &&
                        scale_next_k_chunk_idx < k_loop_count) {
                        const auto& l1_write_buf_as = (scale_copy_in_idx == 0) ? l1_tensor_as_ping : l1_tensor_as_pong;
                        const auto& l1_write_buf_bs = (scale_copy_in_idx == 0) ? l1_tensor_bs_ping : l1_tensor_bs_pong;
                        uint32_t scale_write_slot = get_scale_slot(scale_copy_in_idx);
                        asc_lock(PIPE_MTE2, scale_write_slot);
                        // scale_a: GM -> L1
                        asc::te::copy(
                            gm_to_l1_atom, l1_write_buf_as,
                            as.slice(
                                asc::te::make_coord(m_block_idx * Trait::base_m, scale_next_k_chunk_idx * base_scale_k),
                                asc::te::make_shape(cur_m, step_cur_scale_k)));
                        // scale_b: GM -> L1
                        asc::te::copy(
                            gm_to_l1_atom, l1_write_buf_bs,
                            bs.slice(
                                asc::te::make_coord(scale_next_k_chunk_idx * base_scale_k, n_block_idx * Trait::base_n),
                                asc::te::make_shape(step_cur_scale_k, cur_n)));

                        asc_unlock(PIPE_MTE2, scale_write_slot);
                        scale_next_k_chunk_idx += scale_chunk_step;
                        scale_copy_in_idx ^= 1;
                    }
                }
                // ---- copy quant params: GM -> L1 ----
                asc_lock(PIPE_MTE2, QUANT_INDEX);
                // [Answer-3] 将 quant_scale 从 GM 拷贝到 L1
                asc::te::copy(
                    gm_to_l1_atom, l1_quant_scale,
                    quant_scale.slice(
                        asc::te::make_coord(0, n_block_idx * Trait::base_n),
                        asc::te::make_shape(1, cur_n)));
                asc_unlock(PIPE_MTE2, QUANT_INDEX);

                // ---- copy_out: L0C -> GM (with quantization) ----
                asc_unlock(PIPE_M, L0C_INDEX);
                asc_lock(PIPE_FIX, QUANT_INDEX);
                asc_lock(PIPE_FIX, L0C_INDEX);
                asc::te::l0c_to_gm_params fixpipe_params;
                // [Answer-4] 4 参数 Copy：附加量化张量 l1_quant_scale
                asc::te::copy(
                    l0c_to_gm_atom.with(fixpipe_params),
                    c.slice(
                        asc::te::make_coord(m_block_idx * Trait::base_m, n_block_idx * Trait::base_n),
                        asc::te::make_shape(cur_m, cur_n)),
                    l0_tensor_c, l1_quant_scale);
                asc_unlock(PIPE_FIX, L0C_INDEX);
                asc_unlock(PIPE_FIX, QUANT_INDEX);
            }
        }
    }

    __aicore__ inline void init_compute_params()
    {
        // ---- 1. Compute current-core M/N iteration indexes and GM start offset ----
        constexpr uint32_t m_iter = AscendC::Std::ceil_div(Trait::M, Trait::single_core_m);
        m_iter_idx = block_idx % m_iter;
        n_iter_idx = block_idx / m_iter;

        // ---- 2. Compute actual M/N dimensions of the current core ----
        // The last block may be smaller than single_core.
        actual_single_core_m = Trait::M - m_iter_idx * Trait::single_core_m;
        actual_single_core_m = actual_single_core_m < Trait::single_core_m ? actual_single_core_m : Trait::single_core_m;
        actual_single_core_n = Trait::N - n_iter_idx * Trait::single_core_n;
        actual_single_core_n = actual_single_core_n < Trait::single_core_n ? actual_single_core_n : Trait::single_core_n;

        // ---- 3. Compute the loop counts for the M/N/K dimensions ----
        m_loop_count = AscendC::Std::ceil_div(actual_single_core_m, Trait::base_m);
        n_loop_count = AscendC::Std::ceil_div(actual_single_core_n, Trait::base_n);
        k_loop_count = AscendC::Std::ceil_div(Trait::single_core_k, Trait::base_k);

        // ---- 4. Compute the M/N-direction tiling parameters ----
        tail_m = (actual_single_core_m % Trait::base_m != 0) ? actual_single_core_m % Trait::base_m : Trait::base_m;
        tail_n = (actual_single_core_n % Trait::base_n != 0) ? actual_single_core_n % Trait::base_n : Trait::base_n;
    }

private:
    uint32_t actual_single_core_m, actual_single_core_n;
    uint32_t m_iter_idx, n_iter_idx;
    uint32_t m_loop_count, n_loop_count, k_loop_count;
    uint32_t tail_m, tail_n;
    uint8_t mte1_db_flag = 0;

    static constexpr uint32_t scale_k = align_even(AscendC::Std::ceil_div(Trait::K, SCALE_CEIL_NUMBER));
    static constexpr uint32_t base_scale_k = align_even(AscendC::Std::ceil_div(Trait::base_k, SCALE_CEIL_NUMBER));

    static constexpr size_t l1_size_a = Trait::base_m * (Trait::base_k * Trait::step_k);
    static constexpr size_t l1_size_b = (Trait::base_k * Trait::step_k) * Trait::base_n;
    static constexpr size_t l1_size_as = Trait::base_m * (base_scale_k * Trait::step_k * Trait::scale_factor_k);
    static constexpr size_t l1_size_bs = (base_scale_k * Trait::step_k * Trait::scale_factor_k) * Trait::base_n;

    static constexpr size_t l0_size_a = Trait::base_m * Trait::base_k;
    static constexpr size_t l0_size_b = Trait::base_k * Trait::base_n;
    static constexpr size_t l0_size_c = Trait::base_m * Trait::base_n;

    static constexpr auto gm_to_l1_atom = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    static constexpr auto l1_to_l0a_atom = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
    static constexpr auto l1_to_l0b_atom = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
    static constexpr auto l1_to_l0scalea_atom = asc::te::make_copy(asc::te::copy_l1_to_l0scalea{});
    static constexpr auto l1_to_l0scaleb_atom = asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{});
    static constexpr auto l0c_to_gm_atom = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
    static constexpr auto mmad_atom = asc::te::make_mmad(asc::te::mmad_operation{}, mmad_trait_mx{});

    static constexpr uint32_t data_chunk_step = Trait::step_k;
    static constexpr uint32_t scale_chunk_step = Trait::step_k * Trait::scale_factor_k;
};

template <typename Trait>
__global__ __cube__ void mmadmx_quant_custom(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* as, __gm__ uint8_t* bs, __gm__ uint8_t* c,
    __gm__ uint8_t* quant_scale, __gm__ uint8_t* quant_offset)
{
    asc_init();
    kernel_mmadmx_quant<Trait> op;
    op.process(
        reinterpret_cast<__gm__ fp4x2_e1m2_t*>(a), reinterpret_cast<__gm__ fp4x2_e1m2_t*>(b),
        reinterpret_cast<__gm__ fp8_e8m0_t*>(as), reinterpret_cast<__gm__ fp8_e8m0_t*>(bs),
        reinterpret_cast<__gm__ int8_t*>(c), reinterpret_cast<__gm__ uint64_t*>(quant_scale),
        reinterpret_cast<__gm__ uint64_t*>(quant_offset));

    asc_sync_pipe(PIPE_ALL);
}

#endif

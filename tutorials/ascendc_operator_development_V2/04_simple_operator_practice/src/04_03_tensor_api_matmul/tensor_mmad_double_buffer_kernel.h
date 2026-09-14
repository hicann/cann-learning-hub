#ifndef TENSOR_MMAD_DOUBLE_BUFFER_KERNEL_H
#define TENSOR_MMAD_DOUBLE_BUFFER_KERNEL_H

#include "basic_api/kernel_operator_block_sync_intf.h"
#include "tensor_api/tensor.h"

template <
    int32_t M, int32_t N, int32_t K,
    int32_t singleM, int32_t singleN, int32_t singleK,
    int32_t baseM, int32_t baseN, int32_t baseK>
__cube__ __global__ void TensorMmadDoubleBufferKernel(__gm__ half *x, __gm__ half *y, __gm__ half *z)
{
    using namespace asc::te;
    static_assert(M % singleM == 0 && N % singleN == 0 && K % singleK == 0);
    static_assert(singleM % baseM == 0 && singleN % baseN == 0 && singleK % baseK == 0);

    constexpr uint32_t mCoreLoop = M / singleM;
    constexpr uint32_t nCoreLoop = N / singleN;
    constexpr uint32_t mLoop = singleM / baseM;
    constexpr uint32_t nLoop = singleN / baseN;
    constexpr uint32_t kLoop = singleK / baseK;

    uint32_t blockIdx = block_idx;
    uint32_t mCoreIdx = blockIdx % mCoreLoop;
    uint32_t nCoreIdx = blockIdx / mCoreLoop;

    auto gmATensor = make_tensor(make_mem_ptr(x), make_frame_layout<nd_layout_ptn>(M, K));
    auto gmBTensor = make_tensor(make_mem_ptr(y), make_frame_layout<dn_layout_ptn>(K, N));
    auto gmCTensor = make_tensor(make_mem_ptr(z), make_frame_layout<nd_layout_ptn>(M, N));

    auto gmASingle = gmATensor.slice(make_coord(mCoreIdx * singleM, 0), make_shape(singleM, singleK));
    auto gmBSingle = gmBTensor.slice(make_coord(0, nCoreIdx * singleN), make_shape(singleK, singleN));
    auto gmCSingle = gmCTensor.slice(make_coord(mCoreIdx * singleM, nCoreIdx * singleN), make_shape(singleM, singleN));

    __cbuf__ half l1ABuf[2 * baseM * baseK];
    __cbuf__ half l1BBuf[2 * baseK * baseN];
    __ca__ half l0ABuf[2 * baseM * baseK];
    __cb__ half l0BBuf[2 * baseK * baseN];
    __cc__ float l0CBuf[baseM * baseN];

    auto copyGM2L1Atom = make_copy(copy_gm_to_l1{}, gm_to_l1_trait_default{});
    auto copyL12L0AAtom = make_copy(copy_l1_to_l0a{}, l1_to_l0a_trait_default{});
    auto copyL12L0BAtom = make_copy(copy_l1_to_l0b{}, l1_to_l0b_trait_default{});
    auto copyL0C2GMAtom = make_copy(copy_l0c_to_gm{}, l0c_to_gm_trait_default{});
    auto mmadAtom = make_mmad(mmad_operation{}, mmad_trait_default{});
    auto l0CTensor = make_tensor(make_mem_ptr(l0CBuf), make_frame_layout<nz_layout_ptn>(baseM, baseN));

    constexpr event_t L1_EVENT_0 = EVENT_ID0;
    constexpr event_t L1_EVENT_1 = EVENT_ID1;
    constexpr event_t L0_EVENT_0 = EVENT_ID2;
    constexpr event_t L0_EVENT_1 = EVENT_ID3;
    constexpr event_t L0C_EVENT_ID = EVENT_ID4;
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, L1_EVENT_0);
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, L1_EVENT_1);
    asc_sync_notify(PIPE_M, PIPE_MTE1, L0_EVENT_0);
    asc_sync_notify(PIPE_M, PIPE_MTE1, L0_EVENT_1);
    asc_sync_notify(PIPE_FIX, PIPE_M, L0C_EVENT_ID);

    for (uint32_t mi = 0; mi < mLoop; ++mi) {
        for (uint32_t ni = 0; ni < nLoop; ++ni) {
            for (uint32_t ki = 0; ki < kLoop; ++ki) {
                uint32_t bufIdx = ki & 1;
                event_t l1Event = (bufIdx == 0) ? L1_EVENT_0 : L1_EVENT_1;
                event_t l0Event = (bufIdx == 0) ? L0_EVENT_0 : L0_EVENT_1;

                auto l1ATensor = make_tensor(make_mem_ptr(l1ABuf + bufIdx * baseM * baseK), make_frame_layout<nz_layout_ptn, half>(baseM, baseK));
                auto l1BTensor = make_tensor(make_mem_ptr(l1BBuf + bufIdx * baseK * baseN), make_frame_layout<zn_layout_ptn, half>(baseK, baseN));
                auto l0ATensor = make_tensor(make_mem_ptr(l0ABuf + bufIdx * baseM * baseK), make_frame_layout<nz_layout_ptn, half>(baseM, baseK));
                auto l0BTensor = make_tensor(make_mem_ptr(l0BBuf + bufIdx * baseK * baseN), make_frame_layout<zn_layout_ptn, half>(baseK, baseN));

                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, l1Event);
                copy(copyGM2L1Atom, l1ATensor, gmASingle.slice(make_coord(mi * baseM, ki * baseK), make_shape(baseM, baseK)));
                copy(copyGM2L1Atom, l1BTensor, gmBSingle.slice(make_coord(ki * baseK, ni * baseN), make_shape(baseK, baseN)));
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, l1Event);
                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, l1Event);

                asc_sync_wait(PIPE_M, PIPE_MTE1, l0Event);
                copy(copyL12L0AAtom, l0ATensor, l1ATensor);
                copy(copyL12L0BAtom, l0BTensor, l1BTensor);
                asc_sync_notify(PIPE_MTE1, PIPE_MTE2, l1Event);
                asc_sync_notify(PIPE_MTE1, PIPE_M, l0Event);
                asc_sync_wait(PIPE_MTE1, PIPE_M, l0Event);

                if (ki == 0) {
                    asc_sync_wait(PIPE_FIX, PIPE_M, L0C_EVENT_ID);
                }
                mmad_params params{baseM, baseN, baseK, unit_flag_mode::disable, (ki == 0)};
                mmad(mmadAtom.with(params), l0CTensor, l0ATensor, l0BTensor);
                if (ki + 1 == kLoop) {
                    asc_sync_notify(PIPE_M, PIPE_FIX, L0C_EVENT_ID);
                }
                asc_sync_notify(PIPE_M, PIPE_MTE1, l0Event);
            }

            asc_sync_wait(PIPE_M, PIPE_FIX, L0C_EVENT_ID);
            copy(copyL0C2GMAtom, gmCSingle.slice(make_coord(mi * baseM, ni * baseN), make_shape(baseM, baseN)), l0CTensor);
            asc_sync_notify(PIPE_FIX, PIPE_M, L0C_EVENT_ID);
        }
    }

    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, L1_EVENT_0);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, L1_EVENT_1);
    asc_sync_wait(PIPE_M, PIPE_MTE1, L0_EVENT_0);
    asc_sync_wait(PIPE_M, PIPE_MTE1, L0_EVENT_1);
    asc_sync_wait(PIPE_FIX, PIPE_M, L0C_EVENT_ID);
}

#endif

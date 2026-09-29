#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void row_softmax_kernel_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool mask = asc_create_mask_b32(PAT_ALL);
  auto vreg_0 = simd_inst::vdup<float>(float(-0x1.fffff966ad924p+127f/*-3.402823e+38*/), mask, MODE_ZEROING);
  auto vreg_1 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[0])), 0);
  auto vreg_2 = simd_inst::vcmax(vreg_1, mask, MODE_ZEROING);
  auto vreg_3 = simd_inst::vmax(vreg_0, vreg_2, mask, MODE_ZEROING);
  auto vreg_4 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[64])), 0);
  auto vreg_5 = simd_inst::vcmax(vreg_4, mask, MODE_ZEROING);
  auto vreg_6 = simd_inst::vmax(vreg_3, vreg_5, mask, MODE_ZEROING);
  auto vreg_7 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[128])), 0);
  auto vreg_8 = simd_inst::vcmax(vreg_7, mask, MODE_ZEROING);
  auto vreg_9 = simd_inst::vmax(vreg_6, vreg_8, mask, MODE_ZEROING);
  auto vreg_10 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[192])), 0);
  auto vreg_11 = simd_inst::vcmax(vreg_10, mask, MODE_ZEROING);
  auto vreg_12 = simd_inst::vmax(vreg_9, vreg_11, mask, MODE_ZEROING);
  auto vreg_13 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[256])), 0);
  auto vreg_14 = simd_inst::vcmax(vreg_13, mask, MODE_ZEROING);
  auto vreg_15 = simd_inst::vmax(vreg_12, vreg_14, mask, MODE_ZEROING);
  auto vreg_16 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[320])), 0);
  auto vreg_17 = simd_inst::vcmax(vreg_16, mask, MODE_ZEROING);
  auto vreg_18 = simd_inst::vmax(vreg_15, vreg_17, mask, MODE_ZEROING);
  auto vreg_19 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[384])), 0);
  auto vreg_20 = simd_inst::vcmax(vreg_19, mask, MODE_ZEROING);
  auto vreg_21 = simd_inst::vmax(vreg_18, vreg_20, mask, MODE_ZEROING);
  auto vreg_22 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[448])), 0);
  auto vreg_23 = simd_inst::vcmax(vreg_22, mask, MODE_ZEROING);
  auto vreg_24 = simd_inst::vmax(vreg_21, vreg_23, mask, MODE_ZEROING);
  simd_inst::vsts_1st(vreg_24, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[1536])), 0, mask);
}

__simd_vf__ inline void row_softmax_kernel_kernel_simd_vf_1(__ubuf__ uint8_t* buf_dyn_shmem) {
  auto vreg_1 = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[1536])), 0);
  for (int32_t vchunk = 0; vchunk < 8; ++vchunk) {
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    auto vreg_0 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(vchunk * 64)])), 0);
    auto vreg_2 = simd_inst::vsub(vreg_0, vreg_1, mask, MODE_ZEROING);
    auto vreg_3 = simd_inst::vexp(vreg_2, mask, MODE_ZEROING);
    simd_inst::vsts_norm(vreg_3, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((vchunk * 64) + 512)])), 0, mask);
  }
}

__simd_vf__ inline void row_softmax_kernel_kernel_simd_vf_2(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool mask = asc_create_mask_b32(PAT_ALL);
  auto vreg_0 = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), mask, MODE_ZEROING);
  auto vreg_1 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[512])), 0);
  auto vreg_2 = simd_inst::vcadd(vreg_1, mask, MODE_ZEROING);
  auto vreg_3 = simd_inst::vadd(vreg_0, vreg_2, mask, MODE_ZEROING);
  auto vreg_4 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[576])), 0);
  auto vreg_5 = simd_inst::vcadd(vreg_4, mask, MODE_ZEROING);
  auto vreg_6 = simd_inst::vadd(vreg_3, vreg_5, mask, MODE_ZEROING);
  auto vreg_7 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[640])), 0);
  auto vreg_8 = simd_inst::vcadd(vreg_7, mask, MODE_ZEROING);
  auto vreg_9 = simd_inst::vadd(vreg_6, vreg_8, mask, MODE_ZEROING);
  auto vreg_10 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[704])), 0);
  auto vreg_11 = simd_inst::vcadd(vreg_10, mask, MODE_ZEROING);
  auto vreg_12 = simd_inst::vadd(vreg_9, vreg_11, mask, MODE_ZEROING);
  auto vreg_13 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[768])), 0);
  auto vreg_14 = simd_inst::vcadd(vreg_13, mask, MODE_ZEROING);
  auto vreg_15 = simd_inst::vadd(vreg_12, vreg_14, mask, MODE_ZEROING);
  auto vreg_16 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[832])), 0);
  auto vreg_17 = simd_inst::vcadd(vreg_16, mask, MODE_ZEROING);
  auto vreg_18 = simd_inst::vadd(vreg_15, vreg_17, mask, MODE_ZEROING);
  auto vreg_19 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[896])), 0);
  auto vreg_20 = simd_inst::vcadd(vreg_19, mask, MODE_ZEROING);
  auto vreg_21 = simd_inst::vadd(vreg_18, vreg_20, mask, MODE_ZEROING);
  auto vreg_22 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[960])), 0);
  auto vreg_23 = simd_inst::vcadd(vreg_22, mask, MODE_ZEROING);
  auto vreg_24 = simd_inst::vadd(vreg_21, vreg_23, mask, MODE_ZEROING);
  simd_inst::vsts_1st(vreg_24, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[1544])), 0, mask);
}

__simd_vf__ inline void row_softmax_kernel_kernel_simd_vf_3(__ubuf__ uint8_t* buf_dyn_shmem) {
  auto vreg_1 = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[1544])), 0);
  for (int32_t vchunk = 0; vchunk < 8; ++vchunk) {
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    auto vreg_0 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((vchunk * 64) + 512)])), 0);
    auto vreg_2 = simd_inst::vdiv_0ulp_ftz_true(vreg_0, vreg_1, mask, MODE_ZEROING);
    simd_inst::vsts_norm(vreg_2, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((vchunk * 64) + 1024)])), 0, mask);
  }
}

extern "C" __global__ __vector__ void row_softmax_kernel_kernel(__gm__ float* X, __gm__ float* Y) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(X[(((int32_t)block_idx) * 512)]))), 1, 2048, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 2048, 2048);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  row_softmax_kernel_kernel_simd_vf_0(buf_dyn_shmem);
  row_softmax_kernel_kernel_simd_vf_1(buf_dyn_shmem);
  row_softmax_kernel_kernel_simd_vf_2(buf_dyn_shmem);
  row_softmax_kernel_kernel_simd_vf_3(buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(Y[(((int32_t)block_idx) * 512)]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[1024]))), 1, 2048, static_cast<asc_store_l2_cache_mode>(4), 2048, 2048);
}


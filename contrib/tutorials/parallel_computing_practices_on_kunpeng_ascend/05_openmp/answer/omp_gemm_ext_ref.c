#include <arm_neon.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NTIMES 5      // Repetitions for the optimized versions
#define NTIMES_BASE 1 // Repetitions for the serial baseline (one run takes seconds)

// Cache block sizes, unchanged from Chapter 3: one A block + one B block +
// one C block = 3 x 64 x 64 x 4 B = 48 KiB, which fits in L1D.
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ---------------------------------------------------------
// Timing and result checking (identical to Chapter 3)
// ---------------------------------------------------------
double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// Relative tolerance, SCALED WITH the reduction length K.
// The rounding error of a float32 dot product grows roughly linearly with the
// number of accumulation steps, so a FIXED absolute tolerance would report a
// bogus FAIL once K becomes large.  tol = 2e-8 * K * max|C_ref| tracks that
// growth while still catching real defects.
const char* check_result(const float* restrict ref, const float* restrict test,
                         int M, int N, int K) {
  double max_diff = 0.0, max_ref = 0.0;
  for (size_t i = 0; i < (size_t)M * (size_t)N; i++) {
    double diff = fabs((double)ref[i] - (double)test[i]);
    if (diff > max_diff) max_diff = diff;
    if (fabs((double)ref[i]) > max_ref) max_ref = fabs((double)ref[i]);
  }
  double tol = 2e-8 * (double)K * (max_ref > 1.0 ? max_ref : 1.0);
  return (max_diff <= tol) ? "PASS" : "FAIL";
}

// Print one row of the report table (skip the division when t <= 0)
void report(const char* name, double t, double t_base, double ops,
            const char* chk) {
  if (t <= 0.0) {
    printf("| %-15s | %9.3f |     -   |      - |  %-4s |\n", name, t, chk);
  } else {
    printf("| %-15s | %9.3f | %5.2f x | %6.2f |  %-4s |\n", name, t,
           t_base / t, ops / (t * 1e6), chk);
  }
}

// ---------------------------------------------------------
// Edge handling (identical to Chapter 3): when a dimension is not a multiple
// of 4, the remaining strip is finished with scalar code.
//   _store overwrites  (C  = sum)
//   _accum accumulates (C += sum): for K-blocked versions
// ---------------------------------------------------------
void gemm_edge_store(int M, int N, int K, const float* A, int lda,
                     const float* B, int ldb, float* C, int ldc) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) sum += A[i * lda + k] * B[k * ldb + j];
      C[i * ldc + j] = sum;
    }
  }
}

void gemm_edge_accum(int M, int N, int K, const float* A, int lda,
                     const float* B, int ldb, float* C, int ldc) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) sum += A[i * lda + k] * B[k * ldb + j];
      C[i * ldc + j] += sum;
    }
  }
}

// ---------------------------------------------------------
// 1. Serial baseline (auto-vectorization forced off)
// ---------------------------------------------------------
#if defined(__GNUC__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void gemm_serial_no_vec(const float* restrict A, const float* restrict B, float* restrict C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// ---------------------------------------------------------
// The Chapter 3 result we start from: 4x4 register-blocked micro-kernel
// operating on a PACKED copy of the B block.  On a single core this reached
// 27.71 GFLOPS = about 2/3 of the single-core peak, while using only ~5% of
// the available memory bandwidth.  That headroom is exactly what OpenMP is
// going to spend.
// ---------------------------------------------------------
void pack_matrix_B(int K, int N, const float* B, int ldb, float* buffer) {
  int N_aligned = (N + 3) & ~3;  // round the row stride up to 4 floats
  float* ptr = buffer;
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < N; j++) *ptr++ = B[k * ldb + j];
    for (int j = N; j < N_aligned; j++) *ptr++ = 0.0f;  // pad, keeps 16 B align
  }
}

void microkernel_4x4_packed(int M, int N, int K, const float* A, int lda,
                            const float* B_packed, float* C, int ldc) {
  int ldb = (N + 3) & ~3;  // row stride of the packed buffer
  int i = 0;
  for (; i <= M - 4; i += 4) {
    int j = 0;
    for (; j <= N - 4; j += 4) {
      float32x4_t c_0 = vdupq_n_f32(0.0f);
      float32x4_t c_1 = vdupq_n_f32(0.0f);
      float32x4_t c_2 = vdupq_n_f32(0.0f);
      float32x4_t c_3 = vdupq_n_f32(0.0f);

      for (int k = 0; k < K; k++) {
        float32x4_t b_vec = vld1q_f32(&B_packed[k * ldb + j]);  // sequential
        c_0 = vfmaq_f32(c_0, vdupq_n_f32(A[(i + 0) * lda + k]), b_vec);
        c_1 = vfmaq_f32(c_1, vdupq_n_f32(A[(i + 1) * lda + k]), b_vec);
        c_2 = vfmaq_f32(c_2, vdupq_n_f32(A[(i + 2) * lda + k]), b_vec);
        c_3 = vfmaq_f32(c_3, vdupq_n_f32(A[(i + 3) * lda + k]), b_vec);
      }
      vst1q_f32(&C[(i + 0) * ldc + j],
                vaddq_f32(vld1q_f32(&C[(i + 0) * ldc + j]), c_0));
      vst1q_f32(&C[(i + 1) * ldc + j],
                vaddq_f32(vld1q_f32(&C[(i + 1) * ldc + j]), c_1));
      vst1q_f32(&C[(i + 2) * ldc + j],
                vaddq_f32(vld1q_f32(&C[(i + 2) * ldc + j]), c_2));
      vst1q_f32(&C[(i + 3) * ldc + j],
                vaddq_f32(vld1q_f32(&C[(i + 3) * ldc + j]), c_3));
    }
    if (j < N) gemm_edge_accum(4, N - j, K, &A[i * lda], lda, &B_packed[j], ldb, &C[i * ldc + j], ldc);
  }
  if (i < M) gemm_edge_accum(M - i, N, K, &A[i * lda], lda, B_packed, ldb, &C[i * ldc], ldc);
}

// ---------------------------------------------------------
// 2. Single-thread NEON (Chapter 3 v5, unchanged) - the real starting line
//    NOTE the single packed_B buffer allocated ONCE for the whole call.
//    Perfectly fine with one thread; remember it for v3.
// ---------------------------------------------------------
void gemm_neon_1t(const float* restrict A, const float* restrict B,
                  float* restrict C, int M, int N, int K) {
  int ldb_pack = (BLOCK_N + 3) & ~3;
  float* packed_B =
      (float*)aligned_alloc(16, (size_t)BLOCK_K * ldb_pack * sizeof(float));
  if (!packed_B) {
    printf("Error: packed buffer allocation failed.\n");
    return;
  }

  for (int ii = 0; ii < M; ii += BLOCK_M) {
    int cur_M = MIN(BLOCK_M, M - ii);
    for (int jj = 0; jj < N; jj += BLOCK_N) {
      int cur_N = MIN(BLOCK_N, N - jj);

      for (int r = 0; r < cur_M; r++)
        memset(&C[(ii + r) * N + jj], 0, (size_t)cur_N * sizeof(float));

      for (int kk = 0; kk < K; kk += BLOCK_K) {
        int cur_K = MIN(BLOCK_K, K - kk);
        pack_matrix_B(cur_K, cur_N, &B[kk * N + jj], N, packed_B);
        microkernel_4x4_packed(cur_M, cur_N, cur_K, &A[ii * K + kk], K,
                               packed_B, &C[ii * N + jj], N);
      }
    }
  }
  free(packed_B);
}

// ---------------------------------------------------------
// 3. Naive OpenMP: parallelize the i loop of the SCALAR triple loop.
//    Each i is an independent row of C, so there is no race and no reduction.
//    This is the version whose SPEEDUP looks best and whose GFLOPS is worst -
//    the central lesson of this lab.
// ---------------------------------------------------------
void gemm_omp_naive(const float* restrict A, const float* restrict B,
                    float* restrict C, int M, int N, int K, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static) \
    default(none) shared(A, B, C, M, N, K)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// ---------------------------------------------------------
// 3. The Chapter 3 v4 kernel: 4x4 register blocking + cache blocking, with NO
//    packing step.  The point that matters for this chapter: this routine uses
//    NO scratch buffer at all.  It only reads A and B and writes disjoint
//    blocks of C.  Remember that when you get to v4.
// ---------------------------------------------------------
void microkernel_4x4(int M, int N, int K, const float* A, int lda,
                     const float* B, int ldb, float* C, int ldc) {
  int i = 0;
  for (; i <= M - 4; i += 4) {
    int j = 0;
    for (; j <= N - 4; j += 4) {
      float32x4_t c_0 = vdupq_n_f32(0.0f);
      float32x4_t c_1 = vdupq_n_f32(0.0f);
      float32x4_t c_2 = vdupq_n_f32(0.0f);
      float32x4_t c_3 = vdupq_n_f32(0.0f);

      for (int k = 0; k < K; k++) {
        float32x4_t b_vec = vld1q_f32(&B[k * ldb + j]);
        c_0 = vfmaq_f32(c_0, vdupq_n_f32(A[(i + 0) * lda + k]), b_vec);
        c_1 = vfmaq_f32(c_1, vdupq_n_f32(A[(i + 1) * lda + k]), b_vec);
        c_2 = vfmaq_f32(c_2, vdupq_n_f32(A[(i + 2) * lda + k]), b_vec);
        c_3 = vfmaq_f32(c_3, vdupq_n_f32(A[(i + 3) * lda + k]), b_vec);
      }
      vst1q_f32(&C[(i + 0) * ldc + j],
                vaddq_f32(vld1q_f32(&C[(i + 0) * ldc + j]), c_0));
      vst1q_f32(&C[(i + 1) * ldc + j],
                vaddq_f32(vld1q_f32(&C[(i + 1) * ldc + j]), c_1));
      vst1q_f32(&C[(i + 2) * ldc + j],
                vaddq_f32(vld1q_f32(&C[(i + 2) * ldc + j]), c_2));
      vst1q_f32(&C[(i + 3) * ldc + j],
                vaddq_f32(vld1q_f32(&C[(i + 3) * ldc + j]), c_3));
    }
    if (j < N) gemm_edge_accum(4, N - j, K, &A[i * lda], lda, &B[j], ldb, &C[i * ldc + j], ldc);
  }
  if (i < M) gemm_edge_accum(M - i, N, K, &A[i * lda], lda, B, ldb, &C[i * ldc], ldc);
}

// ---------------------------------------------------------
// 4. First fusion: OpenMP x the cache-blocked NEON kernel.
//
//    ONE pragma, nothing else changed - and it is correct.  Check the three
//    criteria against this loop nest:
//      (1) no cross-iteration dependence: each (ii, jj) task computes its own
//          C block from scratch;
//      (2) disjoint writes: task (ii, jj) writes only C[ii..ii+63][jj..jj+63];
//      (3) no scratch buffer at all - microkernel_4x4 works in registers and
//          reads A and B in place.
//    All three hold, so nothing needs to be made private and no lock is needed.
//
//    collapse(2) matters here: parallelising ii alone gives only M/64 = 16
//    tasks at 1024^3, which cannot be balanced over more than a handful of
//    threads.  Collapsing gives (M/64) x (N/64) = 256.
// ---------------------------------------------------------
void gemm_omp_neon_tiled(const float* restrict A, const float* restrict B,
                         float* restrict C, int M, int N, int K, int threads) {
#pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic) \
    default(none) shared(A, B, C, M, N, K)
  for (int ii = 0; ii < M; ii += BLOCK_M) {
    for (int jj = 0; jj < N; jj += BLOCK_N) {
      int cur_M = MIN(BLOCK_M, M - ii);
      int cur_N = MIN(BLOCK_N, N - jj);

      for (int r = 0; r < cur_M; r++)
        memset(&C[(ii + r) * N + jj], 0, (size_t)cur_N * sizeof(float));

      for (int kk = 0; kk < K; kk += BLOCK_K) {
        int cur_K = MIN(BLOCK_K, K - kk);
        microkernel_4x4(cur_M, cur_N, cur_K, &A[ii * K + kk], K,
                        &B[kk * N + jj], N, &C[ii * N + jj], N);
      }
    }
  }
}

// ---------------------------------------------------------
// 5. [DELIBERATELY BROKEN] The SAME pragma as v3, this time on the PACKED
//    kernel - and now it is wrong.
//
//    Criteria (1) and (2) still hold: the C blocks are disjoint, exactly as in
//    v3.  What changed is criterion (3): gemm_neon_1t owns a scratch buffer,
//    packed_B, and it is allocated ONCE, OUTSIDE the parallel region.  The
//    pointer is therefore shared, so every thread packs its own B block into
//    THE SAME memory and then reads back whatever the last writer left there.
//
//    The result is silently wrong: no crash, no compiler warning, no OpenMP
//    error - just a FAIL in the Check column.
//
//    This function is kept in the benchmark on purpose.  Do not "fix" it here;
//    v5 shows the fix, and the diff is one line.
// ---------------------------------------------------------
void gemm_omp_pack_race(const float* restrict A, const float* restrict B,
                        float* restrict C, int M, int N, int K, int threads) {
  int ldb_pack = (BLOCK_N + 3) & ~3;
  float* packed_B =
      (float*)aligned_alloc(16, (size_t)BLOCK_K * ldb_pack * sizeof(float));
  if (!packed_B) return;

#pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic) \
    default(none) shared(A, B, C, M, N, K, packed_B)
  for (int ii = 0; ii < M; ii += BLOCK_M) {
    for (int jj = 0; jj < N; jj += BLOCK_N) {
      int cur_M = MIN(BLOCK_M, M - ii);
      int cur_N = MIN(BLOCK_N, N - jj);

      for (int r = 0; r < cur_M; r++)
        memset(&C[(ii + r) * N + jj], 0, (size_t)cur_N * sizeof(float));

      for (int kk = 0; kk < K; kk += BLOCK_K) {
        int cur_K = MIN(BLOCK_K, K - kk);
        pack_matrix_B(cur_K, cur_N, &B[kk * N + jj], N, packed_B);   // RACE
        microkernel_4x4_packed(cur_M, cur_N, cur_K, &A[ii * K + kk], K,
                               packed_B, &C[ii * N + jj], N);        // RACE
      }
    }
  }
  free(packed_B);
}

// ---------------------------------------------------------
// 6. OpenMP x the packed NEON kernel, done right.
//
//    Exactly ONE thing changes with respect to v4: the packed buffer is
//    allocated INSIDE the parallel region, so every thread owns one.  That
//    requires splitting "omp parallel for" into "omp parallel { omp for }",
//    because the allocation must happen once per thread, not once per task.
//
//    collapse(2) and schedule(dynamic) are carried over from v3 unchanged -
//    they were never the problem.
// ---------------------------------------------------------
void gemm_omp_pack(const float* restrict A, const float* restrict B,
                   float* restrict C, int M, int N, int K, int threads) {
  int ldb_pack = (BLOCK_N + 3) & ~3;
  size_t buf_bytes = (size_t)BLOCK_K * ldb_pack * sizeof(float);

#pragma omp parallel num_threads(threads) \
    default(none) shared(A, B, C, M, N, K, buf_bytes)
  {
    // Declared inside the parallel region => one private buffer per thread.
    float* packed_B = (float*)aligned_alloc(16, buf_bytes);

    if (packed_B) {
#pragma omp for collapse(2) schedule(dynamic)
      for (int ii = 0; ii < M; ii += BLOCK_M) {
        for (int jj = 0; jj < N; jj += BLOCK_N) {
          int cur_M = MIN(BLOCK_M, M - ii);
          int cur_N = MIN(BLOCK_N, N - jj);

          // Threads write to disjoint (ii, jj) blocks of C: no race, no lock.
          for (int r = 0; r < cur_M; r++)
            memset(&C[(ii + r) * N + jj], 0, (size_t)cur_N * sizeof(float));

          for (int kk = 0; kk < K; kk += BLOCK_K) {
            int cur_K = MIN(BLOCK_K, K - kk);
            pack_matrix_B(cur_K, cur_N, &B[kk * N + jj], N, packed_B);
            microkernel_4x4_packed(cur_M, cur_N, cur_K, &A[ii * K + kk], K,
                                   packed_B, &C[ii * N + jj], N);
          }
        }
      }
      free(packed_B);
    }
  }
}

// ---------------------------------------------------------
// 7. [EXTENSION LAB] OpenMP x the 8x8 double-packed micro-kernel
//    The kernel itself is the Chapter 3 extension answer, reproduced here
//    unchanged.  What is new is purely the OpenMP side: BOTH packing buffers
//    are now per-thread, and the (ii, jj) iteration space is what gets
//    distributed.
//    Arithmetic intensity 2.0 FLOP/Byte instead of 1.0.
// ---------------------------------------------------------
#define EXT_BLOCK_M 128
#define EXT_BLOCK_N 128
#define EXT_BLOCK_K 128

// Pack an (M x K) block of A into strips of 8 rows, k-major.
void pack_A_panel_8(int K, int M, const float* A, int lda, float* buffer) {
  for (int i = 0; i < M; i += 8) {
    for (int k = 0; k < K; k++) {
      for (int r = 0; r < 8; r++)
        *buffer++ = (i + r < M) ? A[(i + r) * lda + k] : 0.0f;
    }
  }
}

// Pack a (K x N) block of B into strips of 8 columns, k-major.
void pack_B_panel_8(int K, int N, const float* B, int ldb, float* buffer) {
  for (int j = 0; j < N; j += 8) {
    for (int k = 0; k < K; k++) {
      for (int c = 0; c < 8; c++)
        *buffer++ = (j + c < N) ? B[k * ldb + (j + c)] : 0.0f;
    }
  }
}

// 8x8 micro-kernel: C[0..7][0..7] += A_panel * B_panel
void microkernel_8x8_packed(int K, const float* A_panel, const float* B_panel,
                            float* C, int ldc) {
  float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f);
  float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f);
  float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f);
  float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f);
  float32x4_t c40 = vdupq_n_f32(0.0f), c41 = vdupq_n_f32(0.0f);
  float32x4_t c50 = vdupq_n_f32(0.0f), c51 = vdupq_n_f32(0.0f);
  float32x4_t c60 = vdupq_n_f32(0.0f), c61 = vdupq_n_f32(0.0f);
  float32x4_t c70 = vdupq_n_f32(0.0f), c71 = vdupq_n_f32(0.0f);

  const float* a_ptr = A_panel;
  const float* b_ptr = B_panel;

  for (int k = 0; k < K; k++) {
    float32x4_t b0 = vld1q_f32(b_ptr);
    float32x4_t b1 = vld1q_f32(b_ptr + 4);
    b_ptr += 8;
    __builtin_prefetch(b_ptr + 64, 0, 3);

    float a0 = *a_ptr++;  c00 = vfmaq_n_f32(c00, b0, a0);  c01 = vfmaq_n_f32(c01, b1, a0);
    float a1 = *a_ptr++;  c10 = vfmaq_n_f32(c10, b0, a1);  c11 = vfmaq_n_f32(c11, b1, a1);
    float a2 = *a_ptr++;  c20 = vfmaq_n_f32(c20, b0, a2);  c21 = vfmaq_n_f32(c21, b1, a2);
    float a3 = *a_ptr++;  c30 = vfmaq_n_f32(c30, b0, a3);  c31 = vfmaq_n_f32(c31, b1, a3);
    float a4 = *a_ptr++;  c40 = vfmaq_n_f32(c40, b0, a4);  c41 = vfmaq_n_f32(c41, b1, a4);
    float a5 = *a_ptr++;  c50 = vfmaq_n_f32(c50, b0, a5);  c51 = vfmaq_n_f32(c51, b1, a5);
    float a6 = *a_ptr++;  c60 = vfmaq_n_f32(c60, b0, a6);  c61 = vfmaq_n_f32(c61, b1, a6);
    float a7 = *a_ptr++;  c70 = vfmaq_n_f32(c70, b0, a7);  c71 = vfmaq_n_f32(c71, b1, a7);
  }

  vst1q_f32(C + 0 * ldc + 0, vaddq_f32(vld1q_f32(C + 0 * ldc + 0), c00));
  vst1q_f32(C + 0 * ldc + 4, vaddq_f32(vld1q_f32(C + 0 * ldc + 4), c01));
  vst1q_f32(C + 1 * ldc + 0, vaddq_f32(vld1q_f32(C + 1 * ldc + 0), c10));
  vst1q_f32(C + 1 * ldc + 4, vaddq_f32(vld1q_f32(C + 1 * ldc + 4), c11));
  vst1q_f32(C + 2 * ldc + 0, vaddq_f32(vld1q_f32(C + 2 * ldc + 0), c20));
  vst1q_f32(C + 2 * ldc + 4, vaddq_f32(vld1q_f32(C + 2 * ldc + 4), c21));
  vst1q_f32(C + 3 * ldc + 0, vaddq_f32(vld1q_f32(C + 3 * ldc + 0), c30));
  vst1q_f32(C + 3 * ldc + 4, vaddq_f32(vld1q_f32(C + 3 * ldc + 4), c31));
  vst1q_f32(C + 4 * ldc + 0, vaddq_f32(vld1q_f32(C + 4 * ldc + 0), c40));
  vst1q_f32(C + 4 * ldc + 4, vaddq_f32(vld1q_f32(C + 4 * ldc + 4), c41));
  vst1q_f32(C + 5 * ldc + 0, vaddq_f32(vld1q_f32(C + 5 * ldc + 0), c50));
  vst1q_f32(C + 5 * ldc + 4, vaddq_f32(vld1q_f32(C + 5 * ldc + 4), c51));
  vst1q_f32(C + 6 * ldc + 0, vaddq_f32(vld1q_f32(C + 6 * ldc + 0), c60));
  vst1q_f32(C + 6 * ldc + 4, vaddq_f32(vld1q_f32(C + 6 * ldc + 4), c61));
  vst1q_f32(C + 7 * ldc + 0, vaddq_f32(vld1q_f32(C + 7 * ldc + 0), c70));
  vst1q_f32(C + 7 * ldc + 4, vaddq_f32(vld1q_f32(C + 7 * ldc + 4), c71));
}

void gemm_omp_pack_8x8(const float* restrict A, const float* restrict B,
                       float* restrict C, int M, int N, int K, int threads) {
  size_t szA = (size_t)EXT_BLOCK_K * (EXT_BLOCK_M + 8) * sizeof(float);
  size_t szB = (size_t)EXT_BLOCK_K * (EXT_BLOCK_N + 8) * sizeof(float);

#pragma omp parallel num_threads(threads) \
    default(none) shared(A, B, C, M, N, K, szA, szB)
  {
    // TWO per-thread buffers now: A and B are both packed.
    float* packed_A = (float*)aligned_alloc(16, szA);
    float* packed_B = (float*)aligned_alloc(16, szB);

    if (packed_A && packed_B) {
#pragma omp for collapse(2) schedule(dynamic)
      for (int ii = 0; ii < M; ii += EXT_BLOCK_M) {
        for (int jj = 0; jj < N; jj += EXT_BLOCK_N) {
          int cur_M = MIN(EXT_BLOCK_M, M - ii);
          int cur_N = MIN(EXT_BLOCK_N, N - jj);

          for (int r = 0; r < cur_M; r++)
            memset(&C[(ii + r) * N + jj], 0, (size_t)cur_N * sizeof(float));

          for (int kk = 0; kk < K; kk += EXT_BLOCK_K) {
            int cur_K = MIN(EXT_BLOCK_K, K - kk);

            pack_A_panel_8(cur_K, cur_M, &A[ii * K + kk], K, packed_A);
            pack_B_panel_8(cur_K, cur_N, &B[kk * N + jj], N, packed_B);

            for (int i = 0; i < cur_M; i += 8) {
              const float* ap = packed_A + (size_t)i * cur_K;
              for (int j = 0; j < cur_N; j += 8) {
                const float* bp = packed_B + (size_t)j * cur_K;
                if (i + 8 <= cur_M && j + 8 <= cur_N) {
                  microkernel_8x8_packed(cur_K, ap, bp,
                                         &C[(ii + i) * N + (jj + j)], N);
                } else {
                  for (int r = 0; r < 8 && i + r < cur_M; r++) {
                    for (int c = 0; c < 8 && j + c < cur_N; c++) {
                      float sum = 0.0f;
                      for (int k = 0; k < cur_K; k++)
                        sum += ap[k * 8 + r] * bp[k * 8 + c];
                      C[(ii + i + r) * N + (jj + j + c)] += sum;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    free(packed_A);
    free(packed_B);
  }
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    printf("Usage: %s <M> <N> <K> [threads]\n", argv[0]);
    printf("Example: %s 1024 1024 1024 8   (0 or omitted = all cores)\n", argv[0]);
    return 1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int threads = (argc == 5) ? atoi(argv[4]) : 0;
  if (threads <= 0) threads = omp_get_max_threads();
  if (M <= 0 || N <= 0 || K <= 0) return 1;

  printf("============================================================\n");
  printf(" OMP ext: + 8x8 Double-Packed Micro-kernel x OpenMP (C = A * B)\n");
  printf(" Matrix : A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);
  printf(" Threads: %d   (cores visible to OpenMP: %d)\n", threads,
         omp_get_max_threads());
  printf(" Loops  : %d  (baseline: %d)\n", NTIMES, NTIMES_BASE);
  printf("============================================================\n");

  // aligned_alloc requires size to be a multiple of the alignment; round to 16 B
  size_t bytes_A = (((size_t)M * K * sizeof(float)) + 15) & ~(size_t)15;
  size_t bytes_B = (((size_t)K * N * sizeof(float)) + 15) & ~(size_t)15;
  size_t bytes_C = (((size_t)M * N * sizeof(float)) + 15) & ~(size_t)15;

  float* A = (float*)aligned_alloc(16, bytes_A);
  float* B = (float*)aligned_alloc(16, bytes_B);
  float* C_ref = (float*)aligned_alloc(16, bytes_C);   // Golden result
  float* C_test = (float*)aligned_alloc(16, bytes_C);  // Reusable buffer

  if (!A || !B || !C_ref || !C_test) {
    printf("Alloc failed\n");
    return 1;
  }

  // Initialization: different moduli for A and B to avoid degenerate periodic data
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) A[(size_t)i * K + k] = (float)((i + k) % 100) * 0.001f;
  for (int k = 0; k < K; k++)
    for (int j = 0; j < N; j++) B[(size_t)k * N + j] = (float)((k + j) % 97) * 0.001f;

  // Golden reference = serial baseline (this call also first-touches A and B)
  gemm_serial_no_vec(A, B, C_ref, M, N, K);

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double start, end;

  printf("\n--------------------------------------------------------------\n");
  printf("| Method          | Time (ms) | Speedup | GFLOPS | Check |\n");
  printf("|-----------------|-----------|---------|--------|-------|\n");

  // Baseline: serial, no vectorization
  memset(C_test, 0, bytes_C);
  start = get_time_ms();
  for (int t = 0; t < NTIMES_BASE; t++) gemm_serial_no_vec(A, B, C_test, M, N, K);
  end = get_time_ms();
  double t_base = (end - start) / NTIMES_BASE;
  report("Serial (No-Vec)", t_base, t_base, ops, "-");

  // Single-thread NEON, the Chapter 3 result
  memset(C_test, 0, bytes_C);   // Clear: otherwise the previous version's
                                // correct result would mask this one's error
  start = get_time_ms();
  for (int t = 0; t < NTIMES; t++) gemm_neon_1t(A, B, C_test, M, N, K);
  end = get_time_ms();
  double t_1t = (end - start) / NTIMES;
  report("NEON 1-Thread", t_1t, t_base, ops, check_result(C_ref, C_test, M, N, K));

  // Naive OpenMP on the scalar triple loop
  memset(C_test, 0, bytes_C);
  start = get_time_ms();
  for (int t = 0; t < NTIMES; t++) gemm_omp_naive(A, B, C_test, M, N, K, threads);
  end = get_time_ms();
  double t_omp = (end - start) / NTIMES;
  report("OMP Naive", t_omp, t_base, ops, check_result(C_ref, C_test, M, N, K));

  // OpenMP x cache-blocked NEON, one pragma, no scratch buffer
  memset(C_test, 0, bytes_C);
  start = get_time_ms();
  for (int t = 0; t < NTIMES; t++)
    gemm_omp_neon_tiled(A, B, C_test, M, N, K, threads);
  end = get_time_ms();
  double t_tiled = (end - start) / NTIMES;
  report("OMP+NEON Tiled", t_tiled, t_base, ops, check_result(C_ref, C_test, M, N, K));

  // The SAME pragma on the packed kernel -- shared scratch buffer, DATA RACE
  memset(C_test, 0, bytes_C);
  start = get_time_ms();
  for (int t = 0; t < NTIMES; t++)
    gemm_omp_pack_race(A, B, C_test, M, N, K, threads);
  end = get_time_ms();
  double t_race = (end - start) / NTIMES;
  report("OMP+Pack Race", t_race, t_base, ops, check_result(C_ref, C_test, M, N, K));

  // Same kernel, buffer moved inside the parallel region -- one line, fixed
  memset(C_test, 0, bytes_C);
  start = get_time_ms();
  for (int t = 0; t < NTIMES; t++) gemm_omp_pack(A, B, C_test, M, N, K, threads);
  end = get_time_ms();
  double t_pack = (end - start) / NTIMES;
  report("OMP+Pack", t_pack, t_base, ops, check_result(C_ref, C_test, M, N, K));

  // [EXTENSION LAB] OpenMP x 8x8 double-packed micro-kernel
  memset(C_test, 0, bytes_C);
  start = get_time_ms();
  for (int t = 0; t < NTIMES; t++)
    gemm_omp_pack_8x8(A, B, C_test, M, N, K, threads);
  end = get_time_ms();
  double t_8x8 = (end - start) / NTIMES;
  report("OMP+NEON 8x8", t_8x8, t_base, ops, check_result(C_ref, C_test, M, N, K));

  printf("--------------------------------------------------------------\n");

  free(A);
  free(B);
  free(C_ref);
  free(C_test);
  return 0;
}

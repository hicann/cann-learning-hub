#include "spmv.hpp"

#include <chrono>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace spmv {
namespace {

void spmv_parallel_rows(const CSRMatrix& matrix, const std::vector<float>& x,
                        std::vector<float>* y, BackendTimings* timings) {
    if (y == nullptr) return;
    y->assign(static_cast<std::size_t>(matrix.rows), 0.0f);
#if defined(_OPENMP)
    if (std::getenv("OMP_NUM_THREADS") == nullptr) omp_set_num_threads(16);
    if (std::getenv("OMP_SCHEDULE") == nullptr) omp_set_schedule(omp_sched_static, 0);
#pragma omp parallel
    {
#pragma omp single
        {
            omp_sched_t kind;
            int chunk = 0;
            omp_get_schedule(&kind, &chunk);
            if (timings != nullptr) {
                timings->actual_threads = omp_get_num_threads();
                timings->schedule_kind = static_cast<std::int32_t>(kind);
                timings->schedule_chunk = chunk;
            }
        }
#pragma omp for schedule(runtime)
#endif
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        (*y)[static_cast<std::size_t>(row)] =
            course_sparse::csr_row_dot(matrix, x, row);
    }
#if defined(_OPENMP)
    }
#endif
}

}  // namespace

std::string CpuOpenMpBackend::name() const { return "CPU OpenMP"; }

bool CpuOpenMpBackend::prepare(const CSRMatrix& matrix, std::string* error) {
    matrix_ = matrix;
    return matrix_.validate(error);
}

bool CpuOpenMpBackend::run(const std::vector<float>& x, std::vector<float>* y,
                           BackendTimings* timings, std::string* error) {
    if (!validate_spmv_run(matrix_, x, y, error)) return false;
    const auto start = std::chrono::steady_clock::now();
    spmv_parallel_rows(matrix_, x, y, timings);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    set_host_compute_timing(elapsed, timings);
    return true;
}

}  // namespace spmv

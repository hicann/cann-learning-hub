#include "spmv.hpp"

#include <chrono>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace spmv {

namespace {

void spmv_parallel_rows(const CSRMatrix& matrix, const std::vector<float>& x, std::vector<float>* y) {
    if (y == nullptr) {
        return;
    }
    y->assign(static_cast<std::size_t>(matrix.rows), 0.0f);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        (*y)[static_cast<std::size_t>(row)] =
            course_sparse::csr_row_dot(matrix, x, row);
    }
}

}  // namespace

std::string CpuOpenMp16Backend::name() const {
    return "CPU OpenMP16";
}

bool CpuOpenMp16Backend::prepare(const CSRMatrix& matrix, std::string* error) {
    matrix_ = matrix;
    return matrix_.validate(error);
}

bool CpuOpenMp16Backend::run(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error) {
    if (!validate_spmv_run(matrix_, x, y, error)) return false;

#if defined(_OPENMP)
    omp_set_num_threads(16);
#endif

    const auto start = std::chrono::steady_clock::now();
    spmv_parallel_rows(matrix_, x, y);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    set_host_compute_timing(elapsed, timings);
    return true;
}

}  // namespace spmv

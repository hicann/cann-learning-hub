#include "spmv_backend.hpp"

#include <chrono>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ascend_gmres {

void csr_spmv_serial(const CSRMatrix& matrix, const std::vector<float>& x, std::vector<float>* y) {
    course_sparse::spmv_csr_reference(matrix, x, y);
}

void csr_spmv_parallel16(const CSRMatrix& matrix, const std::vector<float>& x, std::vector<float>* y) {
    y->assign(static_cast<std::size_t>(matrix.rows), 0.0f);
#if defined(_OPENMP)
    omp_set_num_threads(16);
#pragma omp parallel for schedule(static)
#endif
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        (*y)[static_cast<std::size_t>(row)] =
            course_sparse::csr_row_dot(matrix, x, row);
    }
}

CpuSpmvBackend::CpuSpmvBackend(bool parallel) : parallel_(parallel) {}

std::string CpuSpmvBackend::name() const {
    return parallel_ ? "CPU OpenMP16 SpMV" : "CPU single SpMV";
}

bool CpuSpmvBackend::prepare(const CSRMatrix& matrix, std::string* error) {
    matrix_ = matrix;
    stats_.fp32_csr_bytes = matrix_.row_ptr.size() * sizeof(std::int32_t) + matrix_.col_idx.size() * sizeof(std::int32_t) + matrix_.values.size() * sizeof(float);
    return matrix_.validate(error);
}

bool CpuSpmvBackend::multiply(const std::vector<float>& x, std::vector<float>* y, std::string* error) {
    if (y == nullptr) {
        if (error) *error = "output vector is null";
        return false;
    }
    if (x.size() != static_cast<std::size_t>(matrix_.cols)) {
        if (error) *error = "SpMV input size mismatch";
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    if (parallel_) {
        csr_spmv_parallel16(matrix_, x, y);
    } else {
        csr_spmv_serial(matrix_, x, y);
    }
    const auto end = std::chrono::steady_clock::now();
    stats_.warm_kernel_ms += std::chrono::duration<double, std::milli>(end - start).count();
    return true;
}

}  // namespace ascend_gmres

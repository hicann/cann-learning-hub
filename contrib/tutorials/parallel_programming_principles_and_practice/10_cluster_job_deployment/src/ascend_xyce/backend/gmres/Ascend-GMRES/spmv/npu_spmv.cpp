// LEGACY HOST/OpenMP PROTOTYPE - NOT the current NPU backend.
// This file implements HostPrototypePersistentBf16SpmvBackend, a CPU/OpenMP
// bf16 prototype that only exists for reference. The active AscendDevice
// path is the Device GMRES in dis_gmres (Ascend C RTC); this prototype must
// never be reported as NPU compute.
#include "spmv_backend.hpp"

#include <chrono>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ascend_gmres {

std::uint16_t float32_to_bf16_bits(float value) {
    return course_sparse::float32_to_bf16_bits(value);
}

float bf16_bits_to_float32(std::uint16_t value) {
    return course_sparse::bf16_bits_to_float32(value);
}

std::string HostPrototypePersistentBf16SpmvBackend::name() const {
    return "HostPrototype Persistent BF16 SpMV";
}

bool HostPrototypePersistentBf16SpmvBackend::prepare(const CSRMatrix& matrix, std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    if (!matrix.validate(error)) {
        return false;
    }
    matrix_ = matrix;
    values_bf16_.resize(matrix_.values.size());
    for (std::size_t i = 0; i < matrix_.values.size(); ++i) {
        values_bf16_[i] = float32_to_bf16_bits(matrix_.values[i]);
    }
    stats_.fp32_csr_bytes = matrix_.row_ptr.size() * sizeof(std::int32_t) + matrix_.col_idx.size() * sizeof(std::int32_t) + matrix_.values.size() * sizeof(float);
    stats_.bf16_csr_bytes = matrix_.row_ptr.size() * sizeof(std::int32_t) + matrix_.col_idx.size() * sizeof(std::int32_t) + values_bf16_.size() * sizeof(std::uint16_t);
    stats_.compression_ratio = stats_.bf16_csr_bytes == 0 ? 0.0 : static_cast<double>(stats_.fp32_csr_bytes) / static_cast<double>(stats_.bf16_csr_bytes);
    const auto end = std::chrono::steady_clock::now();
    stats_.cold_start_ms = std::chrono::duration<double, std::milli>(end - start).count();
    initialized_ = true;
    return true;
}

bool HostPrototypePersistentBf16SpmvBackend::multiply(const std::vector<float>& x, std::vector<float>* y, std::string* error) {
    if (!initialized_) {
        if (error) *error = "Host prototype SpMV backend is not initialized";
        return false;
    }
    if (y == nullptr) {
        if (error) *error = "output vector is null";
        return false;
    }
    if (x.size() != static_cast<std::size_t>(matrix_.cols)) {
        if (error) *error = "SpMV input size mismatch";
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    y->assign(static_cast<std::size_t>(matrix_.rows), 0.0f);
#if defined(_OPENMP)
    omp_set_num_threads(16);
#pragma omp parallel for schedule(static)
#endif
    for (std::int32_t row = 0; row < matrix_.rows; ++row) {
        (*y)[static_cast<std::size_t>(row)] = course_sparse::reduce_csr_row(
            matrix_.row_ptr, row, [&](std::size_t index) {
                return bf16_bits_to_float32(values_bf16_[index]) *
                       x[static_cast<std::size_t>(matrix_.col_idx[index])];
            });
    }
    const auto end = std::chrono::steady_clock::now();
    stats_.warm_kernel_ms += std::chrono::duration<double, std::milli>(end - start).count();
    return true;
}

}  // namespace ascend_gmres

#include "spmv_bf16.hpp"
#include "spmv_fp16.hpp"

#include <chrono>

namespace spmv {

void spmv_partitioned_bf16(const CSRMatrixBF16& matrix,
                           const std::vector<std::int32_t>& partitions,
                           const std::vector<std::uint16_t>& x_bf16,
                           std::vector<float>* y_fp32,
                           bool parallel) {
    if (y_fp32 == nullptr) return;
    y_fp32->assign(static_cast<std::size_t>(matrix.rows), 0.0f);
    const auto process_rows = [&](std::int32_t begin_row, std::int32_t end_row) {
        for (std::int32_t row = begin_row; row < end_row; ++row) {
            (*y_fp32)[static_cast<std::size_t>(row)] = course_sparse::reduce_csr_row(
                matrix.row_ptr, row, [&](std::size_t index) {
                    const auto column = static_cast<std::size_t>(matrix.col_idx[index]);
                    return bf16_bits_to_float32(matrix.values_bf16[index]) *
                           bf16_bits_to_float32(x_bf16[column]);
                });
        }
    };
    if (partitions.size() < 2) {
        process_rows(0, matrix.rows);
        return;
    }
    const auto partition_count = static_cast<std::int32_t>(partitions.size() - 1);
#if SPMV_HAS_OPENMP
#pragma omp parallel for if(parallel) schedule(static)
#else
    (void)parallel;
#endif
    for (std::int32_t partition = 0; partition < partition_count; ++partition) {
        const auto index = static_cast<std::size_t>(partition);
        process_rows(partitions[index], partitions[index + 1]);
    }
}

std::int64_t CSRMatrixBF16::nnz() const {
    return static_cast<std::int64_t>(values_bf16.size());
}

std::size_t CSRMatrixBF16::bf16_bytes() const {
    return row_ptr.size() * sizeof(std::int32_t) + col_idx.size() * sizeof(std::int32_t) + values_bf16.size() * sizeof(std::uint16_t);
}

std::uint16_t float32_to_bf16_bits(float value) {
    return course_sparse::float32_to_bf16_bits(value);
}

float bf16_bits_to_float32(std::uint16_t value) {
    return course_sparse::bf16_bits_to_float32(value);
}

CSRMatrixBF16 convert_csr_values_to_bf16(const CSRMatrix& matrix) {
    CSRMatrixBF16 converted;
    converted.rows = matrix.rows;
    converted.cols = matrix.cols;
    converted.row_ptr = matrix.row_ptr;
    converted.col_idx = matrix.col_idx;
    converted.values_bf16.resize(matrix.values.size());
    for (std::size_t index = 0; index < matrix.values.size(); ++index) {
        converted.values_bf16[index] = float32_to_bf16_bits(matrix.values[index]);
    }
    return converted;
}

std::size_t csr_bf16_bytes(const CSRMatrixBF16& matrix) {
    return matrix.bf16_bytes();
}

std::string HostPrototypeBf16Fp32Backend::name() const {
    return "Host BF16-FP32 Prototype";
}

bool HostPrototypeBf16Fp32Backend::prepare(const CSRMatrix& matrix, std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    if (!matrix.validate(error)) {
        return false;
    }

    host_matrix_bf16_ = convert_csr_values_to_bf16(matrix);
    row_partitions_ = build_nnz_aware_partitions(matrix, 32);
    fp32_csr_bytes_ = csr_fp32_bytes(matrix);
    bf16_csr_bytes_ = csr_bf16_bytes(host_matrix_bf16_);

    const auto end = std::chrono::steady_clock::now();
    initialization_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
    return true;
}

bool HostPrototypeBf16Fp32Backend::run(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error) {
    if (!validate_spmv_buffers(host_matrix_bf16_.cols, x.size(), y, error)) return false;

    const auto transfer_in_start = std::chrono::steady_clock::now();
    convert_fp32_vector(x, &host_x_bf16_, float32_to_bf16_bits);
    const auto transfer_in_end = std::chrono::steady_clock::now();
    const auto kernel_start = std::chrono::steady_clock::now();
    spmv_partitioned_bf16(host_matrix_bf16_, row_partitions_, host_x_bf16_, &host_y_fp32_);
    const auto kernel_end = std::chrono::steady_clock::now();
    finish_backend_run(host_y_fp32_, y, initialization_ms_, transfer_in_start,
                       transfer_in_end, kernel_start, kernel_end, timings);
    return true;
}

}  // namespace spmv

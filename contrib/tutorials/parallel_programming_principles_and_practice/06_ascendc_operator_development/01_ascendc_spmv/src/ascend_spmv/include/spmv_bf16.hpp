#pragma once

#include "spmv.hpp"

#include <cstddef>
#include <cstdint>

namespace spmv {

struct CSRMatrixBF16 {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::vector<std::int32_t> row_ptr;
    std::vector<std::int32_t> col_idx;
    std::vector<std::uint16_t> values_bf16;

    std::int64_t nnz() const;
    std::size_t bf16_bytes() const;
};

std::uint16_t float32_to_bf16_bits(float value);
float bf16_bits_to_float32(std::uint16_t value);

CSRMatrixBF16 convert_csr_values_to_bf16(const CSRMatrix& matrix);
std::size_t csr_bf16_bytes(const CSRMatrixBF16& matrix);
void spmv_partitioned_bf16(const CSRMatrixBF16& matrix,
                           const std::vector<std::int32_t>& partitions,
                           const std::vector<std::uint16_t>& x_bf16,
                           std::vector<float>* y_fp32,
                           bool parallel = false);

class HostPrototypeBf16Fp32Backend final : public ISpmvBackend {
public:
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool run(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error = nullptr) override;
    double initialization_ms() const override { return initialization_ms_; }

    std::size_t fp32_csr_bytes() const { return fp32_csr_bytes_; }
    std::size_t bf16_csr_bytes() const { return bf16_csr_bytes_; }
    double compression_ratio() const {
        return bf16_csr_bytes_ == 0 ? 0.0 : static_cast<double>(fp32_csr_bytes_) / static_cast<double>(bf16_csr_bytes_);
    }

private:
    CSRMatrixBF16 host_matrix_bf16_;
    std::vector<std::int32_t> row_partitions_;
    std::vector<std::uint16_t> host_x_bf16_;
    std::vector<float> host_y_fp32_;
    std::size_t fp32_csr_bytes_ = 0;
    std::size_t bf16_csr_bytes_ = 0;
    double initialization_ms_ = 0.0;
};

}  // namespace spmv

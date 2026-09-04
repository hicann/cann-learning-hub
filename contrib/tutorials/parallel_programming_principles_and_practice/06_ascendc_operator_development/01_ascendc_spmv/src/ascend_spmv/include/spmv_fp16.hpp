#pragma once

#include "spmv.hpp"

#include <cstddef>
#include <cstdint>

namespace spmv {

struct CSRMatrixFP16 {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::vector<std::int32_t> row_ptr;
    std::vector<std::int32_t> col_idx;
    std::vector<std::uint16_t> values_fp16;

    std::int64_t nnz() const;
    std::size_t fp16_bytes() const;
};

std::uint16_t float32_to_fp16_bits(float value);
float fp16_bits_to_float32(std::uint16_t value);

CSRMatrixFP16 convert_csr_values_to_fp16(const CSRMatrix& matrix);
std::size_t csr_fp32_bytes(const CSRMatrix& matrix);
std::size_t csr_fp16_bytes(const CSRMatrixFP16& matrix);

class HostPrototypeFp16Fp32Backend final : public ISpmvBackend {
public:
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool run(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error = nullptr) override;
    double initialization_ms() const override { return initialization_ms_; }

    std::size_t fp32_csr_bytes() const { return fp32_csr_bytes_; }
    std::size_t fp16_csr_bytes() const { return fp16_csr_bytes_; }
    double compression_ratio() const {
        return fp32_csr_bytes_ == 0 ? 0.0 : static_cast<double>(fp16_csr_bytes_) / static_cast<double>(fp32_csr_bytes_);
    }

private:
    CSRMatrixFP16 host_matrix_fp16_;
    std::vector<std::int32_t> row_partitions_;
    std::vector<std::uint16_t> host_x_fp16_;
    std::vector<float> host_y_fp32_;
    std::size_t fp32_csr_bytes_ = 0;
    std::size_t fp16_csr_bytes_ = 0;
    double initialization_ms_ = 0.0;
};

}  // namespace spmv

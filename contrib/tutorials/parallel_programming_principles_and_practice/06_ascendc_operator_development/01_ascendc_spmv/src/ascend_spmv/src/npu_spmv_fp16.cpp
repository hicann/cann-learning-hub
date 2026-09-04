#include "spmv_fp16.hpp"

#include <algorithm>
#include <chrono>

namespace spmv {

namespace {

float fp16_row_dot(const CSRMatrixFP16& matrix,
                   const std::vector<std::uint16_t>& x, std::int32_t row) {
    return course_sparse::reduce_csr_row(matrix.row_ptr, row, [&](std::size_t index) {
        const float value = fp16_bits_to_float32(matrix.values_fp16[index]);
        const auto column = static_cast<std::size_t>(matrix.col_idx[index]);
        return value * fp16_bits_to_float32(x[column]);
    });
}

void spmv_partitioned_fp16(const CSRMatrixFP16& matrix,
                           const std::vector<std::int32_t>& partitions,
                           const std::vector<std::uint16_t>& x_fp16,
                           std::vector<float>* y_fp32) {
    if (y_fp32 == nullptr) return;
    y_fp32->assign(static_cast<std::size_t>(matrix.rows), 0.0f);
    const std::vector<std::int32_t> fallback{0, matrix.rows};
    const auto& bounds = partitions.size() < 2 ? fallback : partitions;
    for (std::size_t partition = 0; partition + 1 < bounds.size(); ++partition) {
        const std::int32_t begin_row = bounds[partition];
        const std::int32_t end_row = bounds[partition + 1];
        for (std::int32_t row = begin_row; row < end_row; ++row) {
            (*y_fp32)[static_cast<std::size_t>(row)] = fp16_row_dot(matrix, x_fp16, row);
        }
    }
}

}  // namespace

std::int64_t CSRMatrixFP16::nnz() const {
    return static_cast<std::int64_t>(values_fp16.size());
}

std::size_t CSRMatrixFP16::fp16_bytes() const {
    return row_ptr.size() * sizeof(std::int32_t) + col_idx.size() * sizeof(std::int32_t) + values_fp16.size() * sizeof(std::uint16_t);
}

std::uint16_t float32_to_fp16_bits(float value) {
    std::uint32_t bits = 0;
    std::copy_n(reinterpret_cast<const unsigned char*>(&value), sizeof(value),
                reinterpret_cast<unsigned char*>(&bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::uint32_t exponent = (bits >> 23) & 0xffu;
    std::uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 255u) {
        if (mantissa != 0u) {
            return static_cast<std::uint16_t>(sign | 0x7e00u);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    int adjusted_exponent = static_cast<int>(exponent) - 127 + 15;
    if (adjusted_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (adjusted_exponent <= 0) {
        if (adjusted_exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - adjusted_exponent);
        std::uint16_t half_mantissa = static_cast<std::uint16_t>(mantissa >> shift);
        if ((mantissa >> (shift - 1)) & 1u) {
            ++half_mantissa;
        }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint16_t half_exponent = static_cast<std::uint16_t>(adjusted_exponent << 10);
    std::uint16_t half_mantissa = static_cast<std::uint16_t>(mantissa >> 13);
    if (mantissa & 0x1000u) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 0x7c00u) {
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
        }
    }
    return static_cast<std::uint16_t>(sign | half_exponent | (half_mantissa & 0x3ffu));
}

float fp16_bits_to_float32(std::uint16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000u)) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x3ffu;

    std::uint32_t bits = 0;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            exponent = 1u;
            while ((mantissa & 0x400u) == 0u) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffu;
            const std::uint32_t adjusted_exponent = exponent + (127u - 15u);
            bits = sign | (adjusted_exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        const std::uint32_t adjusted_exponent = exponent + (127u - 15u);
        bits = sign | (adjusted_exponent << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::copy_n(reinterpret_cast<const unsigned char*>(&bits), sizeof(bits),
                reinterpret_cast<unsigned char*>(&result));
    return result;
}

CSRMatrixFP16 convert_csr_values_to_fp16(const CSRMatrix& matrix) {
    CSRMatrixFP16 converted;
    converted.rows = matrix.rows;
    converted.cols = matrix.cols;
    converted.row_ptr = matrix.row_ptr;
    converted.col_idx = matrix.col_idx;
    converted.values_fp16.resize(matrix.values.size());
    for (std::size_t index = 0; index < matrix.values.size(); ++index) {
        converted.values_fp16[index] = float32_to_fp16_bits(matrix.values[index]);
    }
    return converted;
}

std::size_t csr_fp32_bytes(const CSRMatrix& matrix) {
    return matrix.row_ptr.size() * sizeof(std::int32_t) + matrix.col_idx.size() * sizeof(std::int32_t) + matrix.values.size() * sizeof(float);
}

std::size_t csr_fp16_bytes(const CSRMatrixFP16& matrix) {
    return matrix.fp16_bytes();
}

std::string HostPrototypeFp16Fp32Backend::name() const {
    return "Host FP16-FP32 Prototype";
}

bool HostPrototypeFp16Fp32Backend::prepare(const CSRMatrix& matrix, std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    if (!matrix.validate(error)) {
        return false;
    }

    host_matrix_fp16_ = convert_csr_values_to_fp16(matrix);
    row_partitions_ = build_nnz_aware_partitions(matrix, 32);
    fp32_csr_bytes_ = csr_fp32_bytes(matrix);
    fp16_csr_bytes_ = csr_fp16_bytes(host_matrix_fp16_);

    const auto end = std::chrono::steady_clock::now();
    initialization_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
    return true;
}

bool HostPrototypeFp16Fp32Backend::run(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error) {
    if (!validate_spmv_buffers(host_matrix_fp16_.cols, x.size(), y, error)) return false;

    const auto transfer_in_start = std::chrono::steady_clock::now();
    convert_fp32_vector(x, &host_x_fp16_, float32_to_fp16_bits);
    const auto transfer_in_end = std::chrono::steady_clock::now();

    const auto kernel_start = std::chrono::steady_clock::now();
    spmv_partitioned_fp16(host_matrix_fp16_, row_partitions_, host_x_fp16_, &host_y_fp32_);
    const auto kernel_end = std::chrono::steady_clock::now();
    finish_backend_run(host_y_fp32_, y, initialization_ms_, transfer_in_start,
                       transfer_in_end, kernel_start, kernel_end, timings);
    return true;
}

}  // namespace spmv

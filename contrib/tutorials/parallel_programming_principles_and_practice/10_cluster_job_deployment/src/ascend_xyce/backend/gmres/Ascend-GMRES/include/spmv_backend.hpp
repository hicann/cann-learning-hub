#pragma once

#include "csr_matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ascend_gmres {

struct SpmvStats {
    double cold_start_ms = 0.0;
    double warm_kernel_ms = 0.0;
    std::size_t fp32_csr_bytes = 0;
    std::size_t bf16_csr_bytes = 0;
    double compression_ratio = 0.0;
};

class ISpmvBackend {
public:
    virtual ~ISpmvBackend() = default;
    virtual std::string name() const = 0;
    virtual bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) = 0;
    virtual bool multiply(const std::vector<float>& x, std::vector<float>* y, std::string* error = nullptr) = 0;
    virtual const SpmvStats& stats() const = 0;
};

class CpuSpmvBackend final : public ISpmvBackend {
public:
    explicit CpuSpmvBackend(bool parallel = false);
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool multiply(const std::vector<float>& x, std::vector<float>* y, std::string* error = nullptr) override;
    const SpmvStats& stats() const override { return stats_; }

private:
    CSRMatrix matrix_;
    SpmvStats stats_;
    bool parallel_ = false;
};

class HostPrototypePersistentBf16SpmvBackend final : public ISpmvBackend {
public:
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool multiply(const std::vector<float>& x, std::vector<float>* y, std::string* error = nullptr) override;
    const SpmvStats& stats() const override { return stats_; }

private:
    CSRMatrix matrix_;
    std::vector<std::uint16_t> values_bf16_;
    SpmvStats stats_;
    bool initialized_ = false;
};

void csr_spmv_serial(const CSRMatrix& matrix, const std::vector<float>& x, std::vector<float>* y);
void csr_spmv_parallel16(const CSRMatrix& matrix, const std::vector<float>& x, std::vector<float>* y);

std::uint16_t float32_to_bf16_bits(float value);
float bf16_bits_to_float32(std::uint16_t value);

}  // namespace ascend_gmres

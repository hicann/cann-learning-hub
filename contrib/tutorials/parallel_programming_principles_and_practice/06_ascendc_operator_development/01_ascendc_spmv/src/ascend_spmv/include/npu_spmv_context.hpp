#pragma once

#include "spmv_bf16.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spmv {

struct HostPrototypeSpmvContextStats {
    std::size_t fp32_csr_bytes = 0;
    std::size_t bf16_csr_bytes = 0;
    double compression_ratio = 0.0;
    double initialization_ms = 0.0;
    double nnz_balance_ratio = 0.0;
    std::int32_t partition_count = 0;
    std::int64_t max_partition_nnz = 0;
    double avg_partition_nnz = 0.0;
};

std::int32_t host_prototype_partition_count();

class HostPrototypeSpmvContext {
public:
    bool initialize_bf16(const CSRMatrix& matrix, std::int32_t partition_hint, std::string* error = nullptr);
    bool initialize_bf16_autotuned(const CSRMatrix& matrix, const std::vector<std::int32_t>& partition_candidates, std::string* error = nullptr);
    bool run_bf16(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error = nullptr);

    bool initialized() const { return initialized_; }
    const CSRMatrixBF16& host_matrix_bf16() const { return host_matrix_bf16_; }
    const std::vector<std::int32_t>& row_partitions() const { return row_partitions_; }
    const HostPrototypeSpmvContextStats& stats() const { return stats_; }

private:
    CSRMatrixBF16 host_matrix_bf16_;
    std::vector<std::int32_t> row_partitions_;
    std::vector<std::uint16_t> host_x_bf16_;
    std::vector<float> host_y_fp32_;
    HostPrototypeSpmvContextStats stats_;
    bool initialized_ = false;
};

struct HostPrototypePersistentHandle {
    HostPrototypeSpmvContext context;
    std::int32_t ai_core_count = 0;
    bool ready = false;
};

class HostPrototypeBf16PersistentBackend final : public ISpmvBackend {
public:
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool run(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error = nullptr) override;
    double initialization_ms() const override { return handle_.context.stats().initialization_ms; }

    std::size_t fp32_csr_bytes() const { return handle_.context.stats().fp32_csr_bytes; }
    std::size_t bf16_csr_bytes() const { return handle_.context.stats().bf16_csr_bytes; }
    double compression_ratio() const { return handle_.context.stats().compression_ratio; }
    double nnz_balance_ratio() const { return handle_.context.stats().nnz_balance_ratio; }
    std::int32_t partition_count() const { return handle_.context.stats().partition_count; }
    std::int64_t max_partition_nnz() const { return handle_.context.stats().max_partition_nnz; }
    double avg_partition_nnz() const { return handle_.context.stats().avg_partition_nnz; }
    double cold_start_ms() const { return handle_.context.stats().initialization_ms; }

private:
    HostPrototypePersistentHandle handle_;
};

}  // namespace spmv

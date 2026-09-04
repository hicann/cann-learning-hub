#pragma once

#include <course_sparse/cpu_backend.hpp>
#include <course_sparse/spmv_types.hpp>

namespace spmv {

using course_sparse::BackendTimings;
using course_sparse::CpuSingleBackend;
using course_sparse::ISpmvBackend;
using course_sparse::convert_fp32_vector;
using course_sparse::finish_backend_run;
using course_sparse::set_host_compute_timing;
using course_sparse::validate_spmv_buffers;
using course_sparse::validate_spmv_run;

std::vector<std::int32_t> build_equal_row_partitions(const CSRMatrix& matrix,
                                                     std::int32_t partitions);
std::vector<std::int32_t> build_nnz_aware_partitions(const CSRMatrix& matrix,
                                                     std::int32_t partitions);

class CpuOpenMp16Backend final : public ISpmvBackend {
public:
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool run(const std::vector<float>& x, std::vector<float>* y,
             BackendTimings* timings, std::string* error = nullptr) override;

private:
    CSRMatrix matrix_;
};

class HostPrototypeBackend : public ISpmvBackend {
public:
    HostPrototypeBackend(std::string label, bool nnz_aware);
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool run(const std::vector<float>& x, std::vector<float>* y,
             BackendTimings* timings, std::string* error = nullptr) override;
    double initialization_ms() const override { return initialization_ms_; }

private:
    std::string label_;
    bool nnz_aware_;
    CSRMatrix host_matrix_;
    std::vector<std::int32_t> row_partitions_;
    std::vector<float> host_x_;
    std::vector<float> host_y_;
    double initialization_ms_ = 0.0;
};

class HostPrototypeBaselineBackend final : public HostPrototypeBackend {
public:
    HostPrototypeBaselineBackend();
};

class HostPrototypeOptimizedBackend final : public HostPrototypeBackend {
public:
    HostPrototypeOptimizedBackend();
};

}  // namespace spmv

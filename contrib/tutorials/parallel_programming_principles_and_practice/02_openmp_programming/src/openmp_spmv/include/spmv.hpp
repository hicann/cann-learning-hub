#pragma once

#include <course_sparse/cpu_backend.hpp>
#include <course_sparse/spmv_types.hpp>

namespace spmv {

using course_sparse::BackendTimings;
using course_sparse::CpuSingleBackend;
using course_sparse::ISpmvBackend;
using course_sparse::set_host_compute_timing;
using course_sparse::validate_spmv_run;

class CpuOpenMpBackend final : public ISpmvBackend {
public:
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool run(const std::vector<float>& x, std::vector<float>* y,
             BackendTimings* timings, std::string* error = nullptr) override;

private:
    CSRMatrix matrix_;
};

}  // namespace spmv

#pragma once

#include "blas_backend.hpp"
#include "csr_matrix.hpp"
#include "gmres_profiler.hpp"
#include "spmv_backend.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ascend_gmres {

struct GmresOptions {
    std::int32_t restart = 30;
    std::int32_t max_iterations = 10000;
    float tolerance = 1.0e-6f;
};

struct GmresResult {
    std::string solver_name;
    std::int32_t iterations = 0;
    float final_relative_residual = 0.0f;
    bool converged = false;
    double total_ms = 0.0;
    double cold_start_ms = 0.0;
    double warm_ms = 0.0;
    SpmvStats spmv_stats;
    GMRESProfiler profiler;
};

class GmresSolver {
public:
    GmresSolver(std::string solver_name, std::unique_ptr<ISpmvBackend> spmv, std::unique_ptr<IBlasBackend> blas);

    const std::string& name() const { return solver_name_; }
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr);
    GmresResult solve(const std::vector<float>& b, std::vector<float>* x, const GmresOptions& options, std::string* error = nullptr);

private:
    std::string solver_name_;
    std::unique_ptr<ISpmvBackend> spmv_;
    std::unique_ptr<IBlasBackend> blas_;
};

GmresSolver make_cpu_single_gmres_solver();
GmresSolver make_cpu_openmp16_gmres_solver();
GmresSolver make_host_prototype_gmres_solver();

float compute_relative_residual(ISpmvBackend& spmv, IBlasBackend& blas, const std::vector<float>& b, const std::vector<float>& x);

}  // namespace ascend_gmres

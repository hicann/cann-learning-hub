#pragma once

#include "csr_matrix.hpp"
#include "gmres.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ascend_xyce {

enum class LinearSolverKind {
    CpuSingle,
    CpuOpenMP16,
    AscendDevice,
};

struct XyceSimulationOptions {
    std::int32_t gmres_restart = 30;
    std::int32_t gmres_max_iterations = 10000;
    float gmres_tolerance = 1.0e-6f;
};

// Documented solution-error gate: a solver result is only accepted when the
// relative error against the constructed expected solution (and, when a CPU
// reference solution is supplied, against that reference) stays below this
// threshold. The 1e-3 relative-solution-error gate is the same one the
// dis_gmres benchmark uses for its own exit code.
constexpr double kSolutionErrorTolerance = 1.0e-3;

struct XyceSimulationResult {
    std::string solver_name;
    double total_simulation_ms = 0.0;
    double matrix_assembly_ms = 0.0;
    ascend_gmres::GmresResult linear_result;
    double solution_error_vs_cpu = 0.0;
    double solution_error_vs_expected = 0.0;
};

class XyceSparseMatrixAdapter {
public:
    explicit XyceSparseMatrixAdapter(ascend_gmres::CSRMatrix matrix);
    const ascend_gmres::CSRMatrix& csr() const { return matrix_; }

private:
    ascend_gmres::CSRMatrix matrix_;
};

class XyceLinearSolverAdapter {
public:
    explicit XyceLinearSolverAdapter(LinearSolverKind kind);
    ~XyceLinearSolverAdapter();
    bool prepare(const XyceSparseMatrixAdapter& matrix, std::string* error = nullptr);
    ascend_gmres::GmresResult solve(const std::vector<float>& b, std::vector<float>* x, const XyceSimulationOptions& options, std::string* error = nullptr);
    const std::string& name() const;

private:
    struct DeviceState;
    LinearSolverKind kind_;
    std::unique_ptr<ascend_gmres::GmresSolver> host_solver_;
    std::unique_ptr<DeviceState> device_;
};

class XyceApplicationWrapper {
public:
    XyceApplicationWrapper(XyceSparseMatrixAdapter matrix, std::vector<float> expected_solution);
    XyceSimulationResult run(LinearSolverKind kind, const XyceSimulationOptions& options, const std::vector<float>* cpu_reference_solution = nullptr);

private:
    XyceSparseMatrixAdapter matrix_;
    std::vector<float> expected_solution_;
};

std::string solver_kind_name(LinearSolverKind kind);
ascend_gmres::CSRMatrix load_or_create_csr_matrix(const std::string& matrix_dir, const ascend_gmres::MatrixSpec& spec);
std::vector<ascend_gmres::MatrixSpec> default_xyce_matrix_specs();

}  // namespace ascend_xyce

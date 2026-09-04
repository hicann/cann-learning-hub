#include "xyce_adapter.hpp"

#include "spmv_backend.hpp"
#include "include/gmres.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace ascend_xyce {

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::unique_ptr<ascend_gmres::GmresSolver> make_solver(LinearSolverKind kind) {
    switch (kind) {
        case LinearSolverKind::CpuSingle:
            return std::make_unique<ascend_gmres::GmresSolver>(ascend_gmres::make_cpu_single_gmres_solver());
        case LinearSolverKind::CpuOpenMP16:
            return std::make_unique<ascend_gmres::GmresSolver>(ascend_gmres::make_cpu_openmp16_gmres_solver());
        case LinearSolverKind::AscendDevice:
            return nullptr;
    }
    return nullptr;
}

ascend_gmres::CSRMatrix generate_matrix_for_spec(const ascend_gmres::MatrixSpec& spec) {
    if (!spec.name.empty() && spec.name[0] == 'U') {
        return ascend_gmres::generate_uniform_matrix(spec, 42);
    }
    if (!spec.name.empty() && spec.name[0] == 'L') {
        return ascend_gmres::generate_long_tail_matrix(spec, 42);
    }
    return ascend_gmres::generate_block_matrix(spec, 32, 42);
}

}  // namespace

std::string solver_kind_name(LinearSolverKind kind) {
    switch (kind) {
        case LinearSolverKind::CpuSingle:
            return "Xyce CPU single GMRES";
        case LinearSolverKind::CpuOpenMP16:
            return "Xyce CPU OpenMP16 GMRES";
        case LinearSolverKind::AscendDevice:
            return "Xyce Ascend Device GMRES (Ascend C RTC)";
    }
    return "Xyce unknown solver";
}

XyceSparseMatrixAdapter::XyceSparseMatrixAdapter(ascend_gmres::CSRMatrix matrix) : matrix_(std::move(matrix)) {}

struct XyceLinearSolverAdapter::DeviceState {
    dis_gmres::CSRMatrix matrix;
    std::vector<dis_gmres::RowPartition> partitions;
    dis_gmres::HcclCommunicator communicator;
};

XyceLinearSolverAdapter::XyceLinearSolverAdapter(LinearSolverKind kind)
    : kind_(kind), host_solver_(make_solver(kind)) {
    if (kind_ == LinearSolverKind::AscendDevice) device_ = std::make_unique<DeviceState>();
}

XyceLinearSolverAdapter::~XyceLinearSolverAdapter() = default;

bool XyceLinearSolverAdapter::prepare(const XyceSparseMatrixAdapter& matrix, std::string* error) {
    if (host_solver_) return host_solver_->prepare(matrix.csr(), error);
#if !DIS_GMRES_HAS_CANN
    if (error) *error = "Ascend Device solver is unavailable in ASCEND_XYCE_HOST_ONLY mode";
    return false;
#else
    const auto& source = matrix.csr();
    device_->matrix.rows=source.rows;device_->matrix.cols=source.cols;
    device_->matrix.row_ptr=source.row_ptr;device_->matrix.col_idx=source.col_idx;device_->matrix.values=source.values;
    device_->partitions=dis_gmres::partition_rows(device_->matrix,1,false);
    const int device_id = std::getenv("DEVICE_ID") ? std::stoi(std::getenv("DEVICE_ID")) : 0;
    return device_->communicator.initialize(device_id,0,1,"",error);
#endif
}

ascend_gmres::GmresResult XyceLinearSolverAdapter::solve(const std::vector<float>& b, std::vector<float>* x, const XyceSimulationOptions& options, std::string* error) {
    ascend_gmres::GmresOptions gmres_options;
    gmres_options.restart = options.gmres_restart;
    gmres_options.max_iterations = options.gmres_max_iterations;
    gmres_options.tolerance = options.gmres_tolerance;
    if (host_solver_) {
        auto result = host_solver_->solve(b, x, gmres_options, error);
        result.solver_name = solver_kind_name(kind_);
        return result;
    }
    dis_gmres::GmresOptions device_options;
    device_options.restart=options.gmres_restart;device_options.max_iterations=options.gmres_max_iterations;
    device_options.tolerance=options.gmres_tolerance;device_options.zero_initial_guess=true;
    device_options.communication_avoiding=false;
    auto device_result=dis_gmres::distributed_gmres(device_->matrix,device_->partitions,b,x,&device_->communicator,device_options,error);
    ascend_gmres::GmresResult result;result.solver_name=solver_kind_name(kind_);result.iterations=device_result.iterations;
    result.final_relative_residual=device_result.residual;result.converged=device_result.converged;result.total_ms=device_result.profile.total_ms;
    result.profiler.spmv_ms=device_result.profile.spmv_ms;result.profiler.dot_ms=device_result.profile.dot_ms;
    result.profiler.axpy_ms=device_result.profile.axpy_ms;result.profiler.norm_ms=device_result.profile.norm_ms;
    result.profiler.givens_ms=device_result.profile.givens_ms;result.profiler.device_transfer_ms=device_result.profile.transfer_ms;
    result.profiler.communication_ms=device_result.profile.communication_ms;result.profiler.kernel_launch_ms=device_result.profile.kernel_launch_ms;
    result.profiler.synchronization_ms=device_result.profile.synchronization_ms;result.profiler.other_ms=std::max(0.0,result.total_ms-result.profiler.accounted_ms());
    return result;
}

const std::string& XyceLinearSolverAdapter::name() const {
    static const std::string cpu_single = solver_kind_name(LinearSolverKind::CpuSingle);
    static const std::string cpu_openmp = solver_kind_name(LinearSolverKind::CpuOpenMP16);
    static const std::string ascend = solver_kind_name(LinearSolverKind::AscendDevice);
    switch (kind_) {
        case LinearSolverKind::CpuSingle:
            return cpu_single;
        case LinearSolverKind::CpuOpenMP16:
            return cpu_openmp;
        case LinearSolverKind::AscendDevice:
            return ascend;
    }
    return cpu_single;
}

XyceApplicationWrapper::XyceApplicationWrapper(XyceSparseMatrixAdapter matrix, std::vector<float> expected_solution)
    : matrix_(std::move(matrix)), expected_solution_(std::move(expected_solution)) {}

XyceSimulationResult XyceApplicationWrapper::run(LinearSolverKind kind, const XyceSimulationOptions& options, const std::vector<float>* cpu_reference_solution) {
    XyceSimulationResult simulation;
    simulation.solver_name = solver_kind_name(kind);

    const auto total_start = Clock::now();

    const auto assembly_start = Clock::now();
    std::vector<float> b;
    ascend_gmres::csr_spmv_serial(matrix_.csr(), expected_solution_, &b);
    const auto assembly_end = Clock::now();
    simulation.matrix_assembly_ms = elapsed_ms(assembly_start, assembly_end);

    XyceLinearSolverAdapter solver(kind);
    std::string error;
    if (!solver.prepare(matrix_, &error)) {
        throw std::runtime_error(solver.name() + " prepare failed: " + error);
    }

    std::vector<float> x(static_cast<std::size_t>(matrix_.csr().cols), 0.0f);
    simulation.linear_result = solver.solve(b, &x, options, &error);
    if (!simulation.linear_result.converged) {
        throw std::runtime_error(solver.name() + " did not converge, residual=" + std::to_string(simulation.linear_result.final_relative_residual));
    }

    if (cpu_reference_solution != nullptr) {
        simulation.solution_error_vs_cpu = ascend_gmres::relative_error(*cpu_reference_solution, x);
    }
    // Solution-error gate: the constructed expected_solution is always checked
    // (so a wrong CPU reference can never mask a bad solve), and the CPU
    // reference is checked when supplied. Exceeding the documented threshold
    // throws, so the benchmark exits nonzero and never writes a pass row.
    simulation.solution_error_vs_expected = ascend_gmres::relative_error(expected_solution_, x);
    if (simulation.solution_error_vs_expected > kSolutionErrorTolerance) {
        throw std::runtime_error(solver.name() + " solution error vs expected exceeded " +
                                 std::to_string(kSolutionErrorTolerance) + ": " +
                                 std::to_string(simulation.solution_error_vs_expected));
    }
    if (cpu_reference_solution != nullptr && simulation.solution_error_vs_cpu > kSolutionErrorTolerance) {
        throw std::runtime_error(solver.name() + " solution error vs CPU reference exceeded " +
                                 std::to_string(kSolutionErrorTolerance) + ": " +
                                 std::to_string(simulation.solution_error_vs_cpu));
    }

    const auto total_end = Clock::now();
    simulation.total_simulation_ms = elapsed_ms(total_start, total_end);
    return simulation;
}

std::vector<ascend_gmres::MatrixSpec> default_xyce_matrix_specs() {
    return ascend_gmres::default_matrix_specs();
}

ascend_gmres::CSRMatrix load_or_create_csr_matrix(const std::string& matrix_dir, const ascend_gmres::MatrixSpec& spec) {
    fs::create_directories(matrix_dir);
    const fs::path path = fs::path(matrix_dir) / (spec.name + ".csrbin");
    ascend_gmres::CSRMatrix matrix;
    std::string error;
    if (fs::exists(path) && ascend_gmres::CSRMatrix::load_binary(path.string(), &matrix, &error)) {
        return ascend_gmres::make_gmres_ready_matrix(matrix);
    }

    const fs::path ascend_gmres_path = fs::path("..") / "Ascend-GMRES" / "matrices" / (spec.name + ".csrbin");
    if (fs::exists(ascend_gmres_path) && ascend_gmres::CSRMatrix::load_binary(ascend_gmres_path.string(), &matrix, &error)) {
        matrix = ascend_gmres::make_gmres_ready_matrix(matrix);
        matrix.save_binary(path.string(), nullptr);
        return matrix;
    }

    matrix = generate_matrix_for_spec(spec);
    if (!matrix.validate(&error)) {
        throw std::runtime_error("generated matrix validation failed for " + spec.name + ": " + error);
    }
    matrix.save_binary(path.string(), nullptr);
    return matrix;
}

}  // namespace ascend_xyce

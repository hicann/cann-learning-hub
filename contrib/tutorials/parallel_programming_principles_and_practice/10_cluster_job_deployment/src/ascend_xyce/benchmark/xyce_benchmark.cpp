#include "xyce_adapter.hpp"

#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace ascend_xyce {

struct BenchmarkConfig {
    std::int32_t warmup = 0;
    std::int32_t repeat = 10;
    std::string matrix_dir = "matrices";
    std::string results_dir = "results";
    std::string csv_path = "results/xyce_benchmark.csv";
    std::string matrix_filter;
    bool check_matrix_only = false;
};

struct AveragedResult {
    std::string solver_name;
    double total_simulation_ms = 0.0;
    double matrix_assembly_ms = 0.0;
    double solution_error_vs_cpu = 0.0;
    ascend_gmres::GmresResult linear;
};

double per_iteration(double value, std::int32_t iterations) {
    return iterations > 0 ? value / static_cast<double>(iterations) : 0.0;
}

std::string csv_header() {
    return "matrix,rows,cols,nnz,solver,total_simulation_ms,matrix_assembly_ms,linear_solver_ms,iterations,final_residual,converged,speedup_vs_cpu_single,solution_error_vs_cpu,spmv_ms,dot_ms,axpy_ms,norm_ms,givens_ms,residual_ms,other_ms,device_transfer_ms,hccl_ms,kernel_launch_ms,synchronization_ms,avg_spmv_per_iteration,avg_dot_per_iteration,avg_axpy_per_iteration";
}

std::string csv_row(const std::string& matrix_name, const ascend_gmres::CSRMatrix& matrix, const AveragedResult& result, double cpu_single_total_ms) {
    const auto& linear = result.linear;
    const auto& profiler = linear.profiler;
    std::ostringstream out;
    // Timing fields keep fixed 6 decimals; correctness fields (final residual,
    // solution errors) use scientific with 9 significant digits so they stay
    // auditable. Format flags are set per column so one format never leaks
    // into the other columns.
    const auto timing = [&]() { out << std::fixed << std::setprecision(6); };
    const auto precision = [&]() { out << std::scientific << std::setprecision(9); };
    timing();
    out << matrix_name << ','
        << matrix.rows << ','
        << matrix.cols << ','
        << matrix.nnz() << ','
        << result.solver_name << ','
        << result.total_simulation_ms << ','
        << result.matrix_assembly_ms << ','
        << linear.total_ms << ','
        << linear.iterations << ',';
    precision();
    out << linear.final_relative_residual << ',';
    timing();
    out << (linear.converged ? 1 : 0) << ','
        << (result.total_simulation_ms > 0.0 ? cpu_single_total_ms / result.total_simulation_ms : 0.0) << ',';
    precision();
    out << result.solution_error_vs_cpu << ',';
    timing();
    out << profiler.spmv_ms << ','
        << profiler.dot_ms << ','
        << profiler.axpy_ms << ','
        << profiler.norm_ms << ','
        << profiler.givens_ms << ','
        << profiler.residual_ms << ','
        << profiler.other_ms << ','
        << profiler.device_transfer_ms << ','
        << profiler.communication_ms << ','
        << profiler.kernel_launch_ms << ','
        << profiler.synchronization_ms << ','
        << per_iteration(profiler.spmv_ms, linear.iterations) << ','
        << per_iteration(profiler.dot_ms, linear.iterations) << ','
        << per_iteration(profiler.axpy_ms, linear.iterations);
    return out.str();
}

AveragedResult run_solver_repeated(const ascend_gmres::CSRMatrix& matrix,
                                   const std::vector<float>& x_true,
                                   LinearSolverKind kind,
                                   const XyceSimulationOptions& options,
                                   const BenchmarkConfig& config,
                                   const std::vector<float>* cpu_reference_solution) {
    XyceApplicationWrapper app(XyceSparseMatrixAdapter(matrix), x_true);
    // The wrapper is created once outside the repeated loop, and the matrix is
    // loaded/generated before run_solver_repeated is called. Each app.run
    // creates a fresh XyceLinearSolverAdapter; Device prepare initializes the
    // communicator and distributed_gmres rebuilds the solver-local RTC/Device
    // state. So each run is a fresh solver adapter/prepare/Device state, not a
    // full reload of the matrix from disk and not a whole-process cold start;
    // --warmup runs are discarded extra solver invocations and are never
    // reported as "warm" metrics.
    for (std::int32_t i = 0; i < config.warmup; ++i) {
        (void)app.run(kind, options, cpu_reference_solution);
    }

    AveragedResult averaged;
    averaged.solver_name = solver_kind_name(kind);
    ascend_gmres::GMRESProfiler profiler_sum;
    double residual_sum = 0.0;
    std::int32_t iteration_sum = 0;
    bool converged = true;

    for (std::int32_t i = 0; i < config.repeat; ++i) {
        const auto result = app.run(kind, options, cpu_reference_solution);
        averaged.total_simulation_ms += result.total_simulation_ms;
        averaged.matrix_assembly_ms += result.matrix_assembly_ms;
        averaged.linear.total_ms += result.linear_result.total_ms;
        averaged.solution_error_vs_cpu += result.solution_error_vs_cpu;
        profiler_sum.accumulate(result.linear_result.profiler);
        residual_sum += result.linear_result.final_relative_residual;
        iteration_sum += result.linear_result.iterations;
        converged = converged && result.linear_result.converged;
    }

    const double scale = 1.0 / static_cast<double>(config.repeat);
    averaged.total_simulation_ms *= scale;
    averaged.matrix_assembly_ms *= scale;
    averaged.linear.total_ms *= scale;
    averaged.solution_error_vs_cpu *= scale;
    profiler_sum.scale(scale);
    averaged.linear.profiler = profiler_sum;
    averaged.linear.final_relative_residual = static_cast<float>(residual_sum * scale);
    averaged.linear.iterations = iteration_sum / config.repeat;
    averaged.linear.converged = converged;
    averaged.linear.solver_name = averaged.solver_name;
    return averaged;
}

void print_solver(const AveragedResult& result, double cpu_single_ms) {
    const auto& linear = result.linear;
    std::cout << "  " << result.solver_name
              << " | sim_ms=" << result.total_simulation_ms
              << " | linear_ms=" << linear.total_ms
              << " | iter=" << linear.iterations
              << " | residual=" << std::scientific << std::setprecision(9) << linear.final_relative_residual
              << " | speedup=" << std::fixed << (result.total_simulation_ms > 0.0 ? cpu_single_ms / result.total_simulation_ms : 0.0)
              << " | solution_error_vs_cpu=" << std::scientific << result.solution_error_vs_cpu << std::fixed << '\n';
    ascend_gmres::print_profiler_breakdown(std::cout, result.solver_name, linear.profiler, linear.total_ms, linear.iterations);
}

BenchmarkConfig parse_config(int argc, char** argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool known = arg == "--warmup" || arg == "--repeat" ||
                           arg == "--matrix-dir" || arg == "--results-dir" ||
                           arg == "--csv" || arg == "--matrix" ||
                           arg == "--check-matrix";
        if (!known) throw std::runtime_error("unknown argument: " + arg);
        if (i + 1 >= argc) throw std::runtime_error(arg + " requires a value");
        const std::string value = argv[++i];
        if (arg == "--warmup") { config.warmup = std::stoi(value); continue; }
        if (arg == "--repeat") { config.repeat = std::stoi(value); continue; }
        if (arg == "--matrix-dir") { config.matrix_dir = value; continue; }
        if (arg == "--results-dir") { config.results_dir = value; continue; }
        if (arg == "--csv") { config.csv_path = value; continue; }
        if (arg == "--matrix") { config.matrix_filter = value; continue; }
        if (arg == "--check-matrix") {
            config.matrix_filter = value;
            config.check_matrix_only = true;
            continue;
        }
    }
    if (config.warmup < 0 || config.repeat <= 0) {
        throw std::runtime_error("warmup must be >= 0 and repeat must be > 0");
    }
    return config;
}

bool filter_matches(const BenchmarkConfig& config,
                    const std::vector<ascend_gmres::MatrixSpec>& specs) {
    return config.matrix_filter.empty() ||
           std::any_of(specs.begin(), specs.end(), [&](const auto& spec) {
               return spec.name == config.matrix_filter;
           });
}

int validate_selection(const BenchmarkConfig& config,
                       const std::vector<ascend_gmres::MatrixSpec>& specs,
                       bool* proceed) {
    const bool matched = !config.matrix_filter.empty() && filter_matches(config, specs);
    if (config.check_matrix_only) {
        std::cout << (matched ? "matrix ok: " : "unknown matrix: ")
                  << config.matrix_filter << '\n';
        *proceed = false;
        return matched ? 0 : 2;
    }
    if (config.matrix_filter.empty() || matched) return 0;
    std::cerr << "xyce_benchmark: unknown matrix: " << config.matrix_filter
              << "; supported matrices: ";
    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << specs[i].name;
    }
    std::cerr << '\n';
    *proceed = false;
    return 2;
}

std::vector<float> solve_cpu_reference(const ascend_gmres::CSRMatrix& matrix,
                                       const std::vector<float>& x_true) {
    XyceApplicationWrapper app(XyceSparseMatrixAdapter(matrix), x_true);
    (void)app;
    std::vector<float> b;
    ascend_gmres::csr_spmv_serial(matrix, x_true, &b);
    auto solver = ascend_gmres::make_cpu_single_gmres_solver();
    std::string error;
    if (!solver.prepare(matrix, &error)) throw std::runtime_error(error);
    std::vector<float> solution(static_cast<std::size_t>(matrix.cols), 0.0f);
    const auto result = solver.solve(
        b, &solution, ascend_gmres::GmresOptions{}, &error);
    if (!result.converged) {
        throw std::runtime_error("CPU reference solution did not converge");
    }
    return solution;
}

void write_solver_rows(std::ofstream* csv, const std::string& matrix_name,
                       const ascend_gmres::CSRMatrix& matrix,
                       const AveragedResult& cpu_single,
                       const AveragedResult& cpu_openmp,
                       const AveragedResult& ascend_device) {
    const double baseline = cpu_single.total_simulation_ms;
    *csv << csv_row(matrix_name, matrix, cpu_single, baseline) << '\n'
         << csv_row(matrix_name, matrix, cpu_openmp, baseline) << '\n'
         << csv_row(matrix_name, matrix, ascend_device, baseline) << '\n';
}

void run_matrix(const BenchmarkConfig& config,
                const ascend_gmres::MatrixSpec& spec,
                const XyceSimulationOptions& options, std::ofstream* csv) {
    const auto matrix = load_or_create_csr_matrix(config.matrix_dir, spec);
    const auto x_true = ascend_gmres::generate_solution_vector(matrix.cols, 42);
    std::cout << "Matrix: " << spec.name << " rows=" << matrix.rows
              << " cols=" << matrix.cols << " nnz=" << matrix.nnz() << '\n';
    const auto cpu_single = run_solver_repeated(
        matrix, x_true, LinearSolverKind::CpuSingle, options, config, nullptr);
    const auto reference = solve_cpu_reference(matrix, x_true);
    const auto cpu_openmp = run_solver_repeated(
        matrix, x_true, LinearSolverKind::CpuOpenMP16, options, config, &reference);
    const auto ascend_device = run_solver_repeated(
        matrix, x_true, LinearSolverKind::AscendDevice, options, config, &reference);
    print_solver(cpu_single, cpu_single.total_simulation_ms);
    print_solver(cpu_openmp, cpu_single.total_simulation_ms);
    print_solver(ascend_device, cpu_single.total_simulation_ms);
    std::cout << "  Actual backend: Ascend C RTC Device GMRES; device="
              << (std::getenv("DEVICE_ID") ? std::getenv("DEVICE_ID") : "0")
              << "; vectors remain in Device memory during Arnoldi\n"
              << "  Repeats are fresh solver invocations (matrix loaded once; adapter/solver/"
                 "Device state rebuilt per run; no warm benchmark metrics)\n";
    write_solver_rows(csv, spec.name, matrix, cpu_single, cpu_openmp, ascend_device);
}

std::int64_t count_csv_rows(const std::string& path) {
    std::ifstream input(path);
    std::string line;
    std::int64_t count = -1;
    while (std::getline(input, line)) {
        if (!line.empty()) ++count;
    }
    return std::max<std::int64_t>(0, count);
}

int verify_csv(const BenchmarkConfig& config,
               const std::vector<ascend_gmres::MatrixSpec>& specs) {
    const std::int64_t matrices = static_cast<std::int64_t>(std::count_if(
        specs.begin(), specs.end(), [&](const auto& spec) {
            return config.matrix_filter.empty() || config.matrix_filter == spec.name;
        }));
    const std::int64_t expected = matrices * 3;
    const std::int64_t actual = count_csv_rows(config.csv_path);
    if (actual == expected) return 0;
    std::cerr << "xyce_benchmark: expected " << expected
              << " solver rows for the selected matrices but CSV contains "
              << actual << " (file: " << config.csv_path << ")\n";
    return 3;
}

int run_benchmark(const BenchmarkConfig& config,
                  const std::vector<ascend_gmres::MatrixSpec>& specs) {
    fs::create_directories(config.results_dir);
    std::ofstream csv(config.csv_path);
    if (!csv) throw std::runtime_error("failed to open csv output: " + config.csv_path);
    csv << csv_header() << '\n';
    const XyceSimulationOptions options;
    for (const auto& spec : specs) {
        if (!config.matrix_filter.empty() && config.matrix_filter != spec.name) continue;
        run_matrix(config, spec, options, &csv);
    }
    csv.flush();
    csv.close();
    const int result = verify_csv(config, specs);
    if (result == 0) std::cout << "CSV saved to " << config.csv_path << '\n';
    return result;
}

}  // namespace ascend_xyce

int main(int argc, char** argv) {
    try {
        const auto config = ascend_xyce::parse_config(argc, argv);
        const auto specs = ascend_xyce::default_xyce_matrix_specs();
        bool proceed = true;
        const int selection = ascend_xyce::validate_selection(config, specs, &proceed);
        return proceed ? ascend_xyce::run_benchmark(config, specs) : selection;
    } catch (const std::exception& exception) {
        std::cerr << "xyce_benchmark failed: " << exception.what() << '\n';
        return 1;
    }
}

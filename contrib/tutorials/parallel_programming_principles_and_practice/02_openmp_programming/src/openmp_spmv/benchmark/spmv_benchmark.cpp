#include "benchmark.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

spmv::CSRMatrix load_or_generate(const spmv::BenchmarkConfig& config,
                                 const spmv::MatrixSpec& spec) {
    const fs::path path = fs::path(config.matrix_directory) / (spec.name + ".csrbin");
    spmv::CSRMatrix matrix;
    std::string error;
    if (fs::exists(path) && spmv::CSRMatrix::load_binary(path.string(), &matrix, &error)) return matrix;
    if (spec.name.front() == 'U') matrix = spmv::generate_uniform_matrix(spec, 42);
    else if (spec.name.front() == 'L') matrix = spmv::generate_long_tail_matrix(spec, 42);
    else matrix = spmv::generate_block_matrix(spec, 32, 42);
    if (!matrix.validate(&error)) throw std::runtime_error(error);
    fs::create_directories(path.parent_path());
    if (!matrix.save_binary(path.string(), &error)) throw std::runtime_error(error);
    return matrix;
}

template <class Backend>
double benchmark(Backend* backend, const spmv::CSRMatrix& matrix,
                 const std::vector<float>& x, const spmv::BenchmarkConfig& config,
                 std::vector<float>* output, spmv::BackendTimings* measured = nullptr) {
    std::string error;
    if (!backend->prepare(matrix, &error)) throw std::runtime_error(error);
    spmv::BackendTimings timings;
    std::vector<float> y;
    for (std::int32_t i = 0; i < config.warmup_iterations; ++i)
        if (!backend->run(x, &y, &timings, &error)) throw std::runtime_error(error);
    double total = 0.0;
    for (std::int32_t i = 0; i < config.repeat_iterations; ++i) {
        if (!backend->run(x, &y, &timings, &error)) throw std::runtime_error(error);
        total += timings.total_ms;
    }
    if (output) *output = y;
    if (measured) *measured = timings;
    return total / config.repeat_iterations;
}

struct BenchmarkResult {
    double serial_ms = 0.0;
    double openmp_ms = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    double bandwidth_gbps = 0.0;
    double error = 0.0;
    spmv::BackendTimings openmp_info;
    const char* schedule = "unknown";
};

std::string next_value(int argc, char** argv, int* index) {
    if (*index + 1 >= argc) throw std::runtime_error(std::string(argv[*index]) + " requires a value");
    return argv[++*index];
}

void apply_argument(const std::string& arg, const std::string& value,
                    spmv::BenchmarkConfig* config, std::string* matrix_filter) {
    if (arg == "--warmup") { config->warmup_iterations = std::stoi(value); return; }
    if (arg == "--repeat") { config->repeat_iterations = std::stoi(value); return; }
    if (arg == "--matrix") { *matrix_filter = value; return; }
    if (arg == "--matrix-dir") { config->matrix_directory = value; return; }
    if (arg == "--results-dir") { config->results_directory = value; return; }
    if (arg == "--csv") { config->csv_path = value; return; }
    throw std::runtime_error("unknown argument: " + arg);
}

bool is_value_argument(const std::string& arg) {
    return arg == "--warmup" || arg == "--repeat" || arg == "--matrix" ||
           arg == "--matrix-dir" || arg == "--results-dir" || arg == "--csv";
}

spmv::BenchmarkConfig parse_config(int argc, char** argv, std::string* matrix_filter) {
    spmv::BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (!is_value_argument(arg)) throw std::runtime_error("unknown argument: " + arg);
        apply_argument(arg, next_value(argc, argv, &i), &config, matrix_filter);
    }
    if (config.warmup_iterations < 0 || config.repeat_iterations <= 0) {
        throw std::runtime_error("warmup must be >= 0 and repeat must be > 0");
    }
    return config;
}

const char* schedule_name(std::int32_t schedule_kind) {
    switch (schedule_kind & 0x7fffffff) {
        case 1: return "static";
        case 2: return "dynamic";
        case 3: return "guided";
        case 4: return "auto";
        default: return "unknown";
    }
}

BenchmarkResult run_benchmark(const spmv::CSRMatrix& matrix,
                              const spmv::BenchmarkConfig& config) {
    const auto x = spmv::generate_rhs_vector(matrix.cols, 42);
    spmv::CpuSingleBackend serial;
    spmv::CpuOpenMpBackend openmp;
    std::vector<float> reference;
    std::vector<float> parallel;
    BenchmarkResult result;
    result.serial_ms = benchmark(&serial, matrix, x, config, &reference);
    result.openmp_ms = benchmark(&openmp, matrix, x, config, &parallel,
                                 &result.openmp_info);
    result.error = spmv::relative_error(reference, parallel);
    result.speedup = result.openmp_ms > 0.0 ? result.serial_ms / result.openmp_ms : 0.0;
    result.efficiency = result.openmp_info.actual_threads > 0
                            ? result.speedup / result.openmp_info.actual_threads : 0.0;
    const double bytes = static_cast<double>(matrix.nnz()) *
                         (sizeof(float) + sizeof(std::int32_t) + sizeof(float)) +
                         static_cast<double>(matrix.rows + 1) * sizeof(std::int32_t) +
                         static_cast<double>(matrix.rows) * sizeof(float);
    result.bandwidth_gbps = result.openmp_ms > 0.0
                                ? bytes / (result.openmp_ms * 1.0e6) : 0.0;
    result.schedule = schedule_name(result.openmp_info.schedule_kind);
    return result;
}

void print_result(const spmv::MatrixSpec& spec, const BenchmarkResult& result) {
    std::cout << "Matrix: " << spec.name << "\n  CPU single = " << result.serial_ms
              << " ms\n  CPU OpenMP = " << result.openmp_ms << " ms\n  speedup = "
              << result.speedup << "x\n  threads = " << result.openmp_info.actual_threads
              << "\n  schedule = " << result.schedule << " (chunk="
              << result.openmp_info.schedule_chunk << ')'
              << "\n  parallel_efficiency = " << result.efficiency
              << "\n  estimated_bandwidth_gbps = " << result.bandwidth_gbps
              << "\n  relative_error = " << result.error << '\n';
}

void write_result(std::ofstream* csv, const spmv::MatrixSpec& spec,
                  const spmv::CSRMatrix& matrix, const BenchmarkResult& result) {
    *csv << spec.name << ',' << matrix.rows << ',' << matrix.cols << ',' << matrix.nnz()
         << ',' << result.openmp_info.actual_threads << ',' << result.schedule << ','
         << result.openmp_info.schedule_chunk << ',' << result.serial_ms << ','
         << result.openmp_ms << ',' << result.speedup << ',' << result.efficiency << ','
         << result.bandwidth_gbps << ',' << result.error << '\n';
}

bool run_selected(const spmv::BenchmarkConfig& config, const std::string& matrix_filter,
                  std::ofstream* csv) {
    bool selected = false;
    for (const auto& spec : spmv::default_benchmark_specs()) {
        if (matrix_filter != "all" && matrix_filter != spec.name) continue;
        selected = true;
        const auto matrix = load_or_generate(config, spec);
        const auto result = run_benchmark(matrix, config);
        print_result(spec, result);
        write_result(csv, spec, matrix, result);
    }
    return selected;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string matrix_filter = "all";
        const spmv::BenchmarkConfig config = parse_config(argc, argv, &matrix_filter);
        fs::create_directories(config.results_directory);
        std::ofstream csv(config.csv_path);
        if (!csv) throw std::runtime_error("cannot open " + config.csv_path);
        csv << "matrix,rows,cols,nnz,threads,schedule,chunk,cpu_single_ms,cpu_openmp_ms,speedup,efficiency,bandwidth_gbps,error\n";
        std::cout << std::fixed << std::setprecision(6);
        if (!run_selected(config, matrix_filter, &csv)) {
            throw std::runtime_error("unknown matrix: " + matrix_filter);
        }
        std::cout << "CSV saved to " << config.csv_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }
}

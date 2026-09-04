#include "benchmark.hpp"
#if SPMV_REAL_ASCENDC
#include "ascendc_spmv.hpp"
#endif
#include "npu_spmv_context.hpp"
#include "spmv_bf16.hpp"
#include "spmv_fp16.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace spmv {

std::string csv_header() {
    return "matrix,rows,cols,nnz,sparsity,cpu_single,cpu_openmp16,host_prototype_fp32_baseline,host_prototype_fp32_optimized,host_prototype_fp16_mixed,host_prototype_bf16_mixed,host_prototype_bf16_persistent,host_prototype_fp32_baseline_speedup_cpu_single,host_prototype_fp32_baseline_speedup_cpu_openmp16,host_prototype_fp32_optimized_speedup_cpu_single,host_prototype_fp32_optimized_speedup_cpu_openmp16,host_prototype_fp16_mixed_speedup_cpu_single,host_prototype_fp16_mixed_speedup_cpu_openmp16,host_prototype_bf16_mixed_speedup_cpu_single,host_prototype_bf16_mixed_speedup_cpu_openmp16,host_prototype_bf16_persistent_speedup_cpu_single,host_prototype_bf16_persistent_speedup_cpu_openmp16,cpu_openmp16_error,host_prototype_fp32_baseline_error,host_prototype_fp32_optimized_error,host_prototype_fp16_mixed_error,host_prototype_bf16_mixed_error,host_prototype_bf16_persistent_error,host_prototype_fp32_baseline_init_ms,host_prototype_fp32_baseline_transfer_in_ms,host_prototype_fp32_baseline_kernel_ms,host_prototype_fp32_baseline_transfer_out_ms,host_prototype_fp32_baseline_total_ms,host_prototype_fp32_optimized_init_ms,host_prototype_fp32_optimized_transfer_in_ms,host_prototype_fp32_optimized_kernel_ms,host_prototype_fp32_optimized_transfer_out_ms,host_prototype_fp32_optimized_total_ms,host_prototype_fp16_mixed_init_ms,host_prototype_fp16_mixed_transfer_in_ms,host_prototype_fp16_mixed_kernel_ms,host_prototype_fp16_mixed_transfer_out_ms,host_prototype_fp16_mixed_total_ms,host_prototype_fp16_fp32_bytes,host_prototype_fp16_fp16_bytes,host_prototype_fp16_compression_ratio,host_prototype_bf16_init_ms,host_prototype_bf16_transfer_in_ms,host_prototype_bf16_kernel_ms,host_prototype_bf16_transfer_out_ms,host_prototype_bf16_total_ms,host_prototype_bf16_fp32_bytes,host_prototype_bf16_bf16_bytes,host_prototype_bf16_compression_ratio,host_prototype_bf16_persistent_cold_start_ms,host_prototype_bf16_persistent_warm_kernel_ms,host_prototype_bf16_persistent_amortized_runtime_ms,host_prototype_bf16_persistent_init_ms,host_prototype_bf16_persistent_transfer_in_ms,host_prototype_bf16_persistent_kernel_ms,host_prototype_bf16_persistent_transfer_out_ms,host_prototype_bf16_persistent_total_ms,host_prototype_bf16_persistent_fp32_bytes,host_prototype_bf16_persistent_bf16_bytes,host_prototype_bf16_persistent_compression_ratio,host_prototype_bf16_persistent_nnz_balance_ratio,host_prototype_bf16_persistent_partition_count,host_prototype_bf16_persistent_max_partition_nnz,host_prototype_bf16_persistent_avg_partition_nnz,real_ascendc_enabled,actual_backend,ascendc_ms,ascendc_error,ascendc_init_ms,ascendc_transfer_in_ms,ascendc_launch_to_complete_ms,ascendc_transfer_out_ms,ascendc_total_ms";
}

std::string format_error(double value) {
    // Correctness fields: scientific with 9 significant digits so errors like
    // 7.58e-08 survive the CSV/console audit.
    std::ostringstream out;
    out << std::scientific << std::setprecision(9) << value;
    return out.str();
}

void use_timing_format(std::ostringstream* out) {
    *out << std::fixed << std::setprecision(6);
}

void use_error_format(std::ostringstream* out) {
    *out << std::scientific << std::setprecision(9);
}

double speedup(double reference_ms, double measured_ms) {
    return measured_ms > 0.0 ? reference_ms / measured_ms : 0.0;
}

void append_summary_columns(std::ostringstream* out, const BenchmarkRecord& record) {
    *out << record.matrix_name << ','
        << record.rows << ','
        << record.cols << ','
        << record.nnz << ','
        << record.sparsity << ','
        << record.cpu_single_ms << ','
        << record.cpu_openmp16_ms << ','
        << record.host_prototype_baseline_ms << ','
        << record.host_prototype_optimized_ms << ','
        << record.host_prototype_mixed_ms << ','
        << record.host_prototype_bf16_ms << ','
        << record.host_prototype_bf16_persistent_ms << ','
        << speedup(record.cpu_single_ms, record.host_prototype_baseline_ms) << ','
        << speedup(record.cpu_openmp16_ms, record.host_prototype_baseline_ms) << ','
        << speedup(record.cpu_single_ms, record.host_prototype_optimized_ms) << ','
        << speedup(record.cpu_openmp16_ms, record.host_prototype_optimized_ms) << ','
        << record.host_prototype_mixed_speedup_over_cpu_single << ','
        << record.host_prototype_mixed_speedup_over_cpu_openmp16 << ','
        << record.host_prototype_bf16_speedup_over_cpu_single << ','
        << record.host_prototype_bf16_speedup_over_cpu_openmp16 << ','
        << record.host_prototype_bf16_persistent_speedup_over_cpu_single << ','
        << record.host_prototype_bf16_persistent_speedup_over_cpu_openmp16 << ',';
}

void append_error_columns(std::ostringstream* out, const BenchmarkRecord& record) {
    use_error_format(out);
    *out << record.cpu_openmp16_error << ','
        << record.host_prototype_baseline_error << ','
        << record.host_prototype_optimized_error << ','
        << record.host_prototype_mixed_error << ','
        << record.host_prototype_bf16_error << ','
        << record.host_prototype_bf16_persistent_error << ',';
}

void append_fp32_fp16_columns(std::ostringstream* out, const BenchmarkRecord& record) {
    use_timing_format(out);
    *out << record.host_prototype_baseline_init_ms << ','
        << record.host_prototype_baseline_transfer_in_ms << ','
        << record.host_prototype_baseline_kernel_ms << ','
        << record.host_prototype_baseline_transfer_out_ms << ','
        << record.host_prototype_baseline_total_ms << ','
        << record.host_prototype_optimized_init_ms << ','
        << record.host_prototype_optimized_transfer_in_ms << ','
        << record.host_prototype_optimized_kernel_ms << ','
        << record.host_prototype_optimized_transfer_out_ms << ','
        << record.host_prototype_optimized_total_ms << ','
        << record.host_prototype_mixed_init_ms << ','
        << record.host_prototype_mixed_transfer_in_ms << ','
        << record.host_prototype_mixed_kernel_ms << ','
        << record.host_prototype_mixed_transfer_out_ms << ','
        << record.host_prototype_mixed_total_ms << ','
        << record.host_prototype_mixed_fp32_bytes << ','
        << record.host_prototype_mixed_fp16_bytes << ','
        << record.host_prototype_mixed_compression_ratio << ',';
}

void append_bf16_columns(std::ostringstream* out, const BenchmarkRecord& record) {
    *out << record.host_prototype_bf16_init_ms << ','
        << record.host_prototype_bf16_transfer_in_ms << ','
        << record.host_prototype_bf16_kernel_ms << ','
        << record.host_prototype_bf16_transfer_out_ms << ','
        << record.host_prototype_bf16_total_ms << ','
        << record.host_prototype_bf16_fp32_bytes << ','
        << record.host_prototype_bf16_bf16_bytes << ','
        << record.host_prototype_bf16_compression_ratio << ','
        << record.host_prototype_bf16_persistent_cold_start_ms << ','
        << record.host_prototype_bf16_persistent_warm_kernel_ms << ','
        << record.host_prototype_bf16_persistent_amortized_runtime_ms << ','
        << record.host_prototype_bf16_persistent_init_ms << ','
        << record.host_prototype_bf16_persistent_transfer_in_ms << ','
        << record.host_prototype_bf16_persistent_kernel_ms << ','
        << record.host_prototype_bf16_persistent_transfer_out_ms << ','
        << record.host_prototype_bf16_persistent_total_ms << ','
        << record.host_prototype_bf16_persistent_fp32_bytes << ','
        << record.host_prototype_bf16_persistent_bf16_bytes << ','
        << record.host_prototype_bf16_persistent_compression_ratio << ','
        << record.host_prototype_bf16_persistent_nnz_balance_ratio << ','
        << record.host_prototype_bf16_persistent_partition_count << ','
        << record.host_prototype_bf16_persistent_max_partition_nnz << ','
        << record.host_prototype_bf16_persistent_avg_partition_nnz << ',';
}

void append_ascendc_columns(std::ostringstream* out, const BenchmarkRecord& record) {
    *out << record.real_ascendc_enabled << ','
        << record.actual_backend << ','
        << record.ascendc_ms << ',';
    use_error_format(out);
    *out << record.ascendc_error << ',';
    use_timing_format(out);
    *out << record.ascendc_init_ms << ','
        << record.ascendc_transfer_in_ms << ','
        << record.ascendc_launch_to_complete_ms << ','
        << record.ascendc_transfer_out_ms << ',' << record.ascendc_total_ms;
}

std::string csv_row(const BenchmarkRecord& record) {
    std::ostringstream out;
    use_timing_format(&out);
    append_summary_columns(&out, record);
    append_error_columns(&out, record);
    append_fp32_fp16_columns(&out, record);
    append_bf16_columns(&out, record);
    append_ascendc_columns(&out, record);
    return out.str();
}

void ensure_directory(const std::string& path) {
    if (!path.empty()) {
        fs::create_directories(path);
    }
}

std::vector<MatrixSpec> selected_specs(const std::string& matrix_filter) {
    const auto specs = default_benchmark_specs();
    if (matrix_filter == "all") return specs;
    for (const auto& spec : specs) {
        if (spec.name == matrix_filter) return {spec};
    }
    throw std::runtime_error("unknown matrix '" + matrix_filter +
                             "' (expected U1, U2, L1, L2, B1, B2, or all)");
}

std::string matrix_cache_path(const BenchmarkConfig& config, const MatrixSpec& spec) {
    return (fs::path(config.matrix_directory) / (spec.name + ".csrbin")).string();
}

CSRMatrix load_or_generate_matrix(const BenchmarkConfig& config, const MatrixSpec& spec) {
    const fs::path cache_path = matrix_cache_path(config, spec);
    CSRMatrix matrix;
    std::string error_message;

    if (fs::exists(cache_path) && CSRMatrix::load_binary(cache_path.string(), &matrix, &error_message)) {
        return matrix;
    }

    if (spec.name[0] == 'U') {
        matrix = generate_uniform_matrix(spec, 42);
    } else if (spec.name[0] == 'L') {
        matrix = generate_long_tail_matrix(spec, 42);
    } else {
        matrix = generate_block_matrix(spec, 32, 42);
    }

    if (!matrix.validate(&error_message)) {
        throw std::runtime_error("generated matrix validation failed for " + spec.name + ": " + error_message);
    }

    fs::create_directories(cache_path.parent_path());
    if (!matrix.save_binary(cache_path.string(), &error_message)) {
        throw std::runtime_error("failed to save matrix cache for " + spec.name + ": " + error_message);
    }
    return matrix;
}

void write_csv_file(const std::string& csv_path, const std::vector<BenchmarkRecord>& records) {
    std::ofstream out(csv_path);
    if (!out) {
        throw std::runtime_error("failed to open csv output: " + csv_path);
    }
    out << csv_header() << '\n';
    for (const auto& record : records) {
        out << csv_row(record) << '\n';
    }
}

void print_version(const std::string& name, double time_ms,
                   double speedup_single, double speedup_openmp, double error) {
    std::cout << "  " << name << " | " << time_ms << " | " << speedup_single
              << " | " << speedup_openmp << " | " << format_error(error) << '\n';
}

void print_comparison(const BenchmarkRecord& record) {
    std::cout << "Matrix: " << record.matrix_name << '\n';
    std::cout << "  size: " << record.rows << " x " << record.cols << ", nnz="
              << record.nnz << ", sparsity=" << std::fixed << std::setprecision(6)
              << record.sparsity << '\n';
    std::cout << "  version | time(ms) | speedup_vs_cpu_single | "
                 "speedup_vs_cpu_openmp16 | relative_error\n";
    std::cout << "  CPU single | " << record.cpu_single_ms
              << " | 1.000000 | n/a | 0.000000\n";
    print_version("CPU OpenMP16", record.cpu_openmp16_ms,
                  speedup(record.cpu_single_ms, record.cpu_openmp16_ms),
                  1.0, record.cpu_openmp16_error);
    print_version("Host FP32 baseline prototype", record.host_prototype_baseline_ms,
                  speedup(record.cpu_single_ms, record.host_prototype_baseline_ms),
                  speedup(record.cpu_openmp16_ms, record.host_prototype_baseline_ms),
                  record.host_prototype_baseline_error);
    print_version("Host optimized FP32 prototype", record.host_prototype_optimized_ms,
                  speedup(record.cpu_single_ms, record.host_prototype_optimized_ms),
                  speedup(record.cpu_openmp16_ms, record.host_prototype_optimized_ms),
                  record.host_prototype_optimized_error);
    print_version("Host FP16-FP32 Prototype", record.host_prototype_mixed_ms,
                  record.host_prototype_mixed_speedup_over_cpu_single,
                  record.host_prototype_mixed_speedup_over_cpu_openmp16,
                  record.host_prototype_mixed_error);
    print_version("Host BF16-FP32 Prototype", record.host_prototype_bf16_ms,
                  record.host_prototype_bf16_speedup_over_cpu_single,
                  record.host_prototype_bf16_speedup_over_cpu_openmp16,
                  record.host_prototype_bf16_error);
    print_version("Host BF16 Persistent Prototype", record.host_prototype_bf16_persistent_ms,
                  record.host_prototype_bf16_persistent_speedup_over_cpu_single,
                  record.host_prototype_bf16_persistent_speedup_over_cpu_openmp16,
                  record.host_prototype_bf16_persistent_error);
}

void print_stage(const std::string& name, double init_ms, double transfer_in_ms,
                 double kernel_ms, double transfer_out_ms, double total_ms) {
    std::cout << "  " << name << " stages(ms): init=" << init_ms
              << ", transfer_in=" << transfer_in_ms << ", kernel=" << kernel_ms
              << ", transfer_out=" << transfer_out_ms << ", total=" << total_ms << '\n';
}

void print_stages(const BenchmarkRecord& record) {
    print_stage("host_prototype_fp32 baseline", record.host_prototype_baseline_init_ms,
                record.host_prototype_baseline_transfer_in_ms,
                record.host_prototype_baseline_kernel_ms,
                record.host_prototype_baseline_transfer_out_ms,
                record.host_prototype_baseline_total_ms);
    print_stage("host_prototype_fp32 optimized", record.host_prototype_optimized_init_ms,
                record.host_prototype_optimized_transfer_in_ms,
                record.host_prototype_optimized_kernel_ms,
                record.host_prototype_optimized_transfer_out_ms,
                record.host_prototype_optimized_total_ms);
    print_stage("host_prototype_fp16 mixed", record.host_prototype_mixed_init_ms,
                record.host_prototype_mixed_transfer_in_ms,
                record.host_prototype_mixed_kernel_ms,
                record.host_prototype_mixed_transfer_out_ms,
                record.host_prototype_mixed_total_ms);
}

void print_memory(const BenchmarkRecord& record) {
    std::cout << "  mixed memory(bytes): fp32=" << record.host_prototype_mixed_fp32_bytes
              << ", fp16=" << record.host_prototype_mixed_fp16_bytes
              << ", compression_ratio=" << record.host_prototype_mixed_compression_ratio << '\n';
    std::cout << "  bf16 memory(bytes): fp32=" << record.host_prototype_bf16_fp32_bytes
              << ", bf16=" << record.host_prototype_bf16_bf16_bytes
              << ", compression_ratio=" << record.host_prototype_bf16_compression_ratio << '\n';
    std::cout << "  bf16 persistent memory(bytes): fp32="
              << record.host_prototype_bf16_persistent_fp32_bytes << ", bf16="
              << record.host_prototype_bf16_persistent_bf16_bytes << ", compression_ratio="
              << record.host_prototype_bf16_persistent_compression_ratio << '\n';
    std::cout << "  bf16 persistent partitions: count="
              << record.host_prototype_bf16_persistent_partition_count
              << ", nnz_balance_ratio=" << record.host_prototype_bf16_persistent_nnz_balance_ratio
              << ", max_partition_nnz=" << record.host_prototype_bf16_persistent_max_partition_nnz
              << ", avg_partition_nnz=" << record.host_prototype_bf16_persistent_avg_partition_nnz
              << '\n';
    std::cout << "  bf16 persistent runtime: cold_start="
              << record.host_prototype_bf16_persistent_cold_start_ms << ", warm_kernel="
              << record.host_prototype_bf16_persistent_warm_kernel_ms << ", amortized_runtime="
              << record.host_prototype_bf16_persistent_amortized_runtime_ms << '\n';
}

void print_record(const BenchmarkRecord& record) {
    print_comparison(record);
    print_stages(record);
    print_memory(record);
}

template <typename Backend>
double benchmark_backend(Backend& backend,
                         const CSRMatrix& matrix,
                         const std::vector<float>& x,
                         const std::vector<float>& reference,
                         const BenchmarkConfig& config,
                         std::vector<float>* output,
                         double* relative_error_out,
                         BackendTimings* avg_timings) {
    std::string error_message;
    if (!backend.prepare(matrix, &error_message)) {
        throw std::runtime_error(backend.name() + " prepare failed: " + error_message);
    }

    std::vector<float> y;
    BackendTimings timings;
    double total_ms_sum = 0.0;
    double transfer_in_ms_sum = 0.0;
    double kernel_ms_sum = 0.0;
    double transfer_out_ms_sum = 0.0;

    for (std::int32_t i = 0; i < config.warmup_iterations; ++i) {
        if (!backend.run(x, &y, &timings, &error_message)) {
            throw std::runtime_error(backend.name() + " warmup failed: " + error_message);
        }
    }

    for (std::int32_t i = 0; i < config.repeat_iterations; ++i) {
        if (!backend.run(x, &y, &timings, &error_message)) {
            throw std::runtime_error(backend.name() + " run failed: " + error_message);
        }
        total_ms_sum += timings.total_ms;
        transfer_in_ms_sum += timings.transfer_in_ms;
        kernel_ms_sum += timings.kernel_ms;
        transfer_out_ms_sum += timings.transfer_out_ms;
    }

    if (output != nullptr) {
        *output = y;
    }
    if (relative_error_out != nullptr && !reference.empty() && reference.size() == y.size()) {
        *relative_error_out = relative_error(reference, y);
    }
    if (avg_timings != nullptr) {
        avg_timings->initialization_ms = backend.initialization_ms();
        avg_timings->transfer_in_ms = transfer_in_ms_sum / static_cast<double>(config.repeat_iterations);
        avg_timings->kernel_ms = kernel_ms_sum / static_cast<double>(config.repeat_iterations);
        avg_timings->transfer_out_ms = transfer_out_ms_sum / static_cast<double>(config.repeat_iterations);
        avg_timings->total_ms = total_ms_sum / static_cast<double>(config.repeat_iterations);
    }
    return total_ms_sum / static_cast<double>(config.repeat_iterations);
}

BenchmarkConfig parse_config(int argc, char** argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("unknown argument: " + arg);
        const std::string value = argv[++i];
        if (arg == "--warmup") { config.warmup_iterations = std::stoi(value); continue; }
        if (arg == "--repeat") { config.repeat_iterations = std::stoi(value); continue; }
        if (arg == "--matrix-dir") { config.matrix_directory = value; continue; }
        if (arg == "--results-dir") { config.results_directory = value; continue; }
        if (arg == "--csv") { config.csv_path = value; continue; }
        if (arg == "--matrix") { config.matrix_filter = value; continue; }
        throw std::runtime_error("unknown argument: " + arg);
    }
    if (config.warmup_iterations < 0) throw std::runtime_error("--warmup must be >= 0");
    if (config.repeat_iterations <= 0) throw std::runtime_error("--repeat must be > 0");
    return config;
}

BenchmarkRecord make_record(const MatrixSpec& spec, const CSRMatrix& matrix) {
    BenchmarkRecord record;
    record.matrix_name = spec.name;
    record.rows = matrix.rows;
    record.cols = matrix.cols;
    record.nnz = matrix.nnz();
    record.sparsity = 1.0 - static_cast<double>(record.nnz) /
        (static_cast<double>(record.rows) * static_cast<double>(record.cols));
    return record;
}

void benchmark_cpu_backends(const CSRMatrix& matrix, const std::vector<float>& rhs,
                            const BenchmarkConfig& config,
                            std::vector<float>* golden, BenchmarkRecord* record) {
    CpuSingleBackend cpu_single;
    record->cpu_single_ms = benchmark_backend(
        cpu_single, matrix, rhs, *golden, config, golden, nullptr, nullptr);
    record->cpu_single_error = 0.0;
    CpuOpenMp16Backend cpu_openmp;
    record->cpu_openmp16_ms = benchmark_backend(
        cpu_openmp, matrix, rhs, *golden, config, nullptr,
        &record->cpu_openmp16_error, nullptr);
}

void benchmark_fp32_backends(const CSRMatrix& matrix, const std::vector<float>& rhs,
                             const std::vector<float>& golden,
                             const BenchmarkConfig& config, BenchmarkRecord* record) {
    HostPrototypeBaselineBackend baseline;
    BackendTimings baseline_time;
    record->host_prototype_baseline_ms = benchmark_backend(
        baseline, matrix, rhs, golden, config, nullptr,
        &record->host_prototype_baseline_error, &baseline_time);
    record->host_prototype_baseline_init_ms = baseline_time.initialization_ms;
    record->host_prototype_baseline_transfer_in_ms = baseline_time.transfer_in_ms;
    record->host_prototype_baseline_kernel_ms = baseline_time.kernel_ms;
    record->host_prototype_baseline_transfer_out_ms = baseline_time.transfer_out_ms;
    record->host_prototype_baseline_total_ms = baseline_time.total_ms;
    HostPrototypeOptimizedBackend optimized;
    BackendTimings optimized_time;
    record->host_prototype_optimized_ms = benchmark_backend(
        optimized, matrix, rhs, golden, config, nullptr,
        &record->host_prototype_optimized_error, &optimized_time);
    record->host_prototype_optimized_init_ms = optimized_time.initialization_ms;
    record->host_prototype_optimized_transfer_in_ms = optimized_time.transfer_in_ms;
    record->host_prototype_optimized_kernel_ms = optimized_time.kernel_ms;
    record->host_prototype_optimized_transfer_out_ms = optimized_time.transfer_out_ms;
    record->host_prototype_optimized_total_ms = optimized_time.total_ms;
}

void benchmark_fp16_backend(const CSRMatrix& matrix, const std::vector<float>& rhs,
                            const std::vector<float>& golden,
                            const BenchmarkConfig& config, BenchmarkRecord* record) {
    HostPrototypeFp16Fp32Backend backend;
    BackendTimings timing;
    record->host_prototype_mixed_ms = benchmark_backend(
        backend, matrix, rhs, golden, config, nullptr,
        &record->host_prototype_mixed_error, &timing);
    record->host_prototype_mixed_init_ms = timing.initialization_ms;
    record->host_prototype_mixed_transfer_in_ms = timing.transfer_in_ms;
    record->host_prototype_mixed_kernel_ms = timing.kernel_ms;
    record->host_prototype_mixed_transfer_out_ms = timing.transfer_out_ms;
    record->host_prototype_mixed_total_ms = timing.total_ms;
    record->host_prototype_mixed_fp32_bytes = static_cast<double>(backend.fp32_csr_bytes());
    record->host_prototype_mixed_fp16_bytes = static_cast<double>(backend.fp16_csr_bytes());
    record->host_prototype_mixed_compression_ratio = backend.compression_ratio();
}

void benchmark_bf16_backend(const CSRMatrix& matrix, const std::vector<float>& rhs,
                            const std::vector<float>& golden,
                            const BenchmarkConfig& config, BenchmarkRecord* record) {
    HostPrototypeBf16Fp32Backend backend;
    BackendTimings timing;
    record->host_prototype_bf16_ms = benchmark_backend(
        backend, matrix, rhs, golden, config, nullptr,
        &record->host_prototype_bf16_error, &timing);
    record->host_prototype_bf16_init_ms = timing.initialization_ms;
    record->host_prototype_bf16_transfer_in_ms = timing.transfer_in_ms;
    record->host_prototype_bf16_kernel_ms = timing.kernel_ms;
    record->host_prototype_bf16_transfer_out_ms = timing.transfer_out_ms;
    record->host_prototype_bf16_total_ms = timing.total_ms;
    record->host_prototype_bf16_fp32_bytes = static_cast<double>(backend.fp32_csr_bytes());
    record->host_prototype_bf16_bf16_bytes = static_cast<double>(backend.bf16_csr_bytes());
    record->host_prototype_bf16_compression_ratio = backend.compression_ratio();
}

void benchmark_persistent_backend(const CSRMatrix& matrix, const std::vector<float>& rhs,
                                  const std::vector<float>& golden,
                                  const BenchmarkConfig& config, BenchmarkRecord* record) {
    HostPrototypeBf16PersistentBackend backend;
    BackendTimings timing;
    record->host_prototype_bf16_persistent_ms = benchmark_backend(
        backend, matrix, rhs, golden, config, nullptr,
        &record->host_prototype_bf16_persistent_error, &timing);
    record->host_prototype_bf16_persistent_cold_start_ms = timing.initialization_ms;
    record->host_prototype_bf16_persistent_init_ms = timing.initialization_ms;
    record->host_prototype_bf16_persistent_transfer_in_ms = timing.transfer_in_ms;
    record->host_prototype_bf16_persistent_kernel_ms = timing.kernel_ms;
    record->host_prototype_bf16_persistent_transfer_out_ms = timing.transfer_out_ms;
    record->host_prototype_bf16_persistent_total_ms = timing.total_ms;
    record->host_prototype_bf16_persistent_warm_kernel_ms = timing.kernel_ms;
    record->host_prototype_bf16_persistent_amortized_runtime_ms = timing.total_ms +
        timing.initialization_ms / static_cast<double>(config.repeat_iterations);
    record->host_prototype_bf16_persistent_fp32_bytes =
        static_cast<double>(backend.fp32_csr_bytes());
    record->host_prototype_bf16_persistent_bf16_bytes =
        static_cast<double>(backend.bf16_csr_bytes());
    record->host_prototype_bf16_persistent_compression_ratio = backend.compression_ratio();
    record->host_prototype_bf16_persistent_nnz_balance_ratio = backend.nnz_balance_ratio();
    record->host_prototype_bf16_persistent_partition_count = backend.partition_count();
    record->host_prototype_bf16_persistent_max_partition_nnz = backend.max_partition_nnz();
    record->host_prototype_bf16_persistent_avg_partition_nnz = backend.avg_partition_nnz();
}

void calculate_speedups(BenchmarkRecord* record) {
    record->speedup_over_cpu_single = record->cpu_single_ms / record->host_prototype_optimized_ms;
    record->speedup_over_cpu_openmp16 = record->cpu_openmp16_ms / record->host_prototype_optimized_ms;
    record->host_prototype_mixed_speedup_over_cpu_single =
        record->cpu_single_ms / record->host_prototype_mixed_ms;
    record->host_prototype_mixed_speedup_over_cpu_openmp16 =
        record->cpu_openmp16_ms / record->host_prototype_mixed_ms;
    record->host_prototype_bf16_speedup_over_cpu_single =
        record->cpu_single_ms / record->host_prototype_bf16_ms;
    record->host_prototype_bf16_speedup_over_cpu_openmp16 =
        record->cpu_openmp16_ms / record->host_prototype_bf16_ms;
    record->host_prototype_bf16_persistent_speedup_over_cpu_single =
        record->cpu_single_ms / record->host_prototype_bf16_persistent_ms;
    record->host_prototype_bf16_persistent_speedup_over_cpu_openmp16 =
        record->cpu_openmp16_ms / record->host_prototype_bf16_persistent_ms;
}

void benchmark_ascendc_backend(const CSRMatrix& matrix, const std::vector<float>& rhs,
                               const std::vector<float>& golden,
                               const BenchmarkConfig& config, BenchmarkRecord* record) {
#if SPMV_REAL_ASCENDC
    AscendCSpmvBackend backend(0);
    BackendTimings timing;
    record->ascendc_ms = benchmark_backend(
        backend, matrix, rhs, golden, config, nullptr, &record->ascendc_error, &timing);
    record->real_ascendc_enabled = 1;
    record->actual_backend = "ascend_c";
    record->ascendc_init_ms = timing.initialization_ms;
    record->ascendc_transfer_in_ms = timing.transfer_in_ms;
    record->ascendc_launch_to_complete_ms = timing.kernel_ms;
    record->ascendc_transfer_out_ms = timing.transfer_out_ms;
    record->ascendc_total_ms = timing.total_ms;
    std::cout << "Actual Backend=Ascend C RTC\nDevice ID=0\n"
              << "NPU FP32 time=" << record->ascendc_ms << " ms\n"
              << "H2D=" << timing.transfer_in_ms << " ms\n"
              << "Kernel+Sync=" << timing.kernel_ms << " ms\n"
              << "D2H=" << timing.transfer_out_ms << " ms\n"
              << "CPU Reference Error=" << format_error(record->ascendc_error) << '\n';
    if (record->ascendc_error >= 1e-6) {
        throw std::runtime_error("Ascend C correctness check failed");
    }
#else
    (void)matrix;
    (void)rhs;
    (void)golden;
    (void)config;
    record->real_ascendc_enabled = 0;
    record->actual_backend = "host_only";
#endif
}

BenchmarkRecord run_spec(const BenchmarkConfig& config, const MatrixSpec& spec) {
    const CSRMatrix matrix = load_or_generate_matrix(config, spec);
    std::string error;
    if (!matrix.validate(&error)) {
        throw std::runtime_error("matrix validation failed for " + spec.name + ": " + error);
    }
    BenchmarkRecord record = make_record(spec, matrix);
    const std::vector<float> rhs = generate_rhs_vector(matrix.cols, 42);
    std::vector<float> golden;
    benchmark_cpu_backends(matrix, rhs, config, &golden, &record);
    benchmark_fp32_backends(matrix, rhs, golden, config, &record);
    benchmark_fp16_backend(matrix, rhs, golden, config, &record);
    benchmark_bf16_backend(matrix, rhs, golden, config, &record);
    benchmark_persistent_backend(matrix, rhs, golden, config, &record);
    calculate_speedups(&record);
    benchmark_ascendc_backend(matrix, rhs, golden, config, &record);
    print_record(record);
    return record;
}

}  // namespace spmv

int main(int argc, char** argv) {
    try {
        const spmv::BenchmarkConfig config = spmv::parse_config(argc, argv);
        spmv::ensure_directory(config.matrix_directory);
        spmv::ensure_directory(config.results_directory);
        const auto specs = spmv::selected_specs(config.matrix_filter);
        std::vector<spmv::BenchmarkRecord> records;
        records.reserve(specs.size());
        for (const auto& spec : specs) {
            records.push_back(spmv::run_spec(config, spec));
        }
        spmv::write_csv_file(config.csv_path, records);
        std::cout << "CSV saved to " << config.csv_path << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "benchmark failed: " << exception.what() << '\n';
        return 1;
    }
}

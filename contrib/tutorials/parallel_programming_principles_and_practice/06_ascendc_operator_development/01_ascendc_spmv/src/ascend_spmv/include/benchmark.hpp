#pragma once

#include "spmv.hpp"

#include <cstdint>
#include <string>

namespace spmv {

struct BenchmarkConfig {
    std::int32_t warmup_iterations = 10;
    std::int32_t repeat_iterations = 100;
    std::string matrix_directory = "matrices";
    std::string results_directory = "results";
    std::string csv_path = "results/spmv_benchmark.csv";
    std::string matrix_filter = "all";
};

struct BenchmarkRecord {
    std::string matrix_name;
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::int64_t nnz = 0;
    double sparsity = 0.0;

    double cpu_single_ms = 0.0;
    double cpu_openmp16_ms = 0.0;
    double host_prototype_baseline_ms = 0.0;
    double host_prototype_optimized_ms = 0.0;

    double cpu_single_error = 0.0;
    double cpu_openmp16_error = 0.0;
    double host_prototype_baseline_error = 0.0;
    double host_prototype_optimized_error = 0.0;

    double host_prototype_baseline_init_ms = 0.0;
    double host_prototype_baseline_transfer_in_ms = 0.0;
    double host_prototype_baseline_kernel_ms = 0.0;
    double host_prototype_baseline_transfer_out_ms = 0.0;
    double host_prototype_baseline_total_ms = 0.0;

    double host_prototype_optimized_init_ms = 0.0;
    double host_prototype_optimized_transfer_in_ms = 0.0;
    double host_prototype_optimized_kernel_ms = 0.0;
    double host_prototype_optimized_transfer_out_ms = 0.0;
    double host_prototype_optimized_total_ms = 0.0;

    double host_prototype_mixed_ms = 0.0;
    double host_prototype_mixed_error = 0.0;
    double host_prototype_mixed_init_ms = 0.0;
    double host_prototype_mixed_transfer_in_ms = 0.0;
    double host_prototype_mixed_kernel_ms = 0.0;
    double host_prototype_mixed_transfer_out_ms = 0.0;
    double host_prototype_mixed_total_ms = 0.0;
    double host_prototype_mixed_fp32_bytes = 0.0;
    double host_prototype_mixed_fp16_bytes = 0.0;
    double host_prototype_mixed_compression_ratio = 0.0;
    double host_prototype_mixed_speedup_over_cpu_single = 0.0;
    double host_prototype_mixed_speedup_over_cpu_openmp16 = 0.0;

    double host_prototype_bf16_ms = 0.0;
    double host_prototype_bf16_error = 0.0;
    double host_prototype_bf16_init_ms = 0.0;
    double host_prototype_bf16_transfer_in_ms = 0.0;
    double host_prototype_bf16_kernel_ms = 0.0;
    double host_prototype_bf16_transfer_out_ms = 0.0;
    double host_prototype_bf16_total_ms = 0.0;
    double host_prototype_bf16_fp32_bytes = 0.0;
    double host_prototype_bf16_bf16_bytes = 0.0;
    double host_prototype_bf16_compression_ratio = 0.0;
    double host_prototype_bf16_speedup_over_cpu_single = 0.0;
    double host_prototype_bf16_speedup_over_cpu_openmp16 = 0.0;

    double host_prototype_bf16_persistent_ms = 0.0;
    double host_prototype_bf16_persistent_error = 0.0;
    double host_prototype_bf16_persistent_init_ms = 0.0;
    double host_prototype_bf16_persistent_transfer_in_ms = 0.0;
    double host_prototype_bf16_persistent_kernel_ms = 0.0;
    double host_prototype_bf16_persistent_transfer_out_ms = 0.0;
    double host_prototype_bf16_persistent_total_ms = 0.0;
    double host_prototype_bf16_persistent_cold_start_ms = 0.0;
    double host_prototype_bf16_persistent_warm_kernel_ms = 0.0;
    double host_prototype_bf16_persistent_amortized_runtime_ms = 0.0;
    double host_prototype_bf16_persistent_fp32_bytes = 0.0;
    double host_prototype_bf16_persistent_bf16_bytes = 0.0;
    double host_prototype_bf16_persistent_compression_ratio = 0.0;
    double host_prototype_bf16_persistent_speedup_over_cpu_single = 0.0;
    double host_prototype_bf16_persistent_speedup_over_cpu_openmp16 = 0.0;
    double host_prototype_bf16_persistent_nnz_balance_ratio = 0.0;
    std::int32_t host_prototype_bf16_persistent_partition_count = 0;
    std::int64_t host_prototype_bf16_persistent_max_partition_nnz = 0;
    double host_prototype_bf16_persistent_avg_partition_nnz = 0.0;

    double speedup_over_cpu_single = 0.0;
    double speedup_over_cpu_openmp16 = 0.0;

    // Real Ascend C (ACL RTC) backend results; the same BenchmarkRecord row
    // carries them so a CSV row is never a mix of fabricated NPU numbers.
    std::int32_t real_ascendc_enabled = 0;
    std::string actual_backend = "host_only";
    double ascendc_ms = 0.0;
    double ascendc_error = 0.0;
    double ascendc_init_ms = 0.0;
    double ascendc_transfer_in_ms = 0.0;
    double ascendc_launch_to_complete_ms = 0.0;
    double ascendc_transfer_out_ms = 0.0;
    double ascendc_total_ms = 0.0;
};

std::string csv_header();
std::string csv_row(const BenchmarkRecord& record);

}  // namespace spmv

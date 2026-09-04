#include "npu_spmv_context.hpp"
#include "spmv_fp16.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>

namespace spmv {

namespace {

std::int64_t partition_nnz(const CSRMatrixBF16& matrix, std::int32_t begin_row, std::int32_t end_row) {
    const std::int32_t safe_begin = std::max<std::int32_t>(0, begin_row);
    const std::int32_t safe_end = std::min(end_row, matrix.rows);
    if (safe_begin >= safe_end) {
        return 0;
    }
    return static_cast<std::int64_t>(matrix.row_ptr[static_cast<std::size_t>(safe_end)] - matrix.row_ptr[static_cast<std::size_t>(safe_begin)]);
}

std::vector<std::int32_t> normalize_partition_candidates(const CSRMatrix& matrix, std::vector<std::int32_t> candidates) {
    if (candidates.empty()) {
        candidates = {8, 16, 32};
    }
    for (auto& candidate : candidates) {
        candidate = std::max<std::int32_t>(1, std::min(candidate, matrix.rows));
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

void fill_partition_stats(const CSRMatrix& source_matrix, const CSRMatrixBF16& bf16_matrix, const std::vector<std::int32_t>& partitions, HostPrototypeSpmvContextStats* stats) {
    if (stats == nullptr) {
        return;
    }

    stats->fp32_csr_bytes = csr_fp32_bytes(source_matrix);
    stats->bf16_csr_bytes = csr_bf16_bytes(bf16_matrix);
    stats->compression_ratio = stats->bf16_csr_bytes == 0 ? 0.0 : static_cast<double>(stats->fp32_csr_bytes) / static_cast<double>(stats->bf16_csr_bytes);
    stats->partition_count = static_cast<std::int32_t>(std::max<std::size_t>(1, partitions.size() - 1));
    stats->avg_partition_nnz = stats->partition_count > 0 ? static_cast<double>(bf16_matrix.nnz()) / static_cast<double>(stats->partition_count) : 0.0;

    std::int64_t max_partition_nnz = 0;
    for (std::size_t index = 0; index + 1 < partitions.size(); ++index) {
        const std::int64_t part_nnz = partition_nnz(bf16_matrix, partitions[index], partitions[index + 1]);
        max_partition_nnz = std::max(max_partition_nnz, part_nnz);
    }
    stats->max_partition_nnz = max_partition_nnz;
    stats->nnz_balance_ratio = stats->avg_partition_nnz > 0.0 ? static_cast<double>(stats->max_partition_nnz) / stats->avg_partition_nnz : 0.0;
}

}  // namespace

std::int32_t host_prototype_partition_count() {
    const char* env_value = std::getenv("NPU_AI_CORE_COUNT");
    if (env_value == nullptr || *env_value == '\0') {
        env_value = std::getenv("ASCEND_AI_CORE_COUNT");
    }
    if (env_value == nullptr || *env_value == '\0') {
        return 32;
    }

    try {
        const int parsed = std::stoi(env_value);
        return parsed > 0 ? parsed : 32;
    } catch (...) {
        return 32;
    }
}

bool HostPrototypeSpmvContext::initialize_bf16(const CSRMatrix& matrix, std::int32_t partition_hint, std::string* error) {
    return initialize_bf16_autotuned(matrix, {partition_hint}, error);
}

bool HostPrototypeSpmvContext::initialize_bf16_autotuned(const CSRMatrix& matrix, const std::vector<std::int32_t>& partition_candidates, std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    if (!matrix.validate(error)) {
        return false;
    }

    initialized_ = false;
    host_matrix_bf16_ = convert_csr_values_to_bf16(matrix);
    host_x_bf16_.assign(static_cast<std::size_t>(matrix.cols), float32_to_bf16_bits(1.0f));
    host_y_fp32_.resize(static_cast<std::size_t>(matrix.rows));

    const auto candidates = normalize_partition_candidates(matrix, partition_candidates);
    double best_kernel_ms = std::numeric_limits<double>::max();
    std::vector<std::int32_t> best_partitions;

    for (const std::int32_t candidate : candidates) {
        auto partitions = build_nnz_aware_partitions(matrix, candidate);
        const auto kernel_start = std::chrono::steady_clock::now();
        spmv_partitioned_bf16(host_matrix_bf16_, partitions, host_x_bf16_, &host_y_fp32_, true);
        const auto kernel_end = std::chrono::steady_clock::now();
        const double kernel_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
        if (kernel_ms < best_kernel_ms) {
            best_kernel_ms = kernel_ms;
            best_partitions = std::move(partitions);
        }
    }

    row_partitions_ = best_partitions.empty() ? build_nnz_aware_partitions(matrix, 1) : std::move(best_partitions);
    fill_partition_stats(matrix, host_matrix_bf16_, row_partitions_, &stats_);

    const auto end = std::chrono::steady_clock::now();
    stats_.initialization_ms = std::chrono::duration<double, std::milli>(end - start).count();
    initialized_ = true;
    return true;
}

bool HostPrototypeSpmvContext::run_bf16(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error) {
    if (!initialized_) {
        if (error) {
            *error = "Host prototype context is not initialized";
        }
        return false;
    }
    if (!validate_spmv_buffers(host_matrix_bf16_.cols, x.size(), y, error)) return false;

    const auto transfer_in_start = std::chrono::steady_clock::now();
    convert_fp32_vector(x, &host_x_bf16_, float32_to_bf16_bits);
    const auto transfer_in_end = std::chrono::steady_clock::now();
    const auto kernel_start = std::chrono::steady_clock::now();
    spmv_partitioned_bf16(host_matrix_bf16_, row_partitions_, host_x_bf16_, &host_y_fp32_, true);
    const auto kernel_end = std::chrono::steady_clock::now();
    finish_backend_run(host_y_fp32_, y, stats_.initialization_ms,
                       transfer_in_start, transfer_in_end, kernel_start,
                       kernel_end, timings);
    return true;
}

std::string HostPrototypeBf16PersistentBackend::name() const {
    return "Host BF16 Persistent Prototype";
}

bool HostPrototypeBf16PersistentBackend::prepare(const CSRMatrix& matrix, std::string* error) {
    if (handle_.ready) {
        return true;
    }

    handle_.ai_core_count = host_prototype_partition_count();
    std::vector<std::int32_t> candidates = {8, 16, 32, handle_.ai_core_count};
    const bool initialized = handle_.context.initialize_bf16_autotuned(matrix, candidates, error);
    handle_.ready = initialized;
    return initialized;
}

bool HostPrototypeBf16PersistentBackend::run(const std::vector<float>& x, std::vector<float>* y, BackendTimings* timings, std::string* error) {
    return handle_.context.run_bf16(x, y, timings, error);
}

}  // namespace spmv

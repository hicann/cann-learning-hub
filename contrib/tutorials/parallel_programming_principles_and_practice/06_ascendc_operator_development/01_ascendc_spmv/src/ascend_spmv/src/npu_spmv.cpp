#include "spmv.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace spmv {

namespace {

void spmv_partitioned(const CSRMatrix& matrix, const std::vector<std::int32_t>& partitions, const std::vector<float>& x, std::vector<float>* y) {
    if (y == nullptr) {
        return;
    }
    y->assign(static_cast<std::size_t>(matrix.rows), 0.0f);
    if (partitions.size() < 2) {
        spmv_csr_reference(matrix, x, y);
        return;
    }

    for (std::size_t partition = 0; partition + 1 < partitions.size(); ++partition) {
        const std::int32_t begin_row = partitions[partition];
        const std::int32_t end_row = partitions[partition + 1];
        for (std::int32_t row = begin_row; row < end_row; ++row) {
            (*y)[static_cast<std::size_t>(row)] =
                course_sparse::csr_row_dot(matrix, x, row);
        }
    }
}

}  // namespace

std::vector<std::int32_t> build_equal_row_partitions(const CSRMatrix& matrix, std::int32_t partitions) {
    if (matrix.rows <= 0) {
        return {0};
    }
    const std::int32_t partition_count =
        std::min(std::max(partitions, std::int32_t{1}), matrix.rows);
    std::vector<std::int32_t> bounds(static_cast<std::size_t>(partition_count + 1), 0);
    const std::int32_t base = matrix.rows / std::max(partition_count, std::int32_t{1});
    const std::int32_t remainder = matrix.rows % std::max(partition_count, std::int32_t{1});
    std::int32_t row = 0;
    bounds[0] = 0;
    for (std::int32_t p = 0; p < partition_count; ++p) {
        row += base + (p < remainder ? 1 : 0);
        bounds[static_cast<std::size_t>(p + 1)] = row;
    }
    return bounds;
}

std::vector<std::int32_t> build_nnz_aware_partitions(const CSRMatrix& matrix, std::int32_t partitions) {
    if (matrix.rows <= 0) {
        return {0};
    }
    const std::int32_t partition_count =
        std::min(std::max(partitions, std::int32_t{1}), matrix.rows);
    std::vector<std::int64_t> prefix(static_cast<std::size_t>(matrix.rows + 1), 0);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        const std::int64_t row_nnz = static_cast<std::int64_t>(matrix.row_ptr[static_cast<std::size_t>(row + 1)] - matrix.row_ptr[static_cast<std::size_t>(row)]);
        prefix[static_cast<std::size_t>(row + 1)] = prefix[static_cast<std::size_t>(row)] + row_nnz;
    }

    const std::int64_t total_nnz = prefix.back();
    std::vector<std::int32_t> bounds;
    bounds.reserve(static_cast<std::size_t>(partition_count + 1));
    bounds.push_back(0);

    std::int32_t last_row = 0;
    for (std::int32_t part = 1; part < partition_count; ++part) {
        const std::int64_t target =
            (total_nnz * part) / std::max(partition_count, std::int32_t{1});
        auto it = std::lower_bound(prefix.begin() + last_row + 1, prefix.end(), target);
        std::int32_t row = static_cast<std::int32_t>(std::distance(prefix.begin(), it));
        if (row <= last_row) {
            row = std::min(matrix.rows, last_row + 1);
        }
        if (row >= matrix.rows) {
            break;
        }
        bounds.push_back(row);
        last_row = row;
    }

    while (static_cast<std::int32_t>(bounds.size()) < partition_count) {
        const std::int32_t next = std::min(matrix.rows, bounds.back() + 1);
        if (next == bounds.back()) {
            break;
        }
        bounds.push_back(next);
    }

    if (bounds.back() != matrix.rows) {
        bounds.push_back(matrix.rows);
    }

    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    if (bounds.front() != 0) {
        bounds.insert(bounds.begin(), 0);
    }
    if (bounds.back() != matrix.rows) {
        bounds.push_back(matrix.rows);
    }
    return bounds;
}

HostPrototypeBackend::HostPrototypeBackend(std::string label, bool nnz_aware)
    : label_(std::move(label)), nnz_aware_(nnz_aware) {}

std::string HostPrototypeBackend::name() const { return label_; }

bool HostPrototypeBackend::prepare(const CSRMatrix& matrix, std::string* error) {
    const auto start = std::chrono::steady_clock::now();
    host_matrix_ = matrix;
    if (!host_matrix_.validate(error)) return false;
    row_partitions_ = nnz_aware_ ? build_nnz_aware_partitions(host_matrix_, 32)
                                 : build_equal_row_partitions(host_matrix_, 32);
    initialization_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return true;
}

bool HostPrototypeBackend::run(const std::vector<float>& x, std::vector<float>* y,
                               BackendTimings* timings, std::string* error) {
    if (!validate_spmv_run(host_matrix_, x, y, error)) return false;
    const auto transfer_in_start = std::chrono::steady_clock::now();
    host_x_ = x;
    const auto transfer_in_end = std::chrono::steady_clock::now();
    const auto kernel_start = std::chrono::steady_clock::now();
    spmv_partitioned(host_matrix_, row_partitions_, host_x_, &host_y_);
    const auto kernel_end = std::chrono::steady_clock::now();
    finish_backend_run(host_y_, y, initialization_ms_, transfer_in_start,
                       transfer_in_end, kernel_start, kernel_end, timings);
    return true;
}

HostPrototypeBaselineBackend::HostPrototypeBaselineBackend()
    : HostPrototypeBackend("Host FP32 baseline prototype", false) {}

HostPrototypeOptimizedBackend::HostPrototypeOptimizedBackend()
    : HostPrototypeBackend("Host optimized FP32 prototype", true) {}

}  // namespace spmv

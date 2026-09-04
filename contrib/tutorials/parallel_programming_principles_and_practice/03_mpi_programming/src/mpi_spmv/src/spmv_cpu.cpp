#include "spmv.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mpi_spmv {

void spmv_local(const std::vector<std::int32_t>& row_ptr,
                const std::vector<std::int32_t>& col_idx,
                const std::vector<float>& values,
                const std::vector<float>& x,
                std::vector<float>* y) {
    if (y == nullptr || row_ptr.empty()) return;
    y->assign(row_ptr.size() - 1, 0.0f);
    for (std::size_t row = 0; row + 1 < row_ptr.size(); ++row) {
        float sum = 0.0f;
        for (std::int32_t index = row_ptr[row]; index < row_ptr[row + 1]; ++index) {
            const std::size_t offset = static_cast<std::size_t>(index);
            sum += values[offset] * x[static_cast<std::size_t>(col_idx[offset])];
        }
        (*y)[row] = sum;
    }
}

void spmv_cpu(const CSRMatrix& matrix, const std::vector<float>& x, std::vector<float>* y) {
    spmv_local(matrix.row_ptr, matrix.col_idx, matrix.values, x, y);
}

double max_absolute_error(const std::vector<float>& reference,
                          const std::vector<float>& candidate) {
    if (reference.size() != candidate.size()) return std::numeric_limits<double>::infinity();
    double error = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        error = std::max(error, std::abs(static_cast<double>(reference[i]) - candidate[i]));
    }
    return error;
}

double l2_error(const std::vector<float>& reference,
                const std::vector<float>& candidate) {
    if (reference.size() != candidate.size()) return std::numeric_limits<double>::infinity();
    long double squared_sum = 0.0L;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const long double difference = static_cast<long double>(reference[i]) - candidate[i];
        squared_sum += difference * difference;
    }
    return std::sqrt(static_cast<double>(squared_sum));
}

}  // namespace mpi_spmv

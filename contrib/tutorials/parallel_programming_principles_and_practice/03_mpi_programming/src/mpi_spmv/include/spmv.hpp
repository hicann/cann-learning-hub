#pragma once

#include "csr.hpp"

#include <cstdint>
#include <vector>

namespace mpi_spmv {

void spmv_cpu(const CSRMatrix& matrix, const std::vector<float>& x, std::vector<float>* y);
void spmv_local(const std::vector<std::int32_t>& row_ptr,
                const std::vector<std::int32_t>& col_idx,
                const std::vector<float>& values,
                const std::vector<float>& x,
                std::vector<float>* y);

double max_absolute_error(const std::vector<float>& reference,
                          const std::vector<float>& candidate);
double l2_error(const std::vector<float>& reference,
                const std::vector<float>& candidate);

}  // namespace mpi_spmv

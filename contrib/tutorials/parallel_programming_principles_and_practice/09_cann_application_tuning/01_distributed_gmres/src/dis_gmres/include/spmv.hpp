#pragma once

#include "csr_matrix.hpp"

#include <cstddef>
#include <vector>

namespace dis_gmres {

void csr_spmv(const CSRMatrix& local_matrix, const std::vector<float>& global_x,
              std::vector<float>* local_y, bool parallel = true);
double local_dot(const std::vector<float>& x, const std::vector<float>& y, bool parallel = true);
void local_axpy(float alpha, const std::vector<float>& x, std::vector<float>* y, bool parallel = true);
void local_scale(float alpha, std::vector<float>* x, bool parallel = true);
void local_multi_dot(const std::vector<float>& x,
                     const std::vector<std::vector<float>>& basis,
                     std::size_t count, std::vector<float>* coefficients,
                     bool parallel = true);
void local_fused_axpy(float alpha, const std::vector<float>& coefficients,
                      const std::vector<std::vector<float>>& basis,
                      std::size_t count, std::vector<float>* y,
                      bool parallel = true);
int max_compute_threads();
std::size_t omp_min_elements();

}  // namespace dis_gmres

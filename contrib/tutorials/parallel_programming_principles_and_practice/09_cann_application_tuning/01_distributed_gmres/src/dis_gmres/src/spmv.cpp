#include "spmv.hpp"

#include "profiler.hpp"

#include <algorithm>
#include <cstdlib>

#if DIS_GMRES_HAS_OPENMP
#include <omp.h>
#endif

namespace dis_gmres {
namespace {

bool use_openmp(bool requested, std::size_t work_items) {
#if DIS_GMRES_HAS_OPENMP
  return requested && omp_get_max_threads() > 1 && work_items >= omp_min_elements();
#else
  (void)requested;
  (void)work_items;
  return false;
#endif
}

}  // namespace

int max_compute_threads() {
#if DIS_GMRES_HAS_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

std::size_t omp_min_elements() {
  static const std::size_t threshold = [] {
    const char* configured = std::getenv("DIS_GMRES_OMP_MIN_ELEMENTS");
    if (!configured || *configured == '\0') return std::size_t{262144};
    try {
      return static_cast<std::size_t>(std::stoull(configured));
    } catch (...) {
      return std::size_t{262144};
    }
  }();
  return threshold;
}

void csr_spmv(const CSRMatrix& matrix, const std::vector<float>& x,
              std::vector<float>* y, bool parallel) {
  if (!y) return;
  y->assign(static_cast<std::size_t>(matrix.rows), 0.0f);
  const bool threaded = use_openmp(parallel, static_cast<std::size_t>(matrix.nnz()));
  if (!threaded) {
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
      float sum = 0.0f;
      for (auto index = matrix.row_ptr[static_cast<std::size_t>(row)];
           index < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++index)
        sum += matrix.values[static_cast<std::size_t>(index)] *
               x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(index)])];
      (*y)[static_cast<std::size_t>(row)] = sum;
    }
    return;
  }
#if DIS_GMRES_HAS_OPENMP
#pragma omp parallel for schedule(guided, 64)
#else
  (void)parallel;
#endif
  for (std::int32_t row = 0; row < matrix.rows; ++row) {
    float sum = 0.0f;
    for (auto index = matrix.row_ptr[static_cast<std::size_t>(row)];
         index < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++index)
      sum += matrix.values[static_cast<std::size_t>(index)] *
             x[static_cast<std::size_t>(matrix.col_idx[static_cast<std::size_t>(index)])];
    (*y)[static_cast<std::size_t>(row)] = sum;
  }
}

double local_dot(const std::vector<float>& x, const std::vector<float>& y, bool parallel) {
  const auto size = std::min(x.size(), y.size());
  if (!use_openmp(parallel, size)) {
    double sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) sum += static_cast<double>(x[i]) * y[i];
    return sum;
  }
  double sum = 0.0;
#if DIS_GMRES_HAS_OPENMP
#pragma omp parallel for reduction(+ : sum) schedule(static)
#else
  (void)parallel;
#endif
  for (std::size_t i = 0; i < size; ++i) sum += static_cast<double>(x[i]) * y[i];
  return sum;
}

void local_axpy(float alpha, const std::vector<float>& x, std::vector<float>* y, bool parallel) {
  if (!y) return;
  const auto size = std::min(x.size(), y->size());
  if (!use_openmp(parallel, size)) {
    for (std::size_t i = 0; i < size; ++i) (*y)[i] += alpha * x[i];
    return;
  }
#if DIS_GMRES_HAS_OPENMP
#pragma omp parallel for schedule(static)
#else
  (void)parallel;
#endif
  for (std::size_t i = 0; i < size; ++i) (*y)[i] += alpha * x[i];
}

void local_scale(float alpha, std::vector<float>* x, bool parallel) {
  if (!x) return;
  if (!use_openmp(parallel, x->size())) {
    for (auto& value : *x) value *= alpha;
    return;
  }
#if DIS_GMRES_HAS_OPENMP
#pragma omp parallel for schedule(static)
#else
  (void)parallel;
#endif
  for (std::size_t i = 0; i < x->size(); ++i) (*x)[i] *= alpha;
}

void local_multi_dot(const std::vector<float>& x,
                     const std::vector<std::vector<float>>& basis,
                     std::size_t count, std::vector<float>* coefficients,
                     bool parallel) {
  if (!coefficients) return;
  count = std::min(count, basis.size());
  coefficients->assign(count, 0.0f);
  if (count == 0) return;
  std::size_t size = x.size();
  for (std::size_t column = 0; column < count; ++column)
    size = std::min(size, basis[column].size());
  if (!use_openmp(parallel, size)) {
    for (std::size_t column = 0; column < count; ++column) {
      double sum = 0.0;
      for (std::size_t index = 0; index < size; ++index)
        sum += static_cast<double>(x[index]) * basis[column][index];
      (*coefficients)[column] = static_cast<float>(sum);
    }
    return;
  }
#if DIS_GMRES_HAS_OPENMP
  const int thread_count = omp_get_max_threads();
  std::vector<double> partial(static_cast<std::size_t>(thread_count) * count, 0.0);
#pragma omp parallel
  {
    const int thread = omp_get_thread_num();
    double* local = partial.data() + static_cast<std::size_t>(thread) * count;
#pragma omp for schedule(static)
    for (std::size_t index = 0; index < size; ++index) {
      const double value = x[index];
      for (std::size_t column = 0; column < count; ++column)
        local[column] += value * basis[column][index];
    }
  }
  for (std::size_t column = 0; column < count; ++column) {
    double sum = 0.0;
    for (int thread = 0; thread < thread_count; ++thread)
      sum += partial[static_cast<std::size_t>(thread) * count + column];
    (*coefficients)[column] = static_cast<float>(sum);
  }
#else
  (void)parallel;
#endif
}

void local_fused_axpy(float alpha, const std::vector<float>& coefficients,
                      const std::vector<std::vector<float>>& basis,
                      std::size_t count, std::vector<float>* y, bool parallel) {
  if (!y) return;
  count = std::min({count, coefficients.size(), basis.size()});
  if (count == 0) return;
  std::size_t size = y->size();
  for (std::size_t column = 0; column < count; ++column)
    size = std::min(size, basis[column].size());
  if (!use_openmp(parallel, size)) {
    for (std::size_t column = 0; column < count; ++column) {
      const float scaled = alpha * coefficients[column];
      for (std::size_t index = 0; index < size; ++index)
        (*y)[index] += scaled * basis[column][index];
    }
    return;
  }
#if DIS_GMRES_HAS_OPENMP
#pragma omp parallel for schedule(static)
#else
  (void)parallel;
#endif
  for (std::size_t index = 0; index < size; ++index) {
    float sum = 0.0f;
    for (std::size_t column = 0; column < count; ++column)
      sum += coefficients[column] * basis[column][index];
    (*y)[index] += alpha * sum;
  }
}

void Profile::accumulate(const Profile& other) {
  total_ms += other.total_ms;
  spmv_ms += other.spmv_ms;
  dot_ms += other.dot_ms;
  axpy_ms += other.axpy_ms;
  norm_ms += other.norm_ms;
  givens_ms += other.givens_ms;
  communication_ms += other.communication_ms;
  transfer_ms += other.transfer_ms;
  kernel_launch_ms += other.kernel_launch_ms;
  synchronization_ms += other.synchronization_ms;
  allreduce_calls += other.allreduce_calls;
  allgather_calls += other.allgather_calls;
}

void Profile::scale(double factor) {
  total_ms *= factor;
  spmv_ms *= factor;
  dot_ms *= factor;
  axpy_ms *= factor;
  norm_ms *= factor;
  givens_ms *= factor;
  communication_ms *= factor;
  transfer_ms *= factor;
  kernel_launch_ms *= factor;
  synchronization_ms *= factor;
}

}  // namespace dis_gmres

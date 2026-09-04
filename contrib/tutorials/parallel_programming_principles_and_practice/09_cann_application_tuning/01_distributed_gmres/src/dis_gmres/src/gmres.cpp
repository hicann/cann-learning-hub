#include "gmres.hpp"

#include "spmv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace dis_gmres {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}
std::size_t h_index(std::int32_t row, std::int32_t col, std::int32_t restart) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(restart) +
         static_cast<std::size_t>(col);
}
void apply_rotation(float* x, float* y, float cosine, float sine) {
  const float temporary = cosine * *x + sine * *y;
  *y = -sine * *x + cosine * *y;
  *x = temporary;
}
void make_rotation(float x, float y, float* cosine, float* sine) {
  const float magnitude = std::hypot(x, y);
  if (magnitude == 0.0f) {
    *cosine = 1.0f;
    *sine = 0.0f;
  } else {
    *cosine = x / magnitude;
    *sine = y / magnitude;
  }
}

float normalized_residual(float residual_norm, float right_hand_side_norm) {
  return residual_norm / std::max(right_hand_side_norm, 1.0e-30f);
}

bool global_sum(HcclCommunicator* comm, double local, float* global, std::string* error) {
  return comm->allreduce_sum(static_cast<float>(local), global, error);
}

bool gather_and_spmv(const CSRMatrix& matrix, const std::vector<RowPartition>& partitions,
                     const std::vector<float>& local_x, std::vector<float>* local_y,
                     HcclCommunicator* comm, bool parallel, Profile* profile,
                     std::string* error) {
  static thread_local std::vector<float> global_x;
  if (!comm->allgather_vector(local_x, partitions, &global_x, error)) return false;
  const auto begin = Clock::now();
  csr_spmv(matrix, global_x, local_y, parallel);
  const auto end = Clock::now();
  if (profile) profile->spmv_ms += ms(begin, end);
  return true;
}

bool global_norm(const std::vector<float>& local, HcclCommunicator* comm, bool parallel,
                 float* norm, Profile* profile, std::string* error) {
  const auto begin = Clock::now();
  const double local_squared = local_dot(local, local, parallel);
  const auto end = Clock::now();
  if (profile) profile->norm_ms += ms(begin, end);
  float global_squared = 0.0f;
  if (!global_sum(comm, local_squared, &global_squared, error)) return false;
  *norm = std::sqrt(std::max(0.0f, global_squared));
  return true;
}

void update_solution(std::vector<float>* x, const std::vector<std::vector<float>>& basis,
                     const std::vector<float>& h, const std::vector<float>& g,
                     std::int32_t dimension, std::int32_t restart, bool parallel,
                     bool fused_vector_ops, Profile* profile) {
  auto begin = Clock::now();
  std::vector<float> y(static_cast<std::size_t>(dimension), 0.0f);
  for (std::int32_t row = dimension - 1; row >= 0; --row) {
    float value = g[static_cast<std::size_t>(row)];
    for (std::int32_t col = row + 1; col < dimension; ++col)
      value -= h[h_index(row, col, restart)] * y[static_cast<std::size_t>(col)];
    const float diagonal = h[h_index(row, row, restart)];
    y[static_cast<std::size_t>(row)] =
        std::fabs(diagonal) > 1.0e-30f ? value / diagonal : 0.0f;
  }
  auto end = Clock::now();
  if (profile) profile->givens_ms += ms(begin, end);
  if (fused_vector_ops) {
    begin = Clock::now();
    local_fused_axpy(1.0f, y, basis, static_cast<std::size_t>(dimension), x, parallel);
    end = Clock::now();
    if (profile) profile->axpy_ms += ms(begin, end);
  } else {
    for (std::int32_t col = 0; col < dimension; ++col) {
      begin = Clock::now();
      local_axpy(y[static_cast<std::size_t>(col)], basis[static_cast<std::size_t>(col)], x,
                 parallel);
      end = Clock::now();
      if (profile) profile->axpy_ms += ms(begin, end);
    }
  }
}

float relative_residual_with_norm(const CSRMatrix& local_matrix,
                                  const std::vector<RowPartition>& partitions,
                                  const std::vector<float>& local_b,
                                  const std::vector<float>& local_x,
                                  float b_norm, bool parallel,
                                  HcclCommunicator* communicator, Profile* profile,
                                  std::string* error) {
  std::vector<float> ax;
  if (!gather_and_spmv(local_matrix, partitions, local_x, &ax, communicator, parallel,
                       profile, error))
    return std::numeric_limits<float>::infinity();
  std::vector<float> residual(local_b);
  const auto axpy_begin = Clock::now();
  local_axpy(-1.0f, ax, &residual, parallel);
  const auto axpy_end = Clock::now();
  if (profile) profile->axpy_ms += ms(axpy_begin, axpy_end);
  float residual_norm = 0.0f;
  if (!global_norm(residual, communicator, parallel, &residual_norm, profile, error))
    return std::numeric_limits<float>::infinity();
  return normalized_residual(residual_norm, b_norm);
}

struct HostWorkspace {
  explicit HostWorkspace(std::int32_t restart, std::size_t local_size)
      : basis(static_cast<std::size_t>(restart + 1), std::vector<float>(local_size)),
        h(static_cast<std::size_t>(restart + 1) * restart, 0.0f),
        cosine(static_cast<std::size_t>(restart)),
        sine(static_cast<std::size_t>(restart)),
        g(static_cast<std::size_t>(restart + 1)) {}

  std::vector<std::vector<float>> basis;
  std::vector<float> h;
  std::vector<float> cosine;
  std::vector<float> sine;
  std::vector<float> g;
  bool first_cycle = true;
  float b_norm = 0.0f;
};

enum class CycleStatus { ready, finished, failed };

CycleStatus prepare_cycle_residual(
    const CSRMatrix& matrix, const std::vector<RowPartition>& partitions,
    const std::vector<float>& local_b, const std::vector<float>& local_x,
    HcclCommunicator* communicator, const GmresOptions& options,
    HostWorkspace* workspace, GmresResult* result,
    std::vector<float>* residual, float* beta, std::string* error) {
  *beta = workspace->b_norm;
  if (workspace->first_cycle && options.zero_initial_guess) {
    *residual = local_b;
  } else {
    std::vector<float> ax;
    if (!gather_and_spmv(matrix, partitions, local_x, &ax, communicator,
                         options.parallel_compute, &result->profile, error)) {
      return CycleStatus::failed;
    }
    *residual = local_b;
    const auto begin = Clock::now();
    local_axpy(-1.0f, ax, residual, options.parallel_compute);
    const auto end = Clock::now();
    result->profile.axpy_ms += ms(begin, end);
    if (!global_norm(*residual, communicator, options.parallel_compute, beta,
                     &result->profile, error)) return CycleStatus::failed;
  }
  workspace->first_cycle = false;
  result->residual = normalized_residual(*beta, workspace->b_norm);
  if (result->residual <= options.tolerance) {
    result->converged = true;
    return CycleStatus::finished;
  }
  if (*beta <= 1.0e-30f) {
    result->converged = true;
    result->residual = 0.0f;
    return CycleStatus::finished;
  }
  return CycleStatus::ready;
}

void initialize_cycle(std::vector<float> residual, float beta,
                      const GmresOptions& options, HostWorkspace* workspace,
                      Profile* profile) {
  workspace->basis[0] = std::move(residual);
  const auto begin = Clock::now();
  local_scale(1.0f / std::max(beta, 1.0e-30f), &workspace->basis[0],
              options.parallel_compute);
  const auto end = Clock::now();
  profile->axpy_ms += ms(begin, end);
  std::fill(workspace->h.begin(), workspace->h.end(), 0.0f);
  std::fill(workspace->g.begin(), workspace->g.end(), 0.0f);
  workspace->g[0] = beta;
}

bool orthogonalize_cgs(std::int32_t column, std::int32_t restart,
                       HcclCommunicator* communicator,
                       const GmresOptions& options, HostWorkspace* workspace,
                       Profile* profile, std::string* error) {
  auto& w = workspace->basis[static_cast<std::size_t>(column + 1)];
  std::vector<float> local_coefficients(static_cast<std::size_t>(column + 1));
  const auto dot_begin = Clock::now();
  if (options.fused_vector_ops) {
    local_multi_dot(w, workspace->basis, static_cast<std::size_t>(column + 1),
                    &local_coefficients, options.parallel_compute);
  } else {
    for (std::int32_t row = 0; row <= column; ++row) {
      local_coefficients[static_cast<std::size_t>(row)] = static_cast<float>(
          local_dot(w, workspace->basis[static_cast<std::size_t>(row)],
                    options.parallel_compute));
    }
  }
  profile->dot_ms += ms(dot_begin, Clock::now());
  std::vector<float> global_coefficients;
  if (!communicator->allreduce_sum(local_coefficients, &global_coefficients, error)) {
    return false;
  }
  for (std::int32_t row = 0; row <= column; ++row) {
    workspace->h[h_index(row, column, restart)] =
        global_coefficients[static_cast<std::size_t>(row)];
  }
  const auto axpy_begin = Clock::now();
  if (options.fused_vector_ops) {
    local_fused_axpy(-1.0f, global_coefficients, workspace->basis,
                     static_cast<std::size_t>(column + 1), &w,
                     options.parallel_compute);
  } else {
    for (std::int32_t row = 0; row <= column; ++row) {
      local_axpy(-global_coefficients[static_cast<std::size_t>(row)],
                 workspace->basis[static_cast<std::size_t>(row)], &w,
                 options.parallel_compute);
    }
  }
  profile->axpy_ms += ms(axpy_begin, Clock::now());
  return true;
}

bool orthogonalize_mgs(std::int32_t column, std::int32_t restart,
                       HcclCommunicator* communicator,
                       const GmresOptions& options, HostWorkspace* workspace,
                       Profile* profile, std::string* error) {
  auto& w = workspace->basis[static_cast<std::size_t>(column + 1)];
  for (std::int32_t row = 0; row <= column; ++row) {
    const auto dot_begin = Clock::now();
    const double local_coefficient = local_dot(
        w, workspace->basis[static_cast<std::size_t>(row)],
        options.parallel_compute);
    profile->dot_ms += ms(dot_begin, Clock::now());
    float coefficient = 0.0f;
    if (!global_sum(communicator, local_coefficient, &coefficient, error)) return false;
    workspace->h[h_index(row, column, restart)] = coefficient;
    const auto axpy_begin = Clock::now();
    local_axpy(-coefficient, workspace->basis[static_cast<std::size_t>(row)],
               &w, options.parallel_compute);
    profile->axpy_ms += ms(axpy_begin, Clock::now());
  }
  return true;
}

bool orthogonalize(std::int32_t column, std::int32_t restart,
                   HcclCommunicator* communicator,
                   const GmresOptions& options, HostWorkspace* workspace,
                   Profile* profile, std::string* error) {
  if (options.communication_avoiding) {
    return orthogonalize_cgs(column, restart, communicator, options,
                             workspace, profile, error);
  }
  return orthogonalize_mgs(column, restart, communicator, options,
                           workspace, profile, error);
}

bool normalize_next_basis(std::int32_t column, std::int32_t restart,
                          HcclCommunicator* communicator,
                          const GmresOptions& options, HostWorkspace* workspace,
                          Profile* profile, float* next_norm,
                          std::string* error) {
  auto& w = workspace->basis[static_cast<std::size_t>(column + 1)];
  if (!global_norm(w, communicator, options.parallel_compute, next_norm,
                   profile, error)) return false;
  workspace->h[h_index(column + 1, column, restart)] = *next_norm;
  if (*next_norm <= 1.0e-30f) return true;
  const auto begin = Clock::now();
  local_scale(1.0f / *next_norm, &w, options.parallel_compute);
  profile->axpy_ms += ms(begin, Clock::now());
  return true;
}

void apply_givens_column(std::int32_t column, std::int32_t restart,
                         HostWorkspace* workspace, Profile* profile) {
  const auto begin = Clock::now();
  for (std::int32_t row = 0; row < column; ++row) {
    apply_rotation(&workspace->h[h_index(row, column, restart)],
                   &workspace->h[h_index(row + 1, column, restart)],
                   workspace->cosine[static_cast<std::size_t>(row)],
                   workspace->sine[static_cast<std::size_t>(row)]);
  }
  make_rotation(workspace->h[h_index(column, column, restart)],
                workspace->h[h_index(column + 1, column, restart)],
                &workspace->cosine[static_cast<std::size_t>(column)],
                &workspace->sine[static_cast<std::size_t>(column)]);
  apply_rotation(&workspace->h[h_index(column, column, restart)],
                 &workspace->h[h_index(column + 1, column, restart)],
                 workspace->cosine[static_cast<std::size_t>(column)],
                 workspace->sine[static_cast<std::size_t>(column)]);
  apply_rotation(&workspace->g[static_cast<std::size_t>(column)],
                 &workspace->g[static_cast<std::size_t>(column + 1)],
                 workspace->cosine[static_cast<std::size_t>(column)],
                 workspace->sine[static_cast<std::size_t>(column)]);
  profile->givens_ms += ms(begin, Clock::now());
}

bool arnoldi_step(const CSRMatrix& matrix,
                  const std::vector<RowPartition>& partitions,
                  HcclCommunicator* communicator, const GmresOptions& options,
                  std::int32_t restart, std::int32_t column,
                  HostWorkspace* workspace, GmresResult* result,
                  bool* stop, std::string* error) {
  auto& w = workspace->basis[static_cast<std::size_t>(column + 1)];
  if (!gather_and_spmv(matrix, partitions,
                       workspace->basis[static_cast<std::size_t>(column)], &w,
                       communicator, options.parallel_compute,
                       &result->profile, error)) return false;
  if (!orthogonalize(column, restart, communicator, options, workspace,
                     &result->profile, error)) return false;
  float next_norm = 0.0f;
  if (!normalize_next_basis(column, restart, communicator, options, workspace,
                            &result->profile, &next_norm, error)) return false;
  apply_givens_column(column, restart, workspace, &result->profile);
  ++result->iterations;
  result->residual = normalized_residual(
      std::fabs(workspace->g[static_cast<std::size_t>(column + 1)]),
      workspace->b_norm);
  *stop = result->residual <= options.tolerance || next_norm <= 1.0e-30f;
  return true;
}

CycleStatus run_cycle(const CSRMatrix& matrix,
                      const std::vector<RowPartition>& partitions,
                      const std::vector<float>& local_b,
                      std::vector<float>* local_x, HcclCommunicator* communicator,
                      const GmresOptions& options, std::int32_t restart,
                      HostWorkspace* workspace, GmresResult* result,
                      std::string* error) {
  std::vector<float> residual;
  float beta = workspace->b_norm;
  const CycleStatus status = prepare_cycle_residual(
      matrix, partitions, local_b, *local_x, communicator, options, workspace,
      result, &residual, &beta, error);
  if (status != CycleStatus::ready) return status;
  initialize_cycle(std::move(residual), beta, options, workspace, &result->profile);
  std::int32_t used = 0;
  for (std::int32_t column = 0;
       column < restart && result->iterations < options.max_iterations; ++column) {
    bool stop = false;
    if (!arnoldi_step(matrix, partitions, communicator, options, restart, column,
                      workspace, result, &stop, error)) return CycleStatus::failed;
    used = column + 1;
    if (stop) break;
  }
  update_solution(local_x, workspace->basis, workspace->h, workspace->g,
                  used, restart, options.parallel_compute,
                  options.fused_vector_ops, &result->profile);
  result->residual = relative_residual_with_norm(
      matrix, partitions, local_b, *local_x, workspace->b_norm,
      options.parallel_compute, communicator, &result->profile, error);
  if (!std::isfinite(result->residual)) return CycleStatus::failed;
  if (result->residual <= options.tolerance) {
    result->converged = true;
    return CycleStatus::finished;
  }
  return CycleStatus::ready;
}

void copy_comm_stats(const CommStats& stats, Profile* profile) {
  profile->communication_ms = stats.collective_ms;
  profile->transfer_ms = stats.transfer_ms;
  profile->synchronization_ms = stats.synchronization_ms;
  profile->allreduce_calls = stats.allreduce_calls;
  profile->allgather_calls = stats.allgather_calls;
}

}  // namespace

float distributed_relative_residual(const CSRMatrix& local_matrix,
                                    const std::vector<RowPartition>& partitions,
                                    const std::vector<float>& local_b,
                                    const std::vector<float>& local_x,
                                    HcclCommunicator* communicator, Profile* profile,
                                    std::string* error) {
  float b_norm = 0.0f;
  if (!global_norm(local_b, communicator, true, &b_norm, profile, error))
    return std::numeric_limits<float>::infinity();
  return relative_residual_with_norm(local_matrix, partitions, local_b, local_x, b_norm,
                                     true, communicator, profile, error);
}

GmresResult distributed_gmres(const CSRMatrix& local_matrix,
                               const std::vector<RowPartition>& partitions,
                               const std::vector<float>& local_b,
                               std::vector<float>* local_x,
                               HcclCommunicator* communicator,
                               const GmresOptions& options, std::string* error) {
#if DIS_GMRES_HAS_CANN
  if (communicator && communicator->real_hccl()) {
    return distributed_gmres_npu(local_matrix, partitions, local_b, local_x,
                                 communicator, options, error);
  }
#endif
  GmresResult result;
  if (!local_x || !communicator ||
      local_b.size() != static_cast<std::size_t>(local_matrix.rows)) {
    if (error) *error = "invalid local GMRES inputs";
    return result;
  }
  if (local_x->size() != local_b.size()) local_x->assign(local_b.size(), 0.0f);
  const auto total_begin = Clock::now();
  communicator->reset_stats();
  const auto restart = std::max<std::int32_t>(1, options.restart);
  HostWorkspace workspace(restart, local_b.size());
  if (!global_norm(local_b, communicator, options.parallel_compute,
                   &workspace.b_norm, &result.profile, error)) return result;
  while (result.iterations < options.max_iterations) {
    const CycleStatus status = run_cycle(
        local_matrix, partitions, local_b, local_x, communicator, options,
        restart, &workspace, &result, error);
    if (status == CycleStatus::failed) return result;
    if (status == CycleStatus::finished) break;
  }
  result.profile.total_ms = ms(total_begin, Clock::now());
  copy_comm_stats(communicator->stats(), &result.profile);
  return result;
}

}  // namespace dis_gmres

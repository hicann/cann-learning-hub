#include "gmres.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <utility>

namespace ascend_gmres {

namespace {

std::size_t h_index(std::int32_t row, std::int32_t col, std::int32_t restart) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(restart) + static_cast<std::size_t>(col);
}

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void apply_plane_rotation(float* dx, float* dy, float cs, float sn) {
    const float temp = cs * (*dx) + sn * (*dy);
    *dy = -sn * (*dx) + cs * (*dy);
    *dx = temp;
}

void generate_plane_rotation(float dx, float dy, float* cs, float* sn) {
    const float magnitude = std::hypot(dx, dy);
    if (magnitude == 0.0f) {
        *cs = 1.0f;
        *sn = 0.0f;
    } else {
        *cs = dx / magnitude;
        *sn = dy / magnitude;
    }
}

float normalized_residual(float residual_norm, float right_hand_side_norm) {
    return residual_norm / std::max(right_hand_side_norm, 1.0e-30f);
}

std::vector<float> solve_upper_triangular(const std::vector<float>& h, const std::vector<float>& g, std::int32_t dim, std::int32_t restart) {
    std::vector<float> y(static_cast<std::size_t>(dim), 0.0f);
    for (std::int32_t i = dim - 1; i >= 0; --i) {
        float sum = g[static_cast<std::size_t>(i)];
        for (std::int32_t j = i + 1; j < dim; ++j) {
            sum -= h[h_index(i, j, restart)] * y[static_cast<std::size_t>(j)];
        }
        const float diag = h[h_index(i, i, restart)];
        y[static_cast<std::size_t>(i)] = std::fabs(diag) > 1.0e-30f ? sum / diag : 0.0f;
    }
    return y;
}

void update_solution(std::vector<float>* x, const std::vector<std::vector<float>>& v, const std::vector<float>& h, const std::vector<float>& g, std::int32_t dim, std::int32_t restart, IBlasBackend& blas, GMRESProfiler* profiler) {
    const auto solve_start = Clock::now();
    const auto y = solve_upper_triangular(h, g, dim, restart);
    const auto solve_end = Clock::now();
    if (profiler != nullptr) {
        profiler->givens_ms += elapsed_ms(solve_start, solve_end);
    }
    for (std::int32_t i = 0; i < dim; ++i) {
        const auto axpy_start = Clock::now();
        blas.axpy(y[static_cast<std::size_t>(i)], v[static_cast<std::size_t>(i)], x);
        const auto axpy_end = Clock::now();
        if (profiler != nullptr) {
            profiler->axpy_ms += elapsed_ms(axpy_start, axpy_end);
        }
    }
}

float compute_relative_residual_profiled(ISpmvBackend& spmv, IBlasBackend& blas, const std::vector<float>& b, const std::vector<float>& x, GMRESProfiler* profiler) {
    const auto start = Clock::now();
    const float residual = compute_relative_residual(spmv, blas, b, x);
    const auto end = Clock::now();
    if (profiler != nullptr) {
        profiler->residual_ms += elapsed_ms(start, end);
    }
    return residual;
}

struct SolverWorkspace {
    SolverWorkspace(std::int32_t restart, std::size_t size)
        : v(static_cast<std::size_t>(restart + 1), std::vector<float>(size, 0.0f)),
          h(static_cast<std::size_t>(restart + 1) * restart, 0.0f),
          cs(static_cast<std::size_t>(restart), 0.0f),
          sn(static_cast<std::size_t>(restart), 0.0f),
          g(static_cast<std::size_t>(restart + 1), 0.0f) {}

    std::vector<std::vector<float>> v;
    std::vector<float> h;
    std::vector<float> cs;
    std::vector<float> sn;
    std::vector<float> g;
    std::vector<float> ax;
    std::vector<float> r;
    std::vector<float> w;
};

enum class CycleStatus { ready, finished, failed };

CycleStatus prepare_cycle(ISpmvBackend& spmv, IBlasBackend& blas,
                          const std::vector<float>& b, const std::vector<float>& x,
                          float b_norm, const GmresOptions& options,
                          SolverWorkspace* workspace, GmresResult* result,
                          float* relative_residual, std::string* error) {
    const auto begin = Clock::now();
    if (!spmv.multiply(x, &workspace->ax, error)) return CycleStatus::failed;
    workspace->r = b;
    blas.axpy(-1.0f, workspace->ax, &workspace->r);
    const float beta = blas.norm2(workspace->r);
    *relative_residual = normalized_residual(beta, b_norm);
    result->profiler.residual_ms += elapsed_ms(begin, Clock::now());
    if (*relative_residual < options.tolerance) {
        result->converged = true;
        return CycleStatus::finished;
    }
    if (beta <= 1.0e-30f) {
        result->converged = true;
        *relative_residual = 0.0f;
        return CycleStatus::finished;
    }
    blas.copy(workspace->r, &workspace->v[0]);
    const auto normalize_begin = Clock::now();
    blas.scal(1.0f / std::max(beta, 1.0e-30f), &workspace->v[0]);
    result->profiler.axpy_ms += elapsed_ms(normalize_begin, Clock::now());
    std::fill(workspace->h.begin(), workspace->h.end(), 0.0f);
    std::fill(workspace->g.begin(), workspace->g.end(), 0.0f);
    workspace->g[0] = beta;
    return CycleStatus::ready;
}

void orthogonalize_column(std::int32_t column, std::int32_t restart,
                          IBlasBackend& blas, SolverWorkspace* workspace,
                          GMRESProfiler* profiler) {
    for (std::int32_t row = 0; row <= column; ++row) {
        const auto dot_begin = Clock::now();
        workspace->h[h_index(row, column, restart)] =
            blas.dot(workspace->w, workspace->v[static_cast<std::size_t>(row)]);
        profiler->dot_ms += elapsed_ms(dot_begin, Clock::now());
        const auto axpy_begin = Clock::now();
        blas.axpy(-workspace->h[h_index(row, column, restart)],
                  workspace->v[static_cast<std::size_t>(row)], &workspace->w);
        profiler->axpy_ms += elapsed_ms(axpy_begin, Clock::now());
    }
}

float normalize_column(std::int32_t column, std::int32_t restart,
                       IBlasBackend& blas, SolverWorkspace* workspace,
                       GMRESProfiler* profiler) {
    const auto norm_begin = Clock::now();
    const float norm = blas.norm2(workspace->w);
    workspace->h[h_index(column + 1, column, restart)] = norm;
    profiler->norm_ms += elapsed_ms(norm_begin, Clock::now());
    if (norm == 0.0f) return norm;
    if (norm <= 1.0e-30f) return norm;
    blas.copy(workspace->w, &workspace->v[static_cast<std::size_t>(column + 1)]);
    const auto scale_begin = Clock::now();
    blas.scal(1.0f / norm, &workspace->v[static_cast<std::size_t>(column + 1)]);
    profiler->axpy_ms += elapsed_ms(scale_begin, Clock::now());
    return norm;
}

void rotate_column(std::int32_t column, std::int32_t restart,
                   SolverWorkspace* workspace, GMRESProfiler* profiler) {
    const auto begin = Clock::now();
    for (std::int32_t row = 0; row < column; ++row) {
        apply_plane_rotation(&workspace->h[h_index(row, column, restart)],
                             &workspace->h[h_index(row + 1, column, restart)],
                             workspace->cs[static_cast<std::size_t>(row)],
                             workspace->sn[static_cast<std::size_t>(row)]);
    }
    generate_plane_rotation(
        workspace->h[h_index(column, column, restart)],
        workspace->h[h_index(column + 1, column, restart)],
        &workspace->cs[static_cast<std::size_t>(column)],
        &workspace->sn[static_cast<std::size_t>(column)]);
    apply_plane_rotation(
        &workspace->h[h_index(column, column, restart)],
        &workspace->h[h_index(column + 1, column, restart)],
        workspace->cs[static_cast<std::size_t>(column)],
        workspace->sn[static_cast<std::size_t>(column)]);
    apply_plane_rotation(&workspace->g[static_cast<std::size_t>(column)],
                         &workspace->g[static_cast<std::size_t>(column + 1)],
                         workspace->cs[static_cast<std::size_t>(column)],
                         workspace->sn[static_cast<std::size_t>(column)]);
    profiler->givens_ms += elapsed_ms(begin, Clock::now());
}

bool arnoldi_step(std::int32_t column, std::int32_t restart, float b_norm,
                  ISpmvBackend& spmv, IBlasBackend& blas,
                  const GmresOptions& options, SolverWorkspace* workspace,
                  GmresResult* result, std::vector<float>* x,
                  std::int32_t* total_iterations, float* relative_residual,
                  bool* stop, std::string* error) {
    const auto spmv_begin = Clock::now();
    if (!spmv.multiply(workspace->v[static_cast<std::size_t>(column)],
                       &workspace->w, error)) return false;
    result->profiler.spmv_ms += elapsed_ms(spmv_begin, Clock::now());
    orthogonalize_column(column, restart, blas, workspace, &result->profiler);
    normalize_column(column, restart, blas, workspace, &result->profiler);
    rotate_column(column, restart, workspace, &result->profiler);
    ++*total_iterations;
    *relative_residual = normalized_residual(
        std::fabs(workspace->g[static_cast<std::size_t>(column + 1)]), b_norm);
    *stop = *relative_residual < options.tolerance;
    if (*stop) {
        update_solution(x, workspace->v, workspace->h, workspace->g,
                        column + 1, restart, blas, &result->profiler);
        result->converged = true;
    }
    return true;
}

CycleStatus run_cycle(ISpmvBackend& spmv, IBlasBackend& blas,
                      const std::vector<float>& b, std::vector<float>* x,
                      float b_norm, const GmresOptions& options,
                      std::int32_t restart, SolverWorkspace* workspace,
                      GmresResult* result, std::int32_t* total_iterations,
                      float* relative_residual, std::string* error) {
    const CycleStatus status = prepare_cycle(
        spmv, blas, b, *x, b_norm, options, workspace, result,
        relative_residual, error);
    if (status != CycleStatus::ready) return status;
    std::int32_t inner_dim = 0;
    for (std::int32_t column = 0;
         column < restart && *total_iterations < options.max_iterations; ++column) {
        bool stop = false;
        if (!arnoldi_step(column, restart, b_norm, spmv, blas, options,
                          workspace, result, x, total_iterations,
                          relative_residual, &stop, error)) return CycleStatus::failed;
        inner_dim = column + 1;
        if (stop) break;
    }
    if (!result->converged && inner_dim > 0) {
        update_solution(x, workspace->v, workspace->h, workspace->g,
                        inner_dim, restart, blas, &result->profiler);
    }
    *relative_residual = compute_relative_residual_profiled(
        spmv, blas, b, *x, &result->profiler);
    if (*relative_residual < options.tolerance) {
        result->converged = true;
        return CycleStatus::finished;
    }
    return CycleStatus::ready;
}

void finalize_result(const Clock::time_point total_start,
                     const Clock::time_point warm_start,
                     std::int32_t iterations, float relative_residual,
                     ISpmvBackend& spmv, GmresResult* result) {
    const auto end = Clock::now();
    result->iterations = iterations;
    result->final_relative_residual = relative_residual;
    result->total_ms = elapsed_ms(total_start, end);
    result->cold_start_ms = spmv.stats().cold_start_ms;
    result->warm_ms = elapsed_ms(warm_start, end);
    result->spmv_stats = spmv.stats();
    result->profiler.other_ms =
        std::max(0.0, result->total_ms - result->profiler.accounted_ms());
}

}  // namespace

GmresSolver::GmresSolver(std::string solver_name, std::unique_ptr<ISpmvBackend> spmv, std::unique_ptr<IBlasBackend> blas)
    : solver_name_(std::move(solver_name)), spmv_(std::move(spmv)), blas_(std::move(blas)) {}

bool GmresSolver::prepare(const CSRMatrix& matrix, std::string* error) {
    return spmv_->prepare(matrix, error);
}

GmresResult GmresSolver::solve(const std::vector<float>& b, std::vector<float>* x,
                               const GmresOptions& options, std::string* error) {
    GmresResult result;
    result.solver_name = solver_name_;
    if (x == nullptr) {
        if (error) *error = "solution vector is null";
        return result;
    }
    if (x->size() != b.size()) x->assign(b.size(), 0.0f);
    const auto total_start = Clock::now();
    const auto norm_begin = Clock::now();
    const float b_norm = blas_->norm2(b);
    result.profiler.norm_ms += elapsed_ms(norm_begin, Clock::now());
    float relative_residual =
        compute_relative_residual_profiled(*spmv_, *blas_, b, *x, &result.profiler);
    if (relative_residual < options.tolerance) {
        result.converged = true;
        result.final_relative_residual = relative_residual;
        result.total_ms = elapsed_ms(total_start, Clock::now());
        result.cold_start_ms = spmv_->stats().cold_start_ms;
        result.spmv_stats = spmv_->stats();
        result.profiler.other_ms =
            std::max(0.0, result.total_ms - result.profiler.accounted_ms());
        return result;
    }
    const std::int32_t restart = std::max<std::int32_t>(1, options.restart);
    SolverWorkspace workspace(restart, b.size());
    std::int32_t total_iterations = 0;
    const auto warm_start = Clock::now();
    while (total_iterations < options.max_iterations) {
        const CycleStatus status = run_cycle(
            *spmv_, *blas_, b, x, b_norm, options, restart, &workspace,
            &result, &total_iterations, &relative_residual, error);
        if (status == CycleStatus::failed) return result;
        if (status == CycleStatus::finished) break;
    }
    finalize_result(total_start, warm_start, total_iterations, relative_residual,
                    *spmv_, &result);
    return result;
}

float compute_relative_residual(ISpmvBackend& spmv, IBlasBackend& blas, const std::vector<float>& b, const std::vector<float>& x) {
    std::vector<float> ax;
    std::vector<float> r = b;
    std::string error;
    if (!spmv.multiply(x, &ax, &error)) {
        return std::numeric_limits<float>::infinity();
    }
    blas.axpy(-1.0f, ax, &r);
    return normalized_residual(blas.norm2(r), blas.norm2(b));
}

void print_profiler_breakdown(std::ostream& out, const std::string& solver_name, const GMRESProfiler& profiler, double total_ms, int iterations) {
    const auto print_stage = [&](const char* label, double value) {
        const double percent = total_ms > 0.0 ? (100.0 * value / total_ms) : 0.0;
        out << "    " << std::left << std::setw(12) << label << ": "
            << std::right << std::fixed << std::setprecision(6) << value
            << " ms (" << std::setprecision(2) << percent << "%)\n";
    };
    out << "  " << solver_name << "\n";
    out << "  --------------------------------\n";
    print_stage("Total", total_ms);
    print_stage("SpMV", profiler.spmv_ms);
    print_stage("Dot", profiler.dot_ms);
    print_stage("AXPY", profiler.axpy_ms);
    print_stage("Norm", profiler.norm_ms);
    print_stage("Givens", profiler.givens_ms);
    print_stage("Residual", profiler.residual_ms);
    print_stage("Other", profiler.other_ms);
    print_stage("H2D/D2H", profiler.device_transfer_ms);
    print_stage("HCCL", profiler.communication_ms);
    print_stage("Launch", profiler.kernel_launch_ms);
    print_stage("Sync", profiler.synchronization_ms);
    if (iterations > 0) {
        out << "    avg_spmv_per_iteration: " << std::setprecision(6) << profiler.spmv_ms / static_cast<double>(iterations) << " ms\n";
        out << "    avg_dot_per_iteration : " << std::setprecision(6) << profiler.dot_ms / static_cast<double>(iterations) << " ms\n";
        out << "    avg_axpy_per_iteration: " << std::setprecision(6) << profiler.axpy_ms / static_cast<double>(iterations) << " ms\n";
    }
}

}  // namespace ascend_gmres

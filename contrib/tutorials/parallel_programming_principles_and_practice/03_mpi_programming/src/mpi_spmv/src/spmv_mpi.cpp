#include "benchmark.hpp"
#include "spmv.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct LocalCSR {
    std::vector<std::int32_t> row_ptr;
    std::vector<std::int32_t> col_idx;
    std::vector<float> values;
};

struct Distribution {
    LocalCSR local;
    std::vector<int> row_counts;
    std::vector<int> row_displacements;
    std::vector<int> nnz_counts;
    std::vector<int> nnz_displacements;
    double setup_ms = 0.0;
    double nnz_balance_ratio = 0.0;
};

std::vector<std::int32_t> build_nnz_partitions(const mpi_spmv::CSRMatrix& matrix,
                                                int process_count) {
    if (process_count == 0) {
        throw std::runtime_error("MPI process count must not be zero");
    }
    if (process_count < 0 || process_count > matrix.rows) {
        throw std::runtime_error("MPI process count must be between 1 and matrix row count");
    }
    std::vector<std::int32_t> boundaries(static_cast<std::size_t>(process_count + 1));
    boundaries.front() = 0;
    boundaries.back() = matrix.rows;
    for (int rank = 1; rank < process_count; ++rank) {
        const std::int64_t target = matrix.nnz() * rank / process_count;
        const std::int32_t minimum_row = boundaries[static_cast<std::size_t>(rank - 1)] + 1;
        const std::int32_t maximum_row = matrix.rows - (process_count - rank);
        const auto begin = matrix.row_ptr.begin() + minimum_row;
        const auto end = matrix.row_ptr.begin() + maximum_row + 1;
        auto position = std::lower_bound(begin, end, target);
        std::int32_t boundary = position == end
                                    ? maximum_row
                                    : static_cast<std::int32_t>(position - matrix.row_ptr.begin());
        if (boundary > minimum_row) {
            const std::int64_t upper_distance =
                std::abs(static_cast<std::int64_t>(matrix.row_ptr[boundary]) - target);
            const std::int64_t lower_distance =
                std::abs(static_cast<std::int64_t>(matrix.row_ptr[boundary - 1]) - target);
            if (lower_distance <= upper_distance) --boundary;
        }
        boundaries[static_cast<std::size_t>(rank)] =
            std::clamp(boundary, minimum_row, maximum_row);
    }
    return boundaries;
}

std::vector<std::int32_t> broadcast_boundaries(
        const mpi_spmv::CSRMatrix& matrix, int rank, int process_count) {
    std::vector<std::int32_t> boundaries(static_cast<std::size_t>(process_count + 1));
    if (rank == 0) boundaries = build_nnz_partitions(matrix, process_count);
    MPI_Bcast(boundaries.data(), process_count + 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    return boundaries;
}

void populate_distribution(const mpi_spmv::CSRMatrix& matrix,
                           const std::vector<std::int32_t>& boundaries,
                           Distribution* distribution,
                           std::vector<int>* row_ptr_counts) {
    for (std::size_t process = 0; process < distribution->row_counts.size(); ++process) {
        const auto begin_row = boundaries[process];
        const auto end_row = boundaries[process + 1];
        distribution->row_counts[process] = end_row - begin_row;
        distribution->row_displacements[process] = begin_row;
        (*row_ptr_counts)[process] = end_row - begin_row + 1;
        distribution->nnz_displacements[process] = matrix.row_ptr[begin_row];
        distribution->nnz_counts[process] = matrix.row_ptr[end_row] - matrix.row_ptr[begin_row];
    }
}

void broadcast_distribution(Distribution* distribution,
                            std::vector<int>* row_ptr_counts,
                            int process_count) {
    MPI_Bcast(distribution->row_counts.data(), process_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(distribution->row_displacements.data(), process_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(distribution->nnz_counts.data(), process_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(distribution->nnz_displacements.data(), process_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr_counts->data(), process_count, MPI_INT, 0, MPI_COMM_WORLD);
}

void scatter_local_matrix(const mpi_spmv::CSRMatrix& matrix,
                          const std::vector<int>& row_ptr_counts,
                          int rank, Distribution* distribution) {
    const std::size_t index = static_cast<std::size_t>(rank);
    const int local_rows = distribution->row_counts[index];
    const int local_nnz = distribution->nnz_counts[index];
    distribution->local.row_ptr.resize(static_cast<std::size_t>(local_rows + 1));
    distribution->local.col_idx.resize(static_cast<std::size_t>(local_nnz));
    distribution->local.values.resize(static_cast<std::size_t>(local_nnz));
    MPI_Scatterv(rank == 0 ? matrix.row_ptr.data() : nullptr, row_ptr_counts.data(),
                 distribution->row_displacements.data(), MPI_INT32_T,
                 distribution->local.row_ptr.data(), local_rows + 1, MPI_INT32_T,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? matrix.col_idx.data() : nullptr,
                 distribution->nnz_counts.data(), distribution->nnz_displacements.data(),
                 MPI_INT32_T, distribution->local.col_idx.data(), local_nnz, MPI_INT32_T,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? matrix.values.data() : nullptr,
                 distribution->nnz_counts.data(), distribution->nnz_displacements.data(),
                 MPI_FLOAT, distribution->local.values.data(), local_nnz, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    const std::int32_t offset = distribution->local.row_ptr.front();
    for (auto& pointer : distribution->local.row_ptr) pointer -= offset;
}

void finish_distribution(double setup_start, std::int64_t global_nnz,
                         int rank, int process_count, Distribution* distribution) {
    const double local_setup_ms = (MPI_Wtime() - setup_start) * 1000.0;
    MPI_Reduce(&local_setup_ms, &distribution->setup_ms, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return;
    if (process_count == 0) {
        distribution->nnz_balance_ratio = 0.0;
        return;
    }
    if (process_count < 0) {
        distribution->nnz_balance_ratio = 0.0;
        return;
    }
    const int maximum_nnz =
        *std::max_element(distribution->nnz_counts.begin(), distribution->nnz_counts.end());
    const double average_nnz = static_cast<double>(global_nnz) / process_count;
    distribution->nnz_balance_ratio = average_nnz > 0.0
                                          ? maximum_nnz / average_nnz : 0.0;
}

Distribution distribute_matrix(const mpi_spmv::CSRMatrix& global_matrix,
                               std::int64_t global_nnz,
                               int rank,
                               int process_count) {
    if (process_count == 0) {
        throw std::runtime_error("MPI process count must not be zero");
    }
    if (process_count < 0) {
        throw std::runtime_error("MPI process count must be positive");
    }
    Distribution result;
    result.row_counts.resize(static_cast<std::size_t>(process_count));
    result.row_displacements.resize(static_cast<std::size_t>(process_count));
    result.nnz_counts.resize(static_cast<std::size_t>(process_count));
    result.nnz_displacements.resize(static_cast<std::size_t>(process_count));
    std::vector<int> row_ptr_counts(static_cast<std::size_t>(process_count));
    MPI_Barrier(MPI_COMM_WORLD);
    const double setup_start = MPI_Wtime();
    const auto boundaries = broadcast_boundaries(global_matrix, rank, process_count);
    if (rank == 0) {
        populate_distribution(global_matrix, boundaries, &result, &row_ptr_counts);
    }
    broadcast_distribution(&result, &row_ptr_counts, process_count);
    scatter_local_matrix(global_matrix, row_ptr_counts, rank, &result);
    finish_distribution(setup_start, global_nnz, rank, process_count, &result);
    return result;
}

double benchmark_cpu(const mpi_spmv::CSRMatrix& matrix,
                     const std::vector<float>& x,
                     const mpi_spmv::BenchmarkConfig& config,
                     std::vector<float>* y) {
    for (std::int32_t i = 0; i < config.warmup_iterations; ++i) {
        mpi_spmv::spmv_cpu(matrix, x, y);
    }
    double elapsed = 0.0;
    for (std::int32_t i = 0; i < config.repeat_iterations; ++i) {
        const double start = MPI_Wtime();
        mpi_spmv::spmv_cpu(matrix, x, y);
        elapsed += (MPI_Wtime() - start) * 1000.0;
    }
    return elapsed / config.repeat_iterations;
}

struct MpiTimings {
    double total_ms = 0.0;
    double broadcast_ms = 0.0;
    double compute_ms = 0.0;
    double gather_ms = 0.0;
    double average_compute_ms = 0.0;
    double load_imbalance = 0.0;
};

struct LocalTimings {
    double total = 0.0;
    double broadcast = 0.0;
    double compute = 0.0;
    double gather = 0.0;
};

void run_iteration(const Distribution& distribution, int rank,
                   std::vector<float>* x, std::vector<float>* local_y,
                   std::vector<float>* global_y) {
    const int local_rows = distribution.row_counts[static_cast<std::size_t>(rank)];
    MPI_Bcast(x->data(), static_cast<int>(x->size()), MPI_FLOAT, 0, MPI_COMM_WORLD);
    mpi_spmv::spmv_local(distribution.local.row_ptr, distribution.local.col_idx,
                        distribution.local.values, *x, local_y);
    MPI_Gatherv(local_y->data(), local_rows, MPI_FLOAT,
                rank == 0 ? global_y->data() : nullptr,
                distribution.row_counts.data(), distribution.row_displacements.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);
}

LocalTimings measure_iterations(const Distribution& distribution, int rank,
                                std::vector<float>* x, std::vector<float>* local_y,
                                std::vector<float>* global_y, std::int32_t repeats) {
    LocalTimings timing;
    for (std::int32_t i = 0; i < repeats; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double total_start = MPI_Wtime();
        MPI_Bcast(x->data(), static_cast<int>(x->size()), MPI_FLOAT, 0, MPI_COMM_WORLD);
        const double broadcast_end = MPI_Wtime();
        mpi_spmv::spmv_local(distribution.local.row_ptr, distribution.local.col_idx,
                            distribution.local.values, *x, local_y);
        const double compute_end = MPI_Wtime();
        const int local_rows = distribution.row_counts[static_cast<std::size_t>(rank)];
        MPI_Gatherv(local_y->data(), local_rows, MPI_FLOAT,
                    rank == 0 ? global_y->data() : nullptr,
                    distribution.row_counts.data(), distribution.row_displacements.data(),
                    MPI_FLOAT, 0, MPI_COMM_WORLD);
        const double gather_end = MPI_Wtime();
        timing.total += gather_end - total_start;
        timing.broadcast += broadcast_end - total_start;
        timing.compute += compute_end - broadcast_end;
        timing.gather += gather_end - compute_end;
    }
    return timing;
}

std::vector<double> gather_rank_records(const Distribution& distribution,
                                        const LocalTimings& timing,
                                        double scale, int rank) {
    const std::size_t index = static_cast<std::size_t>(rank);
    const int local_rows = distribution.row_counts[index];
    double record[9] = {
        static_cast<double>(rank),
        static_cast<double>(distribution.row_displacements[index]),
        static_cast<double>(distribution.row_displacements[index] + local_rows),
        static_cast<double>(local_rows),
        static_cast<double>(distribution.local.values.size()),
        timing.broadcast * scale, timing.compute * scale,
        timing.gather * scale, timing.total * scale};
    std::vector<double> records;
    if (rank == 0) records.resize(9 * distribution.row_counts.size());
    MPI_Gather(record, 9, MPI_DOUBLE, rank == 0 ? records.data() : nullptr,
               9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return records;
}

void emit_rank_records(const Distribution& distribution,
                       const std::vector<double>& records,
                       std::ostream* per_rank_csv,
                       const std::string& matrix_name) {
    std::cout << "Per-rank records (rank,row_begin,row_end,local_rows,local_nnz,"
                 "broadcast_ms,compute_ms,gather_ms,total_ms):\n";
    for (std::size_t rank = 0; rank < distribution.row_counts.size(); ++rank) {
        const double* value = records.data() + 9 * rank;
        std::cout << "  rank=" << static_cast<int>(value[0]) << " rows=["
                  << static_cast<int>(value[1]) << ',' << static_cast<int>(value[2])
                  << ") local_rows=" << static_cast<int>(value[3])
                  << " local_nnz=" << static_cast<std::int64_t>(value[4])
                  << " broadcast_ms=" << value[5] << " compute_ms=" << value[6]
                  << " gather_ms=" << value[7] << " total_ms=" << value[8] << '\n';
        if (!per_rank_csv) continue;
        *per_rank_csv << matrix_name << ',' << static_cast<int>(value[0]) << ','
                      << static_cast<int>(value[1]) << ',' << static_cast<int>(value[2]) << ','
                      << static_cast<int>(value[3]) << ',' << static_cast<std::int64_t>(value[4])
                      << ',' << value[5] << ',' << value[6] << ',' << value[7]
                      << ',' << value[8] << '\n';
    }
}

MpiTimings reduce_timings(const Distribution& distribution,
                          const LocalTimings& local, double scale, int rank) {
    double local_values[4] = {local.total, local.broadcast, local.compute, local.gather};
    double maximum_values[4] = {};
    MPI_Reduce(local_values, maximum_values, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double compute_sum = 0.0;
    MPI_Reduce(&local.compute, &compute_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    const double average = rank == 0
                                ? compute_sum * scale / distribution.row_counts.size() : 0.0;
    const double maximum = maximum_values[2] * scale;
    const double imbalance = average > 0.0 ? (maximum - average) / average : 0.0;
    return {maximum_values[0] * scale, maximum_values[1] * scale, maximum,
            maximum_values[3] * scale, average, imbalance};
}

MpiTimings benchmark_mpi(const Distribution& distribution,
                         std::vector<float>* x,
                         const mpi_spmv::BenchmarkConfig& config,
                         int rank,
                         std::vector<float>* global_y,
                         std::ostream* per_rank_csv,
                         const std::string& matrix_name) {
    const int local_rows = distribution.row_counts[static_cast<std::size_t>(rank)];
    std::vector<float> local_y(static_cast<std::size_t>(local_rows));
    for (std::int32_t i = 0; i < config.warmup_iterations; ++i) {
        run_iteration(distribution, rank, x, &local_y, global_y);
    }
    const LocalTimings local = measure_iterations(
        distribution, rank, x, &local_y, global_y, config.repeat_iterations);
    const double scale = 1000.0 / config.repeat_iterations;
    const auto records = gather_rank_records(distribution, local, scale, rank);
    if (rank == 0) emit_rank_records(distribution, records, per_rank_csv, matrix_name);
    return reduce_timings(distribution, local, scale, rank);
}

struct OutputFiles {
    std::ofstream csv;
    std::ofstream per_rank_csv;
};

void open_output_files(const mpi_spmv::BenchmarkConfig& config, OutputFiles* output) {
    mpi_spmv::ensure_parent_directory(config.csv_path);
    output->csv.open(config.csv_path);
    if (!output->csv) throw std::runtime_error("failed to open CSV: " + config.csv_path);
    output->csv << "matrix,rows,cols,nnz,processes,warmup,repeat,cpu_time_ms,mpi_time_ms,"
                   "speedup,broadcast_ms,compute_ms,gather_ms,average_compute_ms,load_imbalance,"
                   "distribution_ms,nnz_balance_ratio,l2_error,max_abs_error,correct\n";
    const std::string path = config.csv_path + ".per_rank.csv";
    mpi_spmv::ensure_parent_directory(path);
    output->per_rank_csv.open(path);
    if (!output->per_rank_csv) {
        throw std::runtime_error("failed to open per-rank CSV: " + path);
    }
    output->per_rank_csv << "matrix,rank,row_begin,row_end,local_rows,local_nnz,broadcast_ms,"
                            "compute_ms,gather_ms,total_ms\n";
    std::cout << std::fixed << std::setprecision(6);
}

struct SpecRun {
    mpi_spmv::CSRMatrix matrix;
    std::vector<float> x;
    std::vector<float> cpu_y;
    std::vector<float> mpi_y;
    std::int32_t metadata[2] = {};
    std::int64_t global_nnz = 0;
    double cpu_ms = 0.0;
    Distribution distribution;
    MpiTimings mpi_timings;
};

void prepare_root_problem(const mpi_spmv::BenchmarkConfig& config,
                          const mpi_spmv::MatrixSpec& spec, SpecRun* run) {
    run->matrix = mpi_spmv::load_or_generate_matrix(config, spec);
    run->x = mpi_spmv::generate_rhs_vector(run->matrix.cols, 42);
    run->cpu_ms = benchmark_cpu(run->matrix, run->x, config, &run->cpu_y);
    run->metadata[0] = run->matrix.rows;
    run->metadata[1] = run->matrix.cols;
    run->global_nnz = run->matrix.nnz();
}

void broadcast_problem(SpecRun* run, int rank) {
    MPI_Bcast(run->metadata, 2, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&run->global_nnz, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    if (rank != 0) run->x.resize(static_cast<std::size_t>(run->metadata[1]));
}

void print_summary(const mpi_spmv::MatrixSpec& spec, const SpecRun& run,
                   int process_count, double speedup, double l2,
                   double max_abs, bool correct) {
    const auto& timing = run.mpi_timings;
    std::cout << "Matrix: " << spec.name << '\n'
              << "  matrix_size = " << run.metadata[0] << " x " << run.metadata[1] << '\n'
              << "  nnz = " << run.global_nnz << "\nCPU SpMV:\n"
              << "  time = " << run.cpu_ms << " ms\nMPI SpMV:\n"
              << "  processes = " << process_count << '\n'
              << "  time = " << timing.total_ms << " ms\n  speedup = " << speedup << "x\n"
              << "  broadcast = " << timing.broadcast_ms << " ms\n"
              << "  compute = " << timing.compute_ms << " ms\n"
              << "  average_compute = " << timing.average_compute_ms << " ms\n"
              << "  load_imbalance = " << timing.load_imbalance << '\n'
              << "  gather = " << timing.gather_ms << " ms\n"
              << "  distribution_once = " << run.distribution.setup_ms << " ms\n"
              << "  nnz_balance_ratio = " << run.distribution.nnz_balance_ratio << '\n'
              << "  correctness_error_l2 = " << l2 << '\n'
              << "  correctness_error_max_abs = " << max_abs << '\n'
              << "  correctness = " << (correct ? "PASS" : "FAIL") << '\n';
}

void write_summary(std::ofstream* csv, const mpi_spmv::BenchmarkConfig& config,
                   const mpi_spmv::MatrixSpec& spec, const SpecRun& run,
                   int process_count, double speedup, double l2,
                   double max_abs, bool correct) {
    const auto& timing = run.mpi_timings;
    *csv << spec.name << ',' << run.metadata[0] << ',' << run.metadata[1] << ','
         << run.global_nnz << ',' << process_count << ',' << config.warmup_iterations << ','
         << config.repeat_iterations << ',' << std::setprecision(9) << run.cpu_ms << ','
         << timing.total_ms << ',' << speedup << ',' << timing.broadcast_ms << ','
         << timing.compute_ms << ',' << timing.gather_ms << ','
         << timing.average_compute_ms << ',' << timing.load_imbalance << ','
         << run.distribution.setup_ms << ',' << run.distribution.nnz_balance_ratio << ','
         << l2 << ',' << max_abs << ',' << (correct ? 1 : 0) << '\n';
    std::cout << std::setprecision(6);
}

void report_root_result(const mpi_spmv::BenchmarkConfig& config,
                        const mpi_spmv::MatrixSpec& spec, const SpecRun& run,
                        int process_count, std::ofstream* csv) {
    const double l2 = mpi_spmv::l2_error(run.cpu_y, run.mpi_y);
    const double max_abs = mpi_spmv::max_absolute_error(run.cpu_y, run.mpi_y);
    const bool correct = l2 < 1e-6;
    const double speedup = run.mpi_timings.total_ms > 0.0
                               ? run.cpu_ms / run.mpi_timings.total_ms : 0.0;
    print_summary(spec, run, process_count, speedup, l2, max_abs, correct);
    write_summary(csv, config, spec, run, process_count, speedup, l2, max_abs, correct);
    if (!correct) throw std::runtime_error("correctness check failed for " + spec.name);
}

void run_spec(const mpi_spmv::BenchmarkConfig& config,
              const mpi_spmv::MatrixSpec& spec, int rank, int process_count,
              OutputFiles* output) {
    SpecRun run;
    if (rank == 0) prepare_root_problem(config, spec, &run);
    broadcast_problem(&run, rank);
    run.distribution = distribute_matrix(run.matrix, run.global_nnz, rank, process_count);
    if (rank == 0) run.mpi_y.resize(static_cast<std::size_t>(run.metadata[0]));
    run.mpi_timings = benchmark_mpi(
        run.distribution, &run.x, config, rank, &run.mpi_y,
        rank == 0 ? &output->per_rank_csv : nullptr, spec.name);
    if (rank == 0) report_root_result(config, spec, run, process_count, &output->csv);
}

int run_course(int argc, char** argv, int rank, int process_count) {
    const auto config = mpi_spmv::parse_arguments(argc, argv, "results/mpi_spmv.csv");
    if (config.matrix_name == "__help__") {
        if (rank == 0) mpi_spmv::print_usage(argv[0], true);
        return 0;
    }
    OutputFiles output;
    if (rank == 0) open_output_files(config, &output);
    for (const auto& spec : mpi_spmv::selected_specs(config.matrix_name)) {
        run_spec(config, spec, rank, process_count, &output);
    }
    if (rank == 0) std::cout << "CSV saved to " << config.csv_path << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank = 0;
    int process_count = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    try {
        const int result = run_course(argc, argv, rank, process_count);
        MPI_Finalize();
        return result;
    } catch (const std::exception& exception) {
        std::cerr << "Rank " << rank << " failed: " << exception.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

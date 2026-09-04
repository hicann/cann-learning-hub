#include "csr_matrix.hpp"
#include "gmres.hpp"
#include "hccl_comm.hpp"
#include "spmv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {

struct Options {
  std::string matrix = "U1";
  std::string matrix_dir = "matrices";
  std::string result_csv = "results/dis_gmres.csv";
  std::string rank_table;
  int device = 0;
  int rank = 0;
  int world = 1;
  int warmup = 0;
  int repeat = 3;
  int restart = 30;
  int max_iterations = 10000;
  float tolerance = 1.0e-6f;
  bool nnz_partition = true;
  bool communication_avoiding = false;
  bool parallel_compute = true;
  bool fused_vector_ops = true;
  double single_baseline_ms = -1.0;
};

int environment_integer(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value ? std::stoi(value) : fallback;
}

void print_usage() {
  std::cout
      << "Usage: dis_gmres [options]\n"
      << "  --matrix U1|U2|L1|L2|B1|B2  --matrix-dir DIR\n"
      << "  --rank R --world-size N --device D --rank-table FILE\n"
      << "  --warmup N --repeat N --restart N --max-iterations N\n"
      << "  --warmup N  accepted for compatibility; N extra solver invocations are executed and discarded\n"
      << "  --tolerance X --partition nnz|rows\n"
      << "  --orthogonalization cgs|mgs --single-baseline-ms MS --no-openmp\n"
      << "  --unfused-vector-ops  use the original per-vector OpenMP loops\n";
}

std::string next_value(int argc, char** argv, int* index) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string(argv[*index]) + " requires a value");
  }
  return argv[++*index];
}

bool apply_basic_option(const std::string& argument, const std::string& value,
                        Options* options) {
  if (argument == "--matrix") { options->matrix = value; return true; }
  if (argument == "--matrix-dir") { options->matrix_dir = value; return true; }
  if (argument == "--result-csv") { options->result_csv = value; return true; }
  if (argument == "--rank-table") { options->rank_table = value; return true; }
  if (argument == "--device") { options->device = std::stoi(value); return true; }
  if (argument == "--rank") { options->rank = std::stoi(value); return true; }
  if (argument == "--world-size") { options->world = std::stoi(value); return true; }
  if (argument == "--warmup") { options->warmup = std::stoi(value); return true; }
  if (argument == "--repeat") { options->repeat = std::stoi(value); return true; }
  if (argument == "--restart") { options->restart = std::stoi(value); return true; }
  if (argument == "--max-iterations") { options->max_iterations = std::stoi(value); return true; }
  if (argument == "--tolerance") { options->tolerance = std::stof(value); return true; }
  if (argument == "--single-baseline-ms") {
    options->single_baseline_ms = std::stod(value);
    return true;
  }
  return false;
}

bool apply_mode_option(const std::string& argument, const std::string& value,
                       Options* options) {
  if (argument == "--partition") {
    if (value != "nnz" && value != "rows") {
      throw std::runtime_error("partition must be nnz or rows");
    }
    options->nnz_partition = value == "nnz";
    return true;
  }
  if (argument == "--orthogonalization") {
    if (value != "cgs" && value != "mgs") {
      throw std::runtime_error("orthogonalization must be cgs or mgs");
    }
    options->communication_avoiding = value == "cgs";
    return true;
  }
  return false;
}

bool is_value_option(const std::string& argument) {
  return argument == "--matrix" || argument == "--matrix-dir" ||
         argument == "--result-csv" || argument == "--rank-table" ||
         argument == "--device" || argument == "--rank" ||
         argument == "--world-size" || argument == "--warmup" ||
         argument == "--repeat" || argument == "--restart" ||
         argument == "--max-iterations" || argument == "--tolerance" ||
         argument == "--single-baseline-ms" || argument == "--partition" ||
         argument == "--orthogonalization";
}

void validate_options(const Options& options) {
  if (options.world < 1 || options.rank < 0 || options.rank >= options.world ||
      options.warmup < 0 || options.repeat < 1 || options.restart < 1 ||
      options.max_iterations < 1 || options.tolerance <= 0.0f) {
    throw std::runtime_error("invalid rank or benchmark/GMRES parameter");
  }
}

Options parse_options(int argc, char** argv) {
  Options options;
  options.device = environment_integer("DEVICE_ID", 0);
  options.rank = environment_integer("RANK_ID", 0);
  options.world = environment_integer("RANK_SIZE", 1);
  const char* rank_table = std::getenv("RANK_TABLE_FILE");
  if (rank_table) options.rank_table = rank_table;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help") {
      print_usage();
      std::exit(0);
    }
    if (argument == "--no-openmp") {
      options.parallel_compute = false;
      continue;
    }
    if (argument == "--unfused-vector-ops") {
      options.fused_vector_ops = false;
      continue;
    }
    if (!is_value_option(argument)) {
      throw std::runtime_error("unknown argument: " + argument);
    }
    const std::string value = next_value(argc, argv, &i);
    if (!apply_basic_option(argument, value, &options) &&
        !apply_mode_option(argument, value, &options)) {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  validate_options(options);
  return options;
}

dis_gmres::CSRMatrix load_or_generate(const Options& options) {
  const auto* spec = dis_gmres::find_matrix_spec(options.matrix);
  if (!spec) throw std::runtime_error("unknown matrix: " + options.matrix);
  fs::create_directories(options.matrix_dir);
  const auto path = (fs::path(options.matrix_dir) / (options.matrix + ".csrbin")).string();
  dis_gmres::CSRMatrix matrix;
  std::string error;
  if (fs::exists(path)) {
    if (!dis_gmres::CSRMatrix::load_binary(path, &matrix, &error))
      throw std::runtime_error(error);
    return matrix;
  }
  matrix = dis_gmres::generate_matrix(*spec, 42);
  if (!matrix.save_binary(path, &error)) throw std::runtime_error(error);
  return matrix;
}

// Converts one solver Profile into the 12 benchmark fields. The first 10
// entries are wall-time phases; the last 2 entries are per-solve collective
// call counts. Reduction order is mean_repeat(max_rank): each repeat's 12
// fields are first reduced elementwise MAX across ranks (per-repeat
// distributed critical-path wall time / largest single-rank call count), then
// the per-repeat MAX vectors are averaged over repeats. A cross-rank SUM
// would artificially grow with world size and max_rank(mean_repeat) would
// hide the slowest rank inside each repeat; neither must be used for the
// critical path.
std::vector<float> profile_to_vector(const dis_gmres::Profile& profile) {
  return {static_cast<float>(profile.total_ms),
          static_cast<float>(profile.spmv_ms),
          static_cast<float>(profile.dot_ms),
          static_cast<float>(profile.axpy_ms),
          static_cast<float>(profile.norm_ms),
          static_cast<float>(profile.givens_ms),
          static_cast<float>(profile.communication_ms),
          static_cast<float>(profile.transfer_ms),
          static_cast<float>(profile.kernel_launch_ms),
          static_cast<float>(profile.synchronization_ms),
          static_cast<float>(profile.allreduce_calls),
          static_cast<float>(profile.allgather_calls)};
}

void write_csv(const Options& options, const dis_gmres::CSRMatrix& local_matrix,
               std::int32_t global_rows, std::int64_t global_nnz,
               const dis_gmres::GmresResult& result, const std::vector<float>& p,
               double solution_error, double speedup,
               const dis_gmres::HcclCommunicator& communicator) {
  fs::create_directories(fs::path(options.result_csv).parent_path());
  const bool header = !fs::exists(options.result_csv) || fs::file_size(options.result_csv) == 0;
  std::ofstream csv(options.result_csv, std::ios::app);
  if (!csv) throw std::runtime_error("cannot open result CSV: " + options.result_csv);
  if (header)
    csv << "matrix,rows,cols,nnz,world,rank_rows,rank_nnz,partition,orthogonalization,backend,"
           "iterations,residual,converged,total_ms,speedup,spmv_ms,dot_ms,axpy_ms,norm_ms,"
           "givens_ms,hccl_ms,transfer_ms,kernel_launch_ms,synchronization_ms,allreduce_calls,"
           "allgather_calls,solution_error\n";
  csv << options.matrix << ',' << global_rows << ',' << local_matrix.cols << ',' << global_nnz
      << ',' << options.world << ',' << local_matrix.rows << ',' << local_matrix.nnz() << ','
      << (options.nnz_partition ? "nnz" : "rows") << ','
      << (options.communication_avoiding ? "cgs" : "mgs") << ','
      << (communicator.real_hccl() ? "ACL+HCCL" : "host-stub") << ','
      << result.iterations << ',' << result.residual << ',' << result.converged << ','
      << p[0] << ',' << speedup << ',' << p[1] << ',' << p[2] << ',' << p[3] << ','
      << p[4] << ',' << p[5] << ',' << p[6] << ',' << p[7] << ',' << p[8] << ','
      << p[9] << ',' << p[10] << ',' << p[11] << ',' << solution_error << '\n';
}

struct RunState {
  Options options;
  dis_gmres::CSRMatrix local_matrix;
  std::vector<dis_gmres::RowPartition> partitions;
  dis_gmres::RowPartition own;
  dis_gmres::HcclCommunicator communicator;
  dis_gmres::GmresOptions gmres_options;
  std::vector<float> local_b;
  std::vector<float> local_true;
  std::int32_t global_rows = 0;
  std::int32_t global_cols = 0;
  std::int64_t global_nnz = 0;
  std::string error;
};

dis_gmres::GmresOptions make_gmres_options(const Options& options) {
  dis_gmres::GmresOptions result;
  result.restart = options.restart;
  result.max_iterations = options.max_iterations;
  result.tolerance = options.tolerance;
  result.communication_avoiding = options.communication_avoiding;
  result.parallel_compute = options.parallel_compute;
  result.fused_vector_ops = options.fused_vector_ops;
  result.zero_initial_guess = true;
  return result;
}

void prepare_run(const Options& options, RunState* state) {
  state->options = options;
  auto full_matrix = load_or_generate(options);
  state->global_rows = full_matrix.rows;
  state->global_cols = full_matrix.cols;
  state->global_nnz = full_matrix.nnz();
  state->partitions = dis_gmres::partition_rows(
      full_matrix, options.world, options.nnz_partition);
  state->own = state->partitions[static_cast<std::size_t>(options.rank)];
  state->local_matrix = dis_gmres::extract_rows(
      full_matrix, state->own.first, state->own.last);
  full_matrix = {};
  if (!state->communicator.initialize(options.device, options.rank, options.world,
                                      options.rank_table, &state->error)) {
    throw std::runtime_error(state->error);
  }
  const auto x_true = dis_gmres::generate_solution_vector(state->global_cols, 42);
  dis_gmres::csr_spmv(state->local_matrix, x_true, &state->local_b,
                      options.parallel_compute);
  state->local_true.assign(x_true.begin() + state->own.first,
                           x_true.begin() + state->own.last);
  state->gmres_options = make_gmres_options(options);
}

void run_warmups(RunState* state) {
  for (int warmup = 0; warmup < state->options.warmup; ++warmup) {
    std::vector<float> x(state->local_b.size(), 0.0f);
    const auto result = dis_gmres::distributed_gmres(
        state->local_matrix, state->partitions, state->local_b, &x,
        &state->communicator, state->gmres_options, &state->error);
    if (!result.converged) {
      throw std::runtime_error("discarded GMRES invocation did not converge: " +
                               state->error);
    }
  }
}

struct SolveSummary {
  dis_gmres::Profile profile_sum;
  std::vector<float> global_profile = std::vector<float>(12, 0.0f);
  dis_gmres::GmresResult last_result;
  std::vector<float> last_x;
};

void accumulate_profile(RunState* state, const dis_gmres::Profile& profile,
                        std::vector<float>* sum) {
  std::vector<float> maximum;
  if (!state->communicator.allreduce_max(profile_to_vector(profile),
                                         &maximum, &state->error)) {
    throw std::runtime_error(state->error);
  }
  for (std::size_t field = 0; field < sum->size(); ++field) {
    (*sum)[field] += maximum[field];
  }
}

SolveSummary run_repeats(RunState* state) {
  SolveSummary summary;
  for (int repeat = 0; repeat < state->options.repeat; ++repeat) {
    std::vector<float> x(state->local_b.size(), 0.0f);
    auto result = dis_gmres::distributed_gmres(
        state->local_matrix, state->partitions, state->local_b, &x,
        &state->communicator, state->gmres_options, &state->error);
    if (!result.converged) {
      throw std::runtime_error("GMRES did not converge: residual=" +
                               std::to_string(result.residual) + " " + state->error);
    }
    summary.profile_sum.accumulate(result.profile);
    accumulate_profile(state, result.profile, &summary.global_profile);
    summary.last_result = result;
    summary.last_x = std::move(x);
  }
  for (float& value : summary.global_profile) {
    value /= static_cast<float>(state->options.repeat);
  }
  return summary;
}

struct ErrorSummary {
  double local = 0.0;
  double global = 0.0;
};

ErrorSummary calculate_solution_error(RunState* state, const std::vector<float>& x) {
  double error_squared = 0.0;
  for (std::size_t index = 0; index < x.size(); ++index) {
    const double difference = static_cast<double>(x[index]) - state->local_true[index];
    error_squared += difference * difference;
  }
  const double reference_squared =
      dis_gmres::local_dot(state->local_true, state->local_true, false);
  ErrorSummary result;
  result.local = std::sqrt(error_squared / std::max(reference_squared, 1.0e-30));
  std::vector<float> reduced;
  if (!state->communicator.allreduce_sum(
          {static_cast<float>(error_squared), static_cast<float>(reference_squared)},
          &reduced, &state->error)) {
    throw std::runtime_error(state->error);
  }
  result.global = std::sqrt(reduced[0] / std::max(reduced[1], 1.0e-30f));
  return result;
}

void print_rank_summary(const RunState& state, const SolveSummary& summary,
                        const ErrorSummary& error) {
  const double repeats = static_cast<double>(state.options.repeat);
  std::cout << "[rank " << state.options.rank << "] observed global residual = "
            << std::scientific << summary.last_result.residual
            << ", local solution relative error = " << error.local << std::fixed
            << ", local total = " << summary.profile_sum.total_ms / repeats
            << " ms, local SpMV = " << summary.profile_sum.spmv_ms / repeats
            << " ms, local HCCL comm = "
            << summary.profile_sum.communication_ms / repeats
            << " ms, local AllReduce calls = "
            << summary.profile_sum.allreduce_calls /
                   static_cast<std::size_t>(state.options.repeat)
            << '\n';
}

void print_problem(const RunState& state, const SolveSummary& summary) {
  std::cout << std::fixed << std::setprecision(6)
            << "Matrix:\nrows = " << state.global_rows
            << "\ncols = " << state.global_cols << "\nnnz = " << state.global_nnz
            << "\n\nGMRES:\niteration = " << summary.last_result.iterations
            << "\nresidual = " << std::scientific << summary.last_result.residual
            << "\nconverged = " << (summary.last_result.converged ? "yes" : "no")
            << std::fixed;
}

void print_runtime(const RunState& state) {
  const bool real = state.communicator.real_hccl();
  std::cout << "\n\nRuntime:\nworld size = " << state.options.world
            << "\nbackend = " << (real ? "ACL + HCCL" : "host stub")
            << "\ncompute backend = "
            << (real ? "Ascend C RTC Device GMRES" : "C++ reference kernels (Host test only)")
            << "\nOpenMP threads per rank = " << dis_gmres::max_compute_threads()
            << "\nOpenMP minimum elements = " << dis_gmres::omp_min_elements()
            << "\nvector operations = "
            << (state.options.fused_vector_ops ? "fused" : "unfused baseline")
            << "\npartition = "
            << (state.options.nnz_partition ? "nnz-balanced" : "row-balanced")
            << "\northogonalization = "
            << (state.options.communication_avoiding ? "communication-avoiding CGS" : "MGS")
            << "\nDevice vectors = "
            << (real ? "persistent during solve" : "not available")
            << "\nbenchmark mode = per-solve RTC/Device state rebuild (matrix load/partition and "
               "communicator init outside the loop; --warmup runs are discarded extra solver invocations)";
}

void print_profile(const RunState& state, const SolveSummary& summary,
                   double baseline, double speedup) {
  const auto& p = summary.global_profile;
  std::cout << "\n\nSingle-rank baseline:\ntotal time = " << baseline
            << " ms (single-rank wall time)\n\n"
            << (state.communicator.real_hccl() ? "Distributed NPU:" : "Distributed Host Stub:")
            << "\ntotal time = " << p[0] << " ms (mean over " << state.options.repeat
            << " repeats of the per-repeat MAX across ranks = distributed critical-path wall time; "
               "max_rank(mean_repeat) and SUM/world_size rank means are not wall times and are never "
               "used for total_ms or speedup)\n\nSpeedup:\n" << speedup
            << "x\n\nPerformance breakdown (per-repeat MAX across ranks, then averaged over repeats):\n"
            << "SpMV = " << p[1] << " ms\nDot = " << p[2] << " ms\nAXPY = " << p[3]
            << " ms\nNorm = " << p[4] << " ms\nGivens = " << p[5]
            << " ms\nHCCL communication = " << p[6] << " ms\nACL transfer = " << p[7]
            << " ms\nkernel launch = " << p[8] << " ms\nsynchronization = " << p[9]
            << " ms\nAllReduce calls = " << p[10]
            << " (per-repeat max single-rank per solve, averaged over repeats)\nAllGather calls = "
            << p[11] << " (per-repeat max single-rank per solve, averaged over repeats)";
}

void report_root(const RunState& state, const SolveSummary& summary,
                 const ErrorSummary& error, double baseline, double speedup) {
  print_problem(state, summary);
  print_runtime(state);
  print_profile(state, summary, baseline, speedup);
  std::cout << "\n\nCorrectness:\nsolution relative error = " << std::scientific
            << error.global << "\nRESULT_TOTAL_MS=" << std::fixed
            << summary.global_profile[0] << "\nRESULT_RESIDUAL=" << std::scientific
            << summary.last_result.residual << '\n';
  write_csv(state.options, state.local_matrix, state.global_rows, state.global_nnz,
            summary.last_result, summary.global_profile, error.global, speedup,
            state.communicator);
}

int run_benchmark(const Options& options) {
  RunState state;
  prepare_run(options, &state);
  run_warmups(&state);
  const SolveSummary summary = run_repeats(&state);
  const ErrorSummary error = calculate_solution_error(&state, summary.last_x);
  const double baseline = options.single_baseline_ms > 0.0
                              ? options.single_baseline_ms
                              : (options.world == 1 ? summary.global_profile[0] : 0.0);
  const double speedup = baseline > 0.0 ? baseline / summary.global_profile[0] : 0.0;
  print_rank_summary(state, summary, error);
  if (options.rank == 0) report_root(state, summary, error, baseline, speedup);
  state.communicator.finalize();
  return error.global < 1.0e-3 && summary.last_result.residual <= options.tolerance ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    return run_benchmark(options);
  } catch (const std::exception& exception) {
    std::cerr << "dis_gmres: " << exception.what() << '\n';
    return 2;
  }
}

#include "csr_matrix.hpp"
#include "gmres.hpp"
#include "hccl_comm.hpp"
#include "spmv.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct SolverFixture {
  dis_gmres::CSRMatrix matrix;
  dis_gmres::CSRMatrix local;
  std::vector<dis_gmres::RowPartition> partitions;
  const std::vector<float> expected = {1, -2, 3, -4};
  std::vector<float> b;
  std::vector<float> x = std::vector<float>(4, 0.0f);
  dis_gmres::HcclCommunicator communicator;
  dis_gmres::GmresOptions options;
  dis_gmres::GmresResult result;
  std::string error;
};

bool initialize_fixture(SolverFixture* fixture) {
  fixture->matrix.rows = 4;
  fixture->matrix.cols = 4;
  fixture->matrix.row_ptr = {0, 2, 5, 8, 10};
  fixture->matrix.col_idx = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
  fixture->matrix.values = {4, -1, -1, 4, -1, -1, 4, -1, -1, 3};
  if (!fixture->matrix.validate(&fixture->error)) return false;
  fixture->partitions = dis_gmres::partition_rows(fixture->matrix, 1, true);
  fixture->local = dis_gmres::extract_rows(fixture->matrix, 0, 4);
  dis_gmres::csr_spmv(fixture->local, fixture->expected, &fixture->b, false);
  if (!fixture->communicator.initialize(0, 0, 1, "", &fixture->error)) return false;
  fixture->options.restart = 4;
  fixture->options.max_iterations = 20;
  fixture->options.tolerance = 1.0e-6f;
  fixture->options.parallel_compute = false;
  fixture->options.zero_initial_guess = true;
  fixture->options.communication_avoiding = false;
  fixture->result = dis_gmres::distributed_gmres(
      fixture->local, fixture->partitions, fixture->b, &fixture->x,
      &fixture->communicator, fixture->options, &fixture->error);
  return fixture->result.converged && fixture->result.residual <= 1.0e-6f;
}

bool test_device_regressions(SolverFixture* fixture) {
#if DIS_GMRES_HAS_CANN
  auto cgs_options = fixture->options;
  cgs_options.communication_avoiding = true;
  std::vector<float> cgs_x(4, 0.0f);
  std::string error;
  const auto rejected = dis_gmres::distributed_gmres(
      fixture->local, fixture->partitions, fixture->b, &cgs_x,
      &fixture->communicator, cgs_options, &error);
  if (rejected.converged ||
      error.find("Device CGS multi-dot kernel is not enabled") == std::string::npos) {
    std::cerr << "CGS rejection test failed: converged=" << rejected.converged
              << " error='" << error << "'\n";
    return false;
  }
  if (fixture->result.iterations < 1) return false;
  auto bounded_options = fixture->options;
  bounded_options.max_iterations = fixture->result.iterations;
  std::vector<float> bounded_x(4, 0.0f);
  const auto bounded = dis_gmres::distributed_gmres(
      fixture->local, fixture->partitions, fixture->b, &bounded_x,
      &fixture->communicator, bounded_options, &error);
  if (!bounded.converged || bounded.residual > 1.0e-6f) {
    std::cerr << "bounded max_iterations regression failed: residual="
              << bounded.residual << '\n';
    return false;
  }
#else
  (void)fixture;
#endif
  return true;
}

bool test_solution_variants(SolverFixture* fixture) {
  double squared_error = 0.0;
  for (std::size_t i = 0; i < fixture->x.size(); ++i) {
    const double delta = fixture->x[i] - fixture->expected[i];
    squared_error += delta * delta;
  }
  if (std::sqrt(squared_error) > 1.0e-4) return false;
  auto options = fixture->options;
  options.fused_vector_ops = false;
  std::vector<float> unfused_x(4, 0.0f);
  const auto unfused = dis_gmres::distributed_gmres(
      fixture->local, fixture->partitions, fixture->b, &unfused_x,
      &fixture->communicator, options, &fixture->error);
  if (!unfused.converged || unfused.residual > 1.0e-6f) return false;
  for (std::size_t i = 0; i < fixture->x.size(); ++i) {
    if (std::fabs(fixture->x[i] - unfused_x[i]) > 1.0e-5f) return false;
  }
  return true;
}

bool test_partition_coverage(const dis_gmres::CSRMatrix& matrix) {
  const auto row_parts = dis_gmres::partition_rows(matrix, 3, false);
  const auto nnz_parts = dis_gmres::partition_rows(matrix, 3, true);
  const auto fallback = dis_gmres::partition_rows(matrix, 0, false);
  return row_parts.front().first == 0 && row_parts.back().last == matrix.rows &&
         nnz_parts.front().first == 0 && nnz_parts.back().last == matrix.rows &&
         fallback.size() == 1 && fallback.front().first == 0 &&
         fallback.front().last == matrix.rows;
}

bool test_profile_max() {
  const std::vector<std::vector<float>> per_rank = {
      {10, 1, .5f, .5f, .4f, .3f, 2, 1, .2f, .1f, 3, 1},
      {20, 2, 1, 1, .8f, .6f, 4, 2, .4f, .2f, 2, 0},
      {30, 3, 1.5f, 1.5f, 1.2f, .9f, 6, 3, .6f, .3f, 2, 0},
      {40, 4, 2, 2, 1.6f, 1.2f, 8, 4, .8f, .4f, 1, 0}};
  const auto merged = dis_gmres::merge_profile_max(per_rank);
  const std::vector<float> expected = {40, 4, 2, 2, 1.6f, 1.2f,
                                       8, 4, .8f, .4f, 3, 1};
  for (std::size_t field = 0; field < expected.size(); ++field) {
    if (std::fabs(merged[field] - expected[field]) > 1.0e-6f) return false;
  }
  return merged[0] == 40.0f && merged[10] == 3.0f;
}

bool test_reduction_order() {
  const std::vector<std::vector<float>> repeat1 = {
      {40, 4, 2, 2, 1.6f, 1.2f, 8, 4, .8f, .4f, 3, 1},
      {10, 1, .5f, .5f, .4f, .3f, 2, 1, .2f, .1f, 1, 0}};
  const std::vector<std::vector<float>> repeat2 = {repeat1[1], repeat1[0]};
  const auto max1 = dis_gmres::merge_profile_max(repeat1);
  const auto max2 = dis_gmres::merge_profile_max(repeat2);
  const float mean_of_max = (max1[0] + max2[0]) / 2.0f;
  const float rank0_mean = (repeat1[0][0] + repeat2[0][0]) / 2.0f;
  const float rank1_mean = (repeat1[1][0] + repeat2[1][0]) / 2.0f;
  const float max_of_mean = std::max(rank0_mean, rank1_mean);
  return std::fabs(mean_of_max - 40.0f) <= 1.0e-6f &&
         std::fabs(max_of_mean - 25.0f) <= 1.0e-6f &&
         mean_of_max > max_of_mean;
}

}  // namespace

int main() {
  SolverFixture fixture;
  if (!initialize_fixture(&fixture)) {
    std::cerr << "GMRES setup/solve failed: " << fixture.error << '\n';
    return 1;
  }
  if (!test_device_regressions(&fixture) || !test_solution_variants(&fixture) ||
      !test_partition_coverage(fixture.matrix) || !test_profile_max() ||
      !test_reduction_order()) {
    std::cerr << "core regression failed\n";
    return 1;
  }
  std::cout << "core tests passed\n";
  return 0;
}

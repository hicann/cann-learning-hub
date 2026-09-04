#pragma once

#include "csr_matrix.hpp"
#include "hccl_comm.hpp"
#include "profiler.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dis_gmres {

struct GmresOptions {
  std::int32_t restart = 30;
  std::int32_t max_iterations = 10000;
  float tolerance = 1.0e-6f;
  bool communication_avoiding = true;
  bool parallel_compute = true;
  bool fused_vector_ops = true;
  bool zero_initial_guess = false;
};

struct GmresResult {
  std::int32_t iterations = 0;
  float residual = 0.0f;
  bool converged = false;
  Profile profile;
};

GmresResult distributed_gmres(const CSRMatrix& local_matrix,
                              const std::vector<RowPartition>& partitions,
                              const std::vector<float>& local_b,
                              std::vector<float>* local_x,
                              HcclCommunicator* communicator,
                              const GmresOptions& options,
                              std::string* error = nullptr);
#if DIS_GMRES_HAS_CANN
GmresResult distributed_gmres_npu(const CSRMatrix& local_matrix,
                                  const std::vector<RowPartition>& partitions,
                                  const std::vector<float>& local_b,
                                  std::vector<float>* local_x,
                                  HcclCommunicator* communicator,
                                  const GmresOptions& options,
                                  std::string* error = nullptr);
#endif

float distributed_relative_residual(const CSRMatrix& local_matrix,
                                    const std::vector<RowPartition>& partitions,
                                    const std::vector<float>& local_b,
                                    const std::vector<float>& local_x,
                                    HcclCommunicator* communicator,
                                    Profile* profile = nullptr,
                                    std::string* error = nullptr);

}  // namespace dis_gmres

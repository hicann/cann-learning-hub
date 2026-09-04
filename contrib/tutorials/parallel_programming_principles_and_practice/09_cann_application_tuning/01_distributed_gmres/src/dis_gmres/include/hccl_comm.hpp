#pragma once

#include "csr_matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#if DIS_GMRES_HAS_CANN
#include <acl/acl.h>
#include <hccl/hccl.h>
#endif

namespace dis_gmres {

struct CommStats {
  double collective_ms = 0.0;
  double transfer_ms = 0.0;
  double synchronization_ms = 0.0;
  std::size_t allreduce_calls = 0;
  std::size_t allgather_calls = 0;
};

// Profile-reduction semantics shared by main.cpp and the host tests: every
// profile field is reduced with elementwise MAX across the communicator. The
// first 10 entries are wall-time phases whose MAX is the distributed
// critical-path wall time; the last 2 entries are per-solve collective call
// counts whose MAX is the largest single-rank call count (the logical number
// of collectives one solver invocation issues). A SUM across ranks would
// artificially grow with world size and must not be used for either.
inline std::vector<float> merge_profile_max(
    const std::vector<std::vector<float>>& per_rank) {
  const std::size_t field_count = per_rank.empty() ? 0 : per_rank.front().size();
  std::vector<float> merged(field_count, 0.0f);
  for (std::size_t field = 0; field < field_count; ++field) {
    float maximum = per_rank.empty() ? 0.0f : per_rank.front()[field];
    for (const auto& rank : per_rank) maximum = std::max(maximum, rank[field]);
    merged[field] = maximum;
  }
  return merged;
}

class HcclCommunicator {
 public:
  HcclCommunicator() = default;
  ~HcclCommunicator();
  HcclCommunicator(const HcclCommunicator&) = delete;
  HcclCommunicator& operator=(const HcclCommunicator&) = delete;

  bool initialize(int device_id, int rank, int world_size, const std::string& rank_table,
                  std::string* error = nullptr);
  void finalize();

  bool allreduce_sum(const std::vector<float>& local, std::vector<float>* global,
                     std::string* error = nullptr);
  bool allreduce_sum(float local, float* global, std::string* error = nullptr);
  bool allreduce_max(const std::vector<float>& local, std::vector<float>* global,
                     std::string* error = nullptr);
  bool allgather_vector(const std::vector<float>& local,
                        const std::vector<RowPartition>& partitions,
                        std::vector<float>* global, std::string* error = nullptr);
#if DIS_GMRES_HAS_CANN
  aclrtStream stream() const { return stream_; }
  bool allreduce_device(const void* send, void* receive, std::size_t count,
                        std::string* error = nullptr);
  bool allgather_device(const void* send, void* receive, std::size_t count,
                        std::string* error = nullptr);
#endif

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  bool real_hccl() const { return real_hccl_; }
  const CommStats& stats() const { return stats_; }
  void reset_stats() { stats_ = {}; }

 private:
  bool allreduce_impl(const std::vector<float>& local, std::vector<float>* global,
                      bool maximum, std::string* error);
#if DIS_GMRES_HAS_CANN
  bool allreduce_real(const std::vector<float>& local, std::vector<float>* global,
                      bool maximum, std::string* error);
#endif
  bool ensure_reduce_capacity(std::size_t count, std::string* error);
  bool ensure_gather_capacity(std::size_t count, std::string* error);
#if DIS_GMRES_HAS_CANN
  bool allgather_host_buffers(std::size_t padded_count, std::string* error);
#endif

  int device_id_ = 0;
  int rank_ = 0;
  int world_size_ = 1;
  bool real_hccl_ = false;
  CommStats stats_;
  std::vector<float> gather_host_send_;
  std::vector<float> gather_host_recv_;

#if DIS_GMRES_HAS_CANN
  bool acl_initialized_ = false;
  aclrtStream stream_ = nullptr;
  HcclComm comm_ = nullptr;
  void* reduce_send_ = nullptr;
  void* reduce_recv_ = nullptr;
  std::size_t reduce_capacity_ = 0;
  void* gather_send_ = nullptr;
  void* gather_recv_ = nullptr;
  std::size_t gather_capacity_ = 0;
#endif
};

}  // namespace dis_gmres

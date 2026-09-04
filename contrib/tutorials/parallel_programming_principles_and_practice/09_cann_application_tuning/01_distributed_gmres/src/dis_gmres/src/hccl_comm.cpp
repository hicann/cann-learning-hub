#include "hccl_comm.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <thread>

namespace dis_gmres {
namespace {
#if DIS_GMRES_HAS_CANN
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool initialize_rank_table(const std::string& path, int rank,
                           HcclComm* communicator, HcclResult* result,
                           std::string* error) {
  std::ifstream input(path);
  if (!input) {
    if (error) *error = "rank table cannot be opened: " + path;
    return false;
  }
  *result = HcclCommInitClusterInfo(
      path.c_str(), static_cast<uint32_t>(rank), communicator);
  return true;
}

bool initialize_single_rank(HcclComm* communicator, HcclResult* result) {
  HcclRootInfo root{};
  *result = HcclGetRootInfo(&root);
  if (*result == HCCL_SUCCESS) {
    *result = HcclCommInitRootInfo(1, &root, 0, communicator);
  }
  return true;
}

std::string root_info_path() {
  const char* configured = std::getenv("DIS_GMRES_ROOT_INFO_FILE");
  return configured ? configured : "/tmp/dis_gmres_root_info.bin";
}

bool write_root_info(const std::string& path, HcclRootInfo* root,
                     HcclResult* result, std::string* error) {
  *result = HcclGetRootInfo(root);
  if (*result != HCCL_SUCCESS) return true;
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(root), sizeof(*root));
  if (output) return true;
  if (error) *error = "cannot write HCCL root info: " + path;
  return false;
}

bool wait_for_root_info(const std::string& path, HcclRootInfo* root,
                        std::string* error) {
  for (int attempt = 0; attempt < 600; ++attempt) {
    std::ifstream input(path, std::ios::binary);
    if (input) {
      input.read(reinterpret_cast<char*>(root), sizeof(*root));
      if (input) return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (error) *error = "timed out waiting for HCCL root info: " + path;
  return false;
}

bool initialize_multi_rank(int rank, int world_size, HcclComm* communicator,
                           HcclResult* result, std::string* error) {
  HcclRootInfo root{};
  const std::string path = root_info_path();
  const bool ready = rank == 0
                         ? write_root_info(path, &root, result, error)
                         : wait_for_root_info(path, &root, error);
  if (!ready) return false;
  if (*result == HCCL_SUCCESS) {
    *result = HcclCommInitRootInfo(static_cast<uint32_t>(world_size), &root,
                                   static_cast<uint32_t>(rank), communicator);
  }
  return true;
}

bool initialize_hccl(const std::string& rank_table, int rank, int world_size,
                     HcclComm* communicator, HcclResult* result,
                     std::string* error) {
  if (!rank_table.empty()) {
    return initialize_rank_table(rank_table, rank, communicator, result, error);
  }
  if (world_size == 1) return initialize_single_rank(communicator, result);
  return initialize_multi_rank(rank, world_size, communicator, result, error);
}
#endif

bool validate_allgather(const std::vector<float>& local,
                        const std::vector<RowPartition>& partitions,
                        int rank, int world_size, std::string* error) {
  if (partitions.size() != static_cast<std::size_t>(world_size)) {
    if (error) *error = "invalid AllGather vector/partition metadata";
    return false;
  }
  const auto& partition = partitions[static_cast<std::size_t>(rank)];
  if (local.size() != static_cast<std::size_t>(partition.last - partition.first)) {
    if (error) *error = "local vector does not match this rank's partition";
    return false;
  }
  return true;
}

std::size_t padded_partition_size(const std::vector<RowPartition>& partitions) {
  std::size_t padded = 0;
  for (const auto& partition : partitions) {
    padded = std::max(padded,
                      static_cast<std::size_t>(partition.last - partition.first));
  }
  return padded;
}

#if DIS_GMRES_HAS_CANN
void unpack_partitions(const std::vector<float>& packed, std::size_t padded,
                       const std::vector<RowPartition>& partitions,
                       std::vector<float>* global) {
  global->resize(static_cast<std::size_t>(partitions.back().last));
  for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
    const auto& partition = partitions[rank];
    const auto count = static_cast<std::size_t>(partition.last - partition.first);
    std::copy_n(packed.begin() + rank * padded, count,
                global->begin() + partition.first);
  }
}
#endif
}  // namespace

HcclCommunicator::~HcclCommunicator() { finalize(); }

bool HcclCommunicator::initialize(int device_id, int rank, int world_size,
                                  const std::string& rank_table, std::string* error) {
  device_id_ = device_id;
  rank_ = rank;
  world_size_ = world_size;
  if (world_size < 1 || rank < 0 || rank >= world_size) {
    if (error) *error = "invalid rank/world-size";
    return false;
  }
#if DIS_GMRES_HAS_CANN
  if (aclInit(nullptr) != ACL_SUCCESS) {
    if (error) *error = "aclInit failed";
    return false;
  }
  acl_initialized_ = true;
  if (aclrtSetDevice(device_id_) != ACL_SUCCESS ||
      aclrtCreateStream(&stream_) != ACL_SUCCESS) {
    if (error) *error = "ACL device/stream initialization failed on device " +
                        std::to_string(device_id_);
    finalize();
    return false;
  }

  HcclResult result = HCCL_SUCCESS;
  if (!initialize_hccl(rank_table, rank_, world_size_, &comm_, &result, error)) {
    finalize();
    return false;
  }
  if (result != HCCL_SUCCESS) {
    if (error)
      *error = "HCCL communicator initialization failed, code=" +
               std::to_string(static_cast<int>(result));
    finalize();
    return false;
  }
  real_hccl_ = true;
#else
  (void)device_id;
  (void)rank_table;
  if (world_size_ != 1) {
    if (error) *error = "host stub supports only world-size=1; use CANN/HCCL for multi-rank runs";
    return false;
  }
  real_hccl_ = false;
#endif
  return true;
}

bool HcclCommunicator::ensure_reduce_capacity(std::size_t count, std::string* error) {
#if DIS_GMRES_HAS_CANN
  if (count <= reduce_capacity_) return true;
  if (reduce_send_) aclrtFree(reduce_send_);
  if (reduce_recv_) aclrtFree(reduce_recv_);
  reduce_send_ = nullptr;
  reduce_recv_ = nullptr;
  reduce_capacity_ = 0;
  const auto bytes = count * sizeof(float);
  if (aclrtMalloc(&reduce_send_, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&reduce_recv_, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    if (error) *error = "failed to allocate persistent HCCL AllReduce buffers";
    return false;
  }
  reduce_capacity_ = count;
#else
  (void)count;
  (void)error;
#endif
  return true;
}

bool HcclCommunicator::ensure_gather_capacity(std::size_t count, std::string* error) {
#if DIS_GMRES_HAS_CANN
  if (count <= gather_capacity_) return true;
  if (gather_send_) aclrtFree(gather_send_);
  if (gather_recv_) aclrtFree(gather_recv_);
  gather_send_ = nullptr;
  gather_recv_ = nullptr;
  gather_capacity_ = 0;
  if (aclrtMalloc(&gather_send_, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&gather_recv_, count * static_cast<std::size_t>(world_size_) * sizeof(float),
                  ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    if (error) *error = "failed to allocate persistent HCCL AllGather buffers";
    return false;
  }
  gather_capacity_ = count;
#else
  (void)count;
  (void)error;
#endif
  return true;
}

#if DIS_GMRES_HAS_CANN
bool HcclCommunicator::allreduce_real(const std::vector<float>& local,
                                      std::vector<float>* global, bool maximum,
                                      std::string* error) {
  const std::string operation = maximum ? "AllReduce(MAX)" : "AllReduce";
  const std::string collective_operation = "Hccl" + operation;
  if (!ensure_reduce_capacity(local.size(), error)) return false;
  global->resize(local.size());
  const auto bytes = local.size() * sizeof(float);
  auto begin = Clock::now();
  if (aclrtMemcpy(reduce_send_, bytes, local.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
      ACL_SUCCESS) {
    if (error) *error = operation + " H2D copy failed";
    return false;
  }
  auto copied_in = Clock::now();
  const auto result = HcclAllReduce(reduce_send_, reduce_recv_,
                                    static_cast<uint64_t>(local.size()),
                                    HCCL_DATA_TYPE_FP32,
                                    maximum ? HCCL_REDUCE_MAX : HCCL_REDUCE_SUM,
                                    comm_, stream_);
  auto launched = Clock::now();
  const auto sync_begin = launched;
  const auto sync_result = aclrtSynchronizeStream(stream_);
  auto synchronized = Clock::now();
  if (result != HCCL_SUCCESS || sync_result != ACL_SUCCESS) {
    if (error) *error = collective_operation + " failed";
    return false;
  }
  if (aclrtMemcpy(global->data(), bytes, reduce_recv_, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
      ACL_SUCCESS) {
    if (error) *error = operation + " D2H copy failed";
    return false;
  }
  auto copied_out = Clock::now();
  stats_.transfer_ms += milliseconds(begin, copied_in) + milliseconds(synchronized, copied_out);
  stats_.collective_ms += milliseconds(copied_in, synchronized);
  stats_.synchronization_ms += milliseconds(sync_begin, synchronized);
  ++stats_.allreduce_calls;
  return true;
}
#endif

bool HcclCommunicator::allreduce_impl(const std::vector<float>& local,
                                      std::vector<float>* global, bool maximum,
                                      std::string* error) {
  const std::string operation = maximum ? "AllReduce(MAX)" : "AllReduce";
  if (!global) {
    if (error) *error = operation + " output is null";
    return false;
  }
  if (world_size_ == 1 && !real_hccl_) {
    *global = local;
    ++stats_.allreduce_calls;
    return true;
  }
#if DIS_GMRES_HAS_CANN
  return allreduce_real(local, global, maximum, error);
#else
  (void)error;
  *global = local;
  ++stats_.allreduce_calls;
  return true;
#endif
}

bool HcclCommunicator::allreduce_sum(const std::vector<float>& local,
                                     std::vector<float>* global, std::string* error) {
  return allreduce_impl(local, global, false, error);
}

bool HcclCommunicator::allreduce_sum(float local, float* global, std::string* error) {
  if (!global) {
    if (error) *error = "scalar AllReduce output is null";
    return false;
  }
  std::vector<float> output;
  if (!allreduce_sum(std::vector<float>{local}, &output, error)) return false;
  *global = output.front();
  return true;
}

// MAX reduction: used for the whole profile so the aggregated value is the
// distributed critical-path wall time for timing fields and the largest
// single-rank call count for count fields (see merge_profile_max in
// hccl_comm.hpp). A SUM/world_size rank mean would understate wall time and
// would inflate call counts with world size.
bool HcclCommunicator::allreduce_max(const std::vector<float>& local,
                                     std::vector<float>* global, std::string* error) {
  return allreduce_impl(local, global, true, error);
}

#if DIS_GMRES_HAS_CANN
bool HcclCommunicator::allreduce_device(const void* send, void* receive, std::size_t count,
                                        std::string* error) {
  const auto begin = Clock::now();
  // HCCL 9.x declares the AllReduce send buffer as void*; keep the upper-layer
  // read-only contract and only drop const at this API boundary.
  void* send_buffer = const_cast<void*>(send);
  const auto result = HcclAllReduce(send_buffer, receive, static_cast<uint64_t>(count),
                                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm_, stream_);
  const auto launched = Clock::now();
  const auto sync = aclrtSynchronizeStream(stream_);
  const auto end = Clock::now();
  stats_.collective_ms += milliseconds(begin, end);
  stats_.synchronization_ms += milliseconds(launched, end);
  ++stats_.allreduce_calls;
  if (result != HCCL_SUCCESS || sync != ACL_SUCCESS) {
    if (error) *error = "HcclAllReduce(Device buffer) failed";
    return false;
  }
  return true;
}

bool HcclCommunicator::allgather_device(const void* send, void* receive, std::size_t count,
                                        std::string* error) {
  const auto begin = Clock::now();
  // HCCL 9.x declares the AllGather send buffer as void*; only drop const here.
  void* send_buffer = const_cast<void*>(send);
  const auto result = HcclAllGather(send_buffer, receive, static_cast<uint64_t>(count),
                                    HCCL_DATA_TYPE_FP32, comm_, stream_);
  const auto launched = Clock::now();
  const auto sync = aclrtSynchronizeStream(stream_);
  const auto end = Clock::now();
  stats_.collective_ms += milliseconds(begin, end);
  stats_.synchronization_ms += milliseconds(launched, end);
  ++stats_.allgather_calls;
  if (result != HCCL_SUCCESS || sync != ACL_SUCCESS) {
    if (error) *error = "HcclAllGather(Device buffer) failed";
    return false;
  }
  return true;
}
#endif

#if DIS_GMRES_HAS_CANN
bool HcclCommunicator::allgather_host_buffers(std::size_t padded_count,
                                              std::string* error) {
  if (!ensure_gather_capacity(padded_count, error)) return false;
  gather_host_recv_.resize(padded_count * static_cast<std::size_t>(world_size_));
  const auto send_bytes = padded_count * sizeof(float);
  const auto receive_bytes = gather_host_recv_.size() * sizeof(float);
  const auto begin = Clock::now();
  if (aclrtMemcpy(gather_send_, send_bytes, gather_host_send_.data(), send_bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    if (error) *error = "AllGather H2D copy failed";
    return false;
  }
  const auto copied_in = Clock::now();
  const auto result = HcclAllGather(gather_send_, gather_recv_,
                                    static_cast<uint64_t>(padded_count),
                                    HCCL_DATA_TYPE_FP32, comm_, stream_);
  const auto launched = Clock::now();
  const auto sync_result = aclrtSynchronizeStream(stream_);
  const auto synchronized = Clock::now();
  if (result != HCCL_SUCCESS || sync_result != ACL_SUCCESS) {
    if (error) *error = "HcclAllGather failed";
    return false;
  }
  if (aclrtMemcpy(gather_host_recv_.data(), receive_bytes, gather_recv_, receive_bytes,
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    if (error) *error = "AllGather D2H copy failed";
    return false;
  }
  const auto copied_out = Clock::now();
  stats_.transfer_ms += milliseconds(begin, copied_in) +
                        milliseconds(synchronized, copied_out);
  stats_.collective_ms += milliseconds(copied_in, synchronized);
  stats_.synchronization_ms += milliseconds(launched, synchronized);
  ++stats_.allgather_calls;
  return true;
}
#endif

bool HcclCommunicator::allgather_vector(const std::vector<float>& local,
                                        const std::vector<RowPartition>& partitions,
                                        std::vector<float>* global, std::string* error) {
  if (!global) {
    if (error) *error = "invalid AllGather vector/partition metadata";
    return false;
  }
  if (!validate_allgather(local, partitions, rank_, world_size_, error)) return false;
  if (world_size_ == 1 && !real_hccl_) {
    *global = local;
    ++stats_.allgather_calls;
    return true;
  }
  const std::size_t padded_count = padded_partition_size(partitions);
  gather_host_send_.assign(padded_count, 0.0f);
  std::copy(local.begin(), local.end(), gather_host_send_.begin());
#if DIS_GMRES_HAS_CANN
  if (!allgather_host_buffers(padded_count, error)) return false;
  unpack_partitions(gather_host_recv_, padded_count, partitions, global);
  return true;
#else
  (void)error;
  *global = local;
  ++stats_.allgather_calls;
  return true;
#endif
}

void HcclCommunicator::finalize() {
#if DIS_GMRES_HAS_CANN
  if (comm_) {
    HcclCommDestroy(comm_);
    comm_ = nullptr;
  }
  if (reduce_send_) aclrtFree(reduce_send_);
  if (reduce_recv_) aclrtFree(reduce_recv_);
  if (gather_send_) aclrtFree(gather_send_);
  if (gather_recv_) aclrtFree(gather_recv_);
  reduce_send_ = reduce_recv_ = gather_send_ = gather_recv_ = nullptr;
  reduce_capacity_ = gather_capacity_ = 0;
  if (stream_) {
    aclrtDestroyStream(stream_);
    stream_ = nullptr;
  }
  if (acl_initialized_) {
    aclrtResetDevice(device_id_);
    aclFinalize();
    acl_initialized_ = false;
  }
#endif
  real_hccl_ = false;
}

}  // namespace dis_gmres

#pragma once
#include "spmv.hpp"
#include <cstdint>
#include <string>
#include <vector>
#if HCCL_SPMV_HAS_CANN
#include <acl/acl.h>
#include <hccl/hccl.h>
#endif
namespace hccl_spmv {
struct Timings { double communication_ms=0, transfer_ms=0, kernel_launch_overhead_ms=0, local_spmv_launch_to_complete_ms=0, synchronization_ms=0, total_ms=0; };
class HcclContext {
 public:
  ~HcclContext();
  bool initialize(int device,int rank,int world,const std::string& rank_table,std::string* err=nullptr);
  bool broadcast_device(const std::vector<float>& host, void** device, std::string* err=nullptr);
  bool allgather_device(const void* local, std::size_t count, std::vector<float>* gathered, std::string* err=nullptr);
  // MAX reduction over the communicator: used to aggregate wall-time timings so
  // the result is the distributed critical-path time (a rank mean is not a wall
  // time). Real HCCL REDUCE_MAX path with persistent device buffers.
  bool allreduce_max(const std::vector<float>& local, std::vector<float>* global, std::string* err=nullptr);
  void finalize();
  int rank() const { return rank_; } int world() const { return world_; }
  bool real_hccl() const { return real_; }
  double last_collective_ms() const { return last_collective_ms_; }
  double last_transfer_ms() const { return last_transfer_ms_; }
#if HCCL_SPMV_HAS_CANN
  aclrtStream stream() const { return stream_; }
#endif
 private:
  int rank_=0,world_=1,device_=0; bool real_=false;
  double last_collective_ms_=0,last_transfer_ms_=0;
#if HCCL_SPMV_HAS_CANN
  aclrtStream stream_=nullptr; HcclComm comm_=nullptr; bool acl_ready_=false;
  void* broadcast_buffer_=nullptr; std::size_t broadcast_bytes_=0;
  void* gather_send_buffer_=nullptr; void* gather_recv_buffer_=nullptr;
  std::size_t gather_send_bytes_=0; std::size_t gather_recv_bytes_=0;
  void* reduce_send_buffer_=nullptr; void* reduce_recv_buffer_=nullptr;
  std::size_t reduce_bytes_=0;
#endif
};
class DeviceSpmv {
 public:
  DeviceSpmv(); ~DeviceSpmv();
  bool prepare(HcclContext&,const CSRMatrix&,int first,int last,int chunk,std::string* err=nullptr);
  bool run(const void* device_x,void** device_y,double* launch_overhead_ms,double* sync_ms,std::string* err=nullptr);
 private: struct Impl; Impl* impl_;
};
bool distributed_spmv(HcclContext&,DeviceSpmv&,const std::vector<float>&,std::vector<float>*,Timings*,std::string*);
}

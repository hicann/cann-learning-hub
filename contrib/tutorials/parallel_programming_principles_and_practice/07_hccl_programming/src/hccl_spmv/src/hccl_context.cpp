#include "hccl_context.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
namespace hccl_spmv {
#if HCCL_SPMV_HAS_CANN

namespace {

bool initialize_from_rank_table(const std::string& table, int rank,
                                HcclComm* communicator, HcclResult* result,
                                std::string* error) {
  std::ifstream rank_file(table);
  if (!rank_file.good()) {
    if (error) *error = "rank table cannot be opened: " + table;
    return false;
  }
  *result = HcclCommInitClusterInfo(
      table.c_str(), static_cast<uint32_t>(rank), communicator);
  return true;
}

std::string root_info_path() {
  const char* value = std::getenv("HCCL_SPMV_ROOT_INFO_FILE");
  return value ? value : "/tmp/hccl_spmv_root_info.bin";
}

bool write_root_info(const std::string& path, HcclRootInfo* root,
                     HcclResult* result, std::string* error) {
  *result = HcclGetRootInfo(root);
  if (*result != HCCL_SUCCESS) return true;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(root), sizeof(*root));
  if (out) return true;
  if (error) *error = "cannot write root info file: " + path;
  return false;
}

bool wait_for_root_info(const std::string& path, HcclRootInfo* root,
                        std::string* error) {
  for (int attempt = 0; attempt < 600; ++attempt) {
    std::ifstream in(path, std::ios::binary);
    if (in) {
      in.read(reinterpret_cast<char*>(root), sizeof(*root));
      if (in) return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (error) *error = "timed out waiting for root info file: " + path;
  return false;
}

bool initialize_from_root_info(int rank, int world, HcclComm* communicator,
                               HcclResult* result, std::string* error) {
  HcclRootInfo root{};
  const std::string path = root_info_path();
  const bool ready = rank == 0
                         ? write_root_info(path, &root, result, error)
                         : wait_for_root_info(path, &root, error);
  if (!ready) return false;
  if (*result == HCCL_SUCCESS) {
    *result = HcclCommInitRootInfo(static_cast<uint32_t>(world), &root,
                                   static_cast<uint32_t>(rank), communicator);
  }
  return true;
}

bool initialize_communicator(const std::string& table, int rank, int world,
                             HcclComm* communicator, HcclResult* result,
                             std::string* error) {
  if (!table.empty()) {
    return initialize_from_rank_table(table, rank, communicator, result, error);
  }
  return initialize_from_root_info(rank, world, communicator, result, error);
}

}  // namespace
#endif

bool HcclContext::initialize(int device, int rank, int world,
                             const std::string& table, std::string* error) {
  rank_ = rank;
  world_ = world;
  device_ = device;
  if (world_ < 1 || rank_ < 0 || rank_ >= world_) {
    if (error) *error = "invalid rank/world";
    return false;
  }
#if HCCL_SPMV_HAS_CANN
  if (aclInit(nullptr) != ACL_SUCCESS) {
    if (error) *error = "aclInit failed";
    return false;
  }
  if (aclrtSetDevice(device) != ACL_SUCCESS) {
    aclFinalize();
    if (error) *error = "aclrtSetDevice failed for device " + std::to_string(device);
    return false;
  }
  acl_ready_ = true;
  if (aclrtCreateStream(&stream_) != ACL_SUCCESS) {
    if (error) *error = "aclrtCreateStream failed";
    finalize();
    return false;
  }
  HcclResult result = HCCL_SUCCESS;
  if (!initialize_communicator(table, rank_, world_, &comm_, &result, error)) {
    finalize();
    return false;
  }
  if (result != HCCL_SUCCESS) {
    if (error) {
      *error = "Hccl communicator initialization failed: " +
          std::to_string(static_cast<int>(result)) + " (rank=" +
          std::to_string(rank_) + ", world=" + std::to_string(world_) +
          ", device=" + std::to_string(device_) + ", table=" + table + ")";
    }
    finalize();
    return false;
  }
  real_ = true;
#else
  (void)device;
  (void)table;
  real_ = false;
#endif
  return true;
}
HcclContext::~HcclContext(){finalize();}
bool HcclContext::broadcast_device(const std::vector<float>& host,void** device,std::string* e){
#if HCCL_SPMV_HAS_CANN
  if(!device){if(e)*e="null device broadcast output";return false;} const size_t bytes=host.size()*sizeof(float);
  if(bytes!=broadcast_bytes_){if(broadcast_buffer_)aclrtFree(broadcast_buffer_);broadcast_buffer_=nullptr;broadcast_bytes_=0;if(aclrtMalloc(&broadcast_buffer_,bytes,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS){if(e)*e="ACL input allocation failed";return false;}broadcast_bytes_=bytes;}
  auto t0=std::chrono::steady_clock::now(); if(rank_==0&&aclrtMemcpy(broadcast_buffer_,bytes,host.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE)!=ACL_SUCCESS){if(e)*e="H2D broadcast input failed";return false;} auto t1=std::chrono::steady_clock::now();
  if(HcclBroadcast(broadcast_buffer_,static_cast<uint64_t>(host.size()),HCCL_DATA_TYPE_FP32,0,comm_,stream_)!=HCCL_SUCCESS||aclrtSynchronizeStream(stream_)!=ACL_SUCCESS){if(e)*e="HcclBroadcast device buffer failed";return false;} auto t2=std::chrono::steady_clock::now();
  last_transfer_ms_=std::chrono::duration<double,std::milli>(t1-t0).count();last_collective_ms_=std::chrono::duration<double,std::milli>(t2-t1).count();*device=broadcast_buffer_;return true;
#else
  (void)host;(void)device;if(e)*e="device broadcast is unavailable in Host Stub";return false;
#endif
}
bool HcclContext::allgather_device(const void* local,std::size_t count,std::vector<float>* gathered,std::string* e){
#if HCCL_SPMV_HAS_CANN
  if(!local||!gathered){if(e)*e="null device allgather buffer";return false;} gathered->assign(count*static_cast<size_t>(world_),0.0f); const size_t bytes=gathered->size()*sizeof(float);
  if(bytes!=gather_recv_bytes_){if(gather_recv_buffer_)aclrtFree(gather_recv_buffer_);gather_recv_buffer_=nullptr;gather_recv_bytes_=0;if(aclrtMalloc(&gather_recv_buffer_,bytes,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS){if(e)*e="ACL allgather receive allocation failed";return false;}gather_recv_bytes_=bytes;}
  // HCCL 9.x declares the AllGather send buffer as void*; keep the upper-layer
  // read-only contract and only drop const at this API boundary.
  void* send_buffer=const_cast<void*>(local);
  auto t0=std::chrono::steady_clock::now(); if(HcclAllGather(send_buffer,gather_recv_buffer_,static_cast<uint64_t>(count),HCCL_DATA_TYPE_FP32,comm_,stream_)!=HCCL_SUCCESS||aclrtSynchronizeStream(stream_)!=ACL_SUCCESS){if(e)*e="HcclAllGather device buffer failed";return false;} auto t1=std::chrono::steady_clock::now();
  if(aclrtMemcpy(gathered->data(),bytes,gather_recv_buffer_,bytes,ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS){if(e)*e="D2H allgather output failed";return false;} auto t2=std::chrono::steady_clock::now();last_collective_ms_+=std::chrono::duration<double,std::milli>(t1-t0).count();last_transfer_ms_+=std::chrono::duration<double,std::milli>(t2-t1).count();return true;
#else
  (void)local;(void)count;(void)gathered;if(e)*e="device allgather is unavailable in Host Stub";return false;
#endif
}
bool HcclContext::allreduce_max(const std::vector<float>& local,std::vector<float>* global,std::string* e){
#if HCCL_SPMV_HAS_CANN
  if(!global){if(e)*e="null AllReduce(MAX) output";return false;}
  const size_t bytes=local.size()*sizeof(float);
  if(bytes!=reduce_bytes_){
    if(reduce_send_buffer_)aclrtFree(reduce_send_buffer_);
    if(reduce_recv_buffer_)aclrtFree(reduce_recv_buffer_);
    reduce_send_buffer_=nullptr;reduce_recv_buffer_=nullptr;reduce_bytes_=0;
    if(aclrtMalloc(&reduce_send_buffer_,bytes,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS||
       aclrtMalloc(&reduce_recv_buffer_,bytes,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS){
      if(reduce_send_buffer_){aclrtFree(reduce_send_buffer_);reduce_send_buffer_=nullptr;}
      if(reduce_recv_buffer_){aclrtFree(reduce_recv_buffer_);reduce_recv_buffer_=nullptr;}
      if(e)*e="ACL AllReduce(MAX) buffer allocation failed";return false;
    }
    reduce_bytes_=bytes;
  }
  global->resize(local.size());
  if(aclrtMemcpy(reduce_send_buffer_,bytes,local.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE)!=ACL_SUCCESS){if(e)*e="H2D AllReduce(MAX) input failed";return false;}
  if(HcclAllReduce(reduce_send_buffer_,reduce_recv_buffer_,static_cast<uint64_t>(local.size()),HCCL_DATA_TYPE_FP32,HCCL_REDUCE_MAX,comm_,stream_)!=HCCL_SUCCESS||aclrtSynchronizeStream(stream_)!=ACL_SUCCESS){if(e)*e="HcclAllReduce(MAX) failed";return false;}
  if(aclrtMemcpy(global->data(),bytes,reduce_recv_buffer_,bytes,ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS){if(e)*e="D2H AllReduce(MAX) output failed";return false;}
  return true;
#else
  (void)local;(void)global;if(e)*e="AllReduce(MAX) is unavailable in Host Stub";return false;
#endif
}
void HcclContext::finalize(){
#if HCCL_SPMV_HAS_CANN
  if(comm_) { HcclCommDestroy(comm_); comm_=nullptr; } if(broadcast_buffer_){aclrtFree(broadcast_buffer_);broadcast_buffer_=nullptr;} if(gather_send_buffer_){aclrtFree(gather_send_buffer_);gather_send_buffer_=nullptr;} if(gather_recv_buffer_){aclrtFree(gather_recv_buffer_);gather_recv_buffer_=nullptr;} if(reduce_send_buffer_){aclrtFree(reduce_send_buffer_);reduce_send_buffer_=nullptr;} if(reduce_recv_buffer_){aclrtFree(reduce_recv_buffer_);reduce_recv_buffer_=nullptr;} if(stream_){aclrtDestroyStream(stream_);stream_=nullptr;} if(acl_ready_){aclrtResetDevice(device_);aclFinalize();acl_ready_=false;}
#endif
  real_=false;
}
}

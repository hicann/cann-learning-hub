#pragma once
#include "csr_matrix.hpp"
#include "profiler.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#if DIS_GMRES_HAS_CANN
#include <acl/acl.h>
#endif
namespace dis_gmres {
struct DeviceVector { void* data=nullptr; std::size_t size=0; };
class NpuCompute {
 public:
  explicit NpuCompute(
#if DIS_GMRES_HAS_CANN
      aclrtStream stream
#else
      void* stream
#endif
  );
  ~NpuCompute(); NpuCompute(const NpuCompute&)=delete; NpuCompute& operator=(const NpuCompute&)=delete;
  bool prepare(const CSRMatrix&,std::string*); bool allocate(std::size_t,DeviceVector*,std::string*);
  void release(DeviceVector*); bool upload(const std::vector<float>&,DeviceVector*,std::string*);
  bool download(const DeviceVector&,std::vector<float>*,std::string*); bool copy(const DeviceVector&,DeviceVector*,std::string*);
  bool spmv(const DeviceVector&,DeviceVector*,Profile*,std::string*); bool dot(const DeviceVector&,const DeviceVector&,DeviceVector*,Profile*,std::string*);
  bool axpy(float,const DeviceVector&,DeviceVector*,Profile*,std::string*); bool scale(float,DeviceVector*,Profile*,std::string*);
  bool subtract(const DeviceVector&,const DeviceVector&,DeviceVector*,Profile*,std::string*);
 private: struct Impl; std::unique_ptr<Impl> p_;
};
}

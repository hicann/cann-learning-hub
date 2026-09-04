#include "hccl_context.hpp"
#if HCCL_SPMV_HAS_CANN
#include <acl/acl_rt_compile.h>
#include <chrono>
#include <fstream>
#include <sstream>
#endif
namespace hccl_spmv {
struct DeviceSpmv::Impl {
#if HCCL_SPMV_HAS_CANN
  aclrtStream stream=nullptr; aclrtcProg program=nullptr; aclrtBinHandle binary=nullptr; aclrtFuncHandle function=nullptr; aclrtArgsHandle args=nullptr; aclrtParamHandle param=nullptr;
  void *row=nullptr,*col=nullptr,*val=nullptr,*y=nullptr; std::vector<char> elf; int64_t first=0,last=0; int32_t blocks=32; bool ready=false;
#endif
};
DeviceSpmv::DeviceSpmv():impl_(new Impl){}
DeviceSpmv::~DeviceSpmv(){
#if HCCL_SPMV_HAS_CANN
 if(impl_->binary)aclrtBinaryUnLoad(impl_->binary);if(impl_->y)aclrtFree(impl_->y);if(impl_->val)aclrtFree(impl_->val);if(impl_->col)aclrtFree(impl_->col);if(impl_->row)aclrtFree(impl_->row);if(impl_->program)aclrtcDestroyProg(&impl_->program);
#endif
 delete impl_;
}
#if HCCL_SPMV_HAS_CANN
namespace { bool ok(aclError r,const char* call,std::string* e){if(r==ACL_ERROR_NONE)return true;if(e)*e=std::string(call)+" failed, aclError="+std::to_string(r);return false;} }
#endif
bool DeviceSpmv::prepare(HcclContext& ctx,const CSRMatrix& m,int first,int last,int chunk,std::string* e){
#if HCCL_SPMV_HAS_CANN
 impl_->stream=ctx.stream();impl_->first=first;impl_->last=last;std::ifstream in(HCCL_SPMV_KERNEL_SOURCE);std::ostringstream ss;ss<<in.rdbuf();const std::string src=ss.str();if(src.empty()){if(e)*e="local Ascend C kernel source is empty";return false;}
 if(!ok(aclrtcCreateProg(&impl_->program,src.c_str(),"spmv_local_fp32",0,nullptr,nullptr),"aclrtcCreateProg",e))return false;const char* opts[]={"--npu-arch=dav-2201"};if(!ok(aclrtcCompileProg(impl_->program,1,opts),"aclrtcCompileProg",e))return false;size_t n=0;if(!ok(aclrtcGetBinDataSize(impl_->program,&n),"aclrtcGetBinDataSize",e))return false;impl_->elf.resize(n);if(!ok(aclrtcGetBinData(impl_->program,impl_->elf.data()),"aclrtcGetBinData",e))return false;
 aclrtBinaryLoadOption opt{};opt.type=ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC;opt.value.magic=ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;aclrtBinaryLoadOptions load{};load.options=&opt;load.numOpt=1;if(!ok(aclrtBinaryLoadFromData(impl_->elf.data(),n,&load,&impl_->binary),"aclrtBinaryLoadFromData",e)||!ok(aclrtBinaryGetFunction(impl_->binary,"spmv_local_fp32",&impl_->function),"aclrtBinaryGetFunction",e))return false;
 size_t rb=m.row_ptr.size()*sizeof(int32_t),cb=m.col_idx.size()*sizeof(int32_t),vb=m.values.size()*sizeof(float),yb=static_cast<size_t>(chunk)*sizeof(float);if(!ok(aclrtMalloc(&impl_->row,rb,ACL_MEM_MALLOC_HUGE_FIRST),"aclrtMalloc row",e)||!ok(aclrtMalloc(&impl_->col,cb,ACL_MEM_MALLOC_HUGE_FIRST),"aclrtMalloc col",e)||!ok(aclrtMalloc(&impl_->val,vb,ACL_MEM_MALLOC_HUGE_FIRST),"aclrtMalloc values",e)||!ok(aclrtMalloc(&impl_->y,yb,ACL_MEM_MALLOC_HUGE_FIRST),"aclrtMalloc local y",e))return false;
 if(!ok(aclrtMemset(impl_->y,yb,0,yb),"aclrtMemset local y",e))return false;
 if(!ok(aclrtMemcpy(impl_->row,rb,m.row_ptr.data(),rb,ACL_MEMCPY_HOST_TO_DEVICE),"H2D row",e)||!ok(aclrtMemcpy(impl_->col,cb,m.col_idx.data(),cb,ACL_MEMCPY_HOST_TO_DEVICE),"H2D col",e)||!ok(aclrtMemcpy(impl_->val,vb,m.values.data(),vb,ACL_MEMCPY_HOST_TO_DEVICE),"H2D values",e))return false;
 impl_->ready=true;return true;
#else
 (void)ctx;(void)m;(void)first;(void)last;(void)chunk;if(e)*e="local NPU SpMV requires real CANN";return false;
#endif
}
bool DeviceSpmv::run(const void* x,void** y,double* launch_overhead_ms,double* sync_ms,std::string* e){
#if HCCL_SPMV_HAS_CANN
 if(!impl_->ready||!x||!y){if(e)*e="DeviceSpmv is not ready";return false;}if(!impl_->args){if(!ok(aclrtKernelArgsInit(impl_->function,&impl_->args),"aclrtKernelArgsInit",e))return false;void* mutable_x=const_cast<void*>(x);auto ap=[&](void* p,size_t n,const char* s){return ok(aclrtKernelArgsAppend(impl_->args,reinterpret_cast<void**>(p),n,&impl_->param),s,e);};if(!ap(&impl_->row,sizeof(uintptr_t),"arg row")||!ap(&impl_->col,sizeof(uintptr_t),"arg col")||!ap(&impl_->val,sizeof(uintptr_t),"arg values")||!ap(&mutable_x,sizeof(uintptr_t),"arg x")||!ap(&impl_->y,sizeof(uintptr_t),"arg y")||!ap(&impl_->first,sizeof(int64_t),"arg first")||!ap(&impl_->last,sizeof(int64_t),"arg last")||!ok(aclrtKernelArgsFinalize(impl_->args),"aclrtKernelArgsFinalize",e))return false;}
 auto k0=std::chrono::steady_clock::now();if(!ok(aclrtLaunchKernelWithConfig(impl_->function,static_cast<uint32_t>(impl_->blocks),impl_->stream,nullptr,impl_->args,nullptr),"aclrtLaunchKernelWithConfig",e))return false;auto s0=std::chrono::steady_clock::now();if(!ok(aclrtSynchronizeStream(impl_->stream),"aclrtSynchronizeStream",e))return false;auto s1=std::chrono::steady_clock::now();if(launch_overhead_ms)*launch_overhead_ms=std::chrono::duration<double,std::milli>(s0-k0).count();if(sync_ms)*sync_ms=std::chrono::duration<double,std::milli>(s1-s0).count();*y=impl_->y;return true;
#else
 (void)x;(void)y;(void)launch_overhead_ms;(void)sync_ms;if(e)*e="local NPU SpMV requires real CANN";return false;
#endif
}
}

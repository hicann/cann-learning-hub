#include "npu_compute.hpp"
#if DIS_GMRES_HAS_CANN
#include <acl/acl_rt_compile.h>
#include <chrono>
#include <fstream>
#include <sstream>
#endif
namespace dis_gmres {
struct NpuCompute::Impl {
#if DIS_GMRES_HAS_CANN
 aclrtStream stream=nullptr; aclrtcProg prog=nullptr; aclrtBinHandle bin=nullptr;
 aclrtFuncHandle spmv=nullptr,dot=nullptr,axpy=nullptr,scale=nullptr,sub=nullptr;
  void *row=nullptr,*col=nullptr,*val=nullptr; int64_t rows=0; int32_t blocks=32; std::vector<char> elf;
#endif
};
#if DIS_GMRES_HAS_CANN
namespace {using Clock=std::chrono::steady_clock;double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}bool ok(aclError r,const char*n,std::string*e){if(r==ACL_ERROR_NONE)return true;if(e)*e=std::string(n)+" failed, aclError="+std::to_string(r);return false;}}
#endif
NpuCompute::NpuCompute(
#if DIS_GMRES_HAS_CANN
aclrtStream s
#else
void* s
#endif
):p_(new Impl){
#if DIS_GMRES_HAS_CANN
p_->stream=s;
#else
(void)s;
#endif
}
NpuCompute::~NpuCompute(){
#if DIS_GMRES_HAS_CANN
 if(p_->bin)aclrtBinaryUnLoad(p_->bin);
 if(p_->val)aclrtFree(p_->val);
 if(p_->col)aclrtFree(p_->col);
 if(p_->row)aclrtFree(p_->row);
 if(p_->prog)aclrtcDestroyProg(&p_->prog);
#endif
}
bool NpuCompute::prepare(const CSRMatrix&m,std::string*e){
#if DIS_GMRES_HAS_CANN
 std::ifstream in(DIS_GMRES_KERNEL_SOURCE);std::ostringstream ss;ss<<in.rdbuf();auto src=ss.str();
 if(src.empty()){
  if(e)*e="GMRES Ascend C source is empty";
  return false;
 }
 if(!ok(aclrtcCreateProg(&p_->prog,src.c_str(),"gmres_ops",0,nullptr,nullptr),"aclrtcCreateProg",e))return false;
 const char*opts[]={"--npu-arch=dav-2201"};
 if(!ok(aclrtcCompileProg(p_->prog,1,opts),"aclrtcCompileProg",e))return false;
 size_t n=0;
 if(!ok(aclrtcGetBinDataSize(p_->prog,&n),"aclrtcGetBinDataSize",e))return false;
 p_->elf.resize(n);
 if(!ok(aclrtcGetBinData(p_->prog,p_->elf.data()),"aclrtcGetBinData",e))return false;
 aclrtBinaryLoadOption o{};o.type=ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC;o.value.magic=ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;aclrtBinaryLoadOptions os{};os.options=&o;os.numOpt=1;
 if(!ok(aclrtBinaryLoadFromData(p_->elf.data(),n,&os,&p_->bin),"aclrtBinaryLoadFromData",e))return false;
 auto f=[&](const char*n,aclrtFuncHandle*h){return ok(aclrtBinaryGetFunction(p_->bin,n,h),n,e);};if(!f("gmres_spmv",&p_->spmv)||!f("gmres_dot",&p_->dot)||!f("gmres_axpy",&p_->axpy)||!f("gmres_scale",&p_->scale)||!f("gmres_sub",&p_->sub))return false;
 p_->rows=m.rows;size_t rb=m.row_ptr.size()*4,cb=m.col_idx.size()*4,vb=m.values.size()*4;
 if(!ok(aclrtMalloc(&p_->row,rb,ACL_MEM_MALLOC_HUGE_FIRST),"malloc row",e)||!ok(aclrtMalloc(&p_->col,cb,ACL_MEM_MALLOC_HUGE_FIRST),"malloc col",e)||!ok(aclrtMalloc(&p_->val,vb,ACL_MEM_MALLOC_HUGE_FIRST),"malloc val",e))return false;
 return ok(aclrtMemcpy(p_->row,rb,m.row_ptr.data(),rb,ACL_MEMCPY_HOST_TO_DEVICE),"copy row",e)&&ok(aclrtMemcpy(p_->col,cb,m.col_idx.data(),cb,ACL_MEMCPY_HOST_TO_DEVICE),"copy col",e)&&ok(aclrtMemcpy(p_->val,vb,m.values.data(),vb,ACL_MEMCPY_HOST_TO_DEVICE),"copy val",e);
#else
(void)m;if(e)*e="real Ascend C compute is required";return false;
#endif
}
bool NpuCompute::allocate(std::size_t n,DeviceVector*v,std::string*e){if(!v)return false;
#if DIS_GMRES_HAS_CANN
 v->size=n;return ok(aclrtMalloc(&v->data,n*sizeof(float),ACL_MEM_MALLOC_HUGE_FIRST),"aclrtMalloc vector",e);
#else
(void)n;(void)e;return false;
#endif
}
void NpuCompute::release(DeviceVector*v){
#if DIS_GMRES_HAS_CANN
if(v&&v->data)aclrtFree(v->data);
#endif
if(v){v->data=nullptr;v->size=0;}}
bool NpuCompute::upload(const std::vector<float>&h,DeviceVector*v,std::string*e){if(!v->data&&!allocate(h.size(),v,e))return false;
#if DIS_GMRES_HAS_CANN
return h.size()==v->size&&ok(aclrtMemcpy(v->data,h.size()*4,h.data(),h.size()*4,ACL_MEMCPY_HOST_TO_DEVICE),"H2D vector",e);
#else
(void)h;(void)e;return false;
#endif
}
bool NpuCompute::download(const DeviceVector&v,std::vector<float>*h,std::string*e){if(!h)return false;
#if DIS_GMRES_HAS_CANN
h->resize(v.size);return ok(aclrtMemcpy(h->data(),v.size*4,v.data,v.size*4,ACL_MEMCPY_DEVICE_TO_HOST),"D2H vector",e);
#else
(void)v;(void)e;return false;
#endif
}
bool NpuCompute::copy(const DeviceVector&a,DeviceVector*b,std::string*e){
#if DIS_GMRES_HAS_CANN
 if(!b->data&&!allocate(a.size,b,e))return false;
 return ok(aclrtMemcpy(b->data,a.size*4,a.data,a.size*4,ACL_MEMCPY_DEVICE_TO_DEVICE),"D2D vector",e);
#else
(void)a;(void)b;(void)e;return false;
#endif
}
#if DIS_GMRES_HAS_CANN
namespace {bool launch(aclrtFuncHandle f,uint32_t blocks,aclrtStream s,std::initializer_list<std::pair<void*,size_t>> av,Profile*p,double*field,std::string*e){
 aclrtArgsHandle a=nullptr;aclrtParamHandle ph=nullptr;
 if(!ok(aclrtKernelArgsInit(f,&a),"KernelArgsInit",e))return false;
 for(auto&x:av){
  if(!ok(aclrtKernelArgsAppend(a,reinterpret_cast<void**>(x.first),x.second,&ph),"KernelArgsAppend",e))return false;
 }
 if(!ok(aclrtKernelArgsFinalize(a),"KernelArgsFinalize",e))return false;
 auto t0=Clock::now();
 if(!ok(aclrtLaunchKernelWithConfig(f,blocks,s,nullptr,a,nullptr),"LaunchKernel",e))return false;
 auto t1=Clock::now();
 if(!ok(aclrtSynchronizeStream(s),"SynchronizeStream",e))return false;
 auto t2=Clock::now();
 if(p){p->kernel_launch_ms+=elapsed(t0,t1);p->synchronization_ms+=elapsed(t1,t2);if(field)*field+=elapsed(t0,t2);}
 return true;
}}
#endif
bool NpuCompute::spmv(const DeviceVector&x,DeviceVector*y,Profile*p,std::string*e){
#if DIS_GMRES_HAS_CANN
 if(!y->data&&!allocate(p_->rows,y,e))return false;
 return launch(p_->spmv,p_->blocks,p_->stream,{{&p_->row,sizeof(uintptr_t)},{&p_->col,sizeof(uintptr_t)},{&p_->val,sizeof(uintptr_t)},{const_cast<void**>(&x.data),sizeof(uintptr_t)},{&y->data,sizeof(uintptr_t)},{&p_->rows,8}},p,p?&p->spmv_ms:nullptr,e);
#else
(void)x;(void)y;(void)p;(void)e;return false;
#endif
}
bool NpuCompute::dot(const DeviceVector&a,const DeviceVector&b,DeviceVector*out,Profile*p,std::string*e){
#if DIS_GMRES_HAS_CANN
 if(!out->data&&!allocate(1,out,e))return false;
 int64_t n=a.size;
 return launch(p_->dot,1,p_->stream,{{const_cast<void**>(&a.data),sizeof(uintptr_t)},{const_cast<void**>(&b.data),sizeof(uintptr_t)},{&out->data,sizeof(uintptr_t)},{&n,8}},p,p?&p->dot_ms:nullptr,e);
#else
(void)a;(void)b;(void)out;(void)p;(void)e;return false;
#endif
}
bool NpuCompute::axpy(float alpha,const DeviceVector&x,DeviceVector*y,Profile*p,std::string*e){
#if DIS_GMRES_HAS_CANN
int64_t n=x.size;return launch(p_->axpy,p_->blocks,p_->stream,{{const_cast<void**>(&x.data),sizeof(uintptr_t)},{&y->data,sizeof(uintptr_t)},{&n,8},{&alpha,4}},p,p?&p->axpy_ms:nullptr,e);
#else
(void)alpha;(void)x;(void)y;(void)p;(void)e;return false;
#endif
}
bool NpuCompute::scale(float alpha,DeviceVector*x,Profile*p,std::string*e){
#if DIS_GMRES_HAS_CANN
int64_t n=x->size;return launch(p_->scale,p_->blocks,p_->stream,{{&x->data,sizeof(uintptr_t)},{&n,8},{&alpha,4}},p,p?&p->axpy_ms:nullptr,e);
#else
(void)alpha;(void)x;(void)p;(void)e;return false;
#endif
}
bool NpuCompute::subtract(const DeviceVector&a,const DeviceVector&b,DeviceVector*out,Profile*p,std::string*e){
#if DIS_GMRES_HAS_CANN
 if(!out->data&&!allocate(a.size,out,e))return false;
 int64_t n=a.size;
 return launch(p_->sub,p_->blocks,p_->stream,{{const_cast<void**>(&a.data),sizeof(uintptr_t)},{const_cast<void**>(&b.data),sizeof(uintptr_t)},{&out->data,sizeof(uintptr_t)},{&n,8}},p,p?&p->axpy_ms:nullptr,e);
#else
(void)a;(void)b;(void)out;(void)p;(void)e;return false;
#endif
}
}

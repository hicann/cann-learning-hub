#include "gmres.hpp"
#include "npu_compute.hpp"
#if DIS_GMRES_HAS_CANN
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
namespace dis_gmres {
namespace {
using Clock=std::chrono::steady_clock;
std::size_t hi(int r,int c,int restart){return static_cast<std::size_t>(r)*restart+c;}
void rotate(float*x,float*y,float c,float s){float t=c**x+s**y;*y=-s**x+c**y;*x=t;}
void make_rotate(float x,float y,float*c,float*s){if(y==0){*c=1;*s=0;}else if(std::fabs(y)>std::fabs(x)){float t=x/y;*s=1/std::sqrt(1+t*t);*c=t**s;}else{float t=y/x;*c=1/std::sqrt(1+t*t);*s=t**c;}}
struct Arena{NpuCompute&ops;std::vector<DeviceVector*> v;~Arena(){for(auto*x:v)ops.release(x);}void own(DeviceVector*x){v.push_back(x);}};
}
GmresResult distributed_gmres_npu(const CSRMatrix&m,const std::vector<RowPartition>&parts,
 const std::vector<float>&b,std::vector<float>*host_x,HcclCommunicator*comm,const GmresOptions&o,std::string*e){
 GmresResult result;const auto total0=Clock::now();comm->reset_stats();
 if(o.communication_avoiding){if(e)*e="Device CGS multi-dot kernel is not enabled; use --orthogonalization mgs";return result;}
 NpuCompute ops(comm->stream());if(!ops.prepare(m,e))return result;const int restart=std::max(1,o.restart);
 DeviceVector db,dx,residual,ax,scalar_local,scalar_global,global_send,global_x;std::vector<DeviceVector>basis(static_cast<std::size_t>(restart+1));const bool single_rank=comm->world_size()==1;
  Arena arena{ops,{}};for(auto*v:{&db,&dx,&residual,&ax,&scalar_local})arena.own(v);if(!single_rank)for(auto*v:{&scalar_global,&global_send,&global_x})arena.own(v);for(auto&v:basis){arena.own(&v);if(!ops.allocate(b.size(),&v,e))return result;}
 const auto upload0=Clock::now();if(!ops.upload(b,&db,e)||!ops.upload(*host_x,&dx,e)||!ops.allocate(b.size(),&residual,e)||!ops.allocate(b.size(),&ax,e)||!ops.allocate(1,&scalar_local,e))return result;if(!single_rank&&(!ops.allocate(1,&scalar_global,e)||!ops.allocate(m.cols,&global_send,e)||!ops.allocate(m.cols,&global_x,e)))return result;result.profile.transfer_ms+=std::chrono::duration<double,std::milli>(Clock::now()-upload0).count();
 const auto&own=parts[static_cast<std::size_t>(comm->rank())];std::vector<float>scalar_host(1);
 auto gather_spmv=[&](const DeviceVector&local,DeviceVector*out){if(single_rank)return ops.spmv(local,out,&result.profile,e);if(aclrtMemset(global_send.data,global_send.size*4,0,global_send.size*4)!=ACL_SUCCESS){if(e)*e="Device global vector memset failed";return false;}auto*dst=static_cast<char*>(global_send.data)+static_cast<std::size_t>(own.first)*4;if(aclrtMemcpy(dst,local.size*4,local.data,local.size*4,ACL_MEMCPY_DEVICE_TO_DEVICE)!=ACL_SUCCESS){if(e)*e="Device local vector placement failed";return false;}if(!comm->allreduce_device(global_send.data,global_x.data,global_x.size,e))return false;return ops.spmv(global_x,out,&result.profile,e);};
 auto norm=[&](const DeviceVector&v,float*out){const double before=result.profile.dot_ms;if(!ops.dot(v,v,&scalar_local,&result.profile,e))return false;const double cost=result.profile.dot_ms-before;result.profile.dot_ms-=cost;result.profile.norm_ms+=cost;if(single_rank){const auto d0=Clock::now();if(!ops.download(scalar_local,&scalar_host,e))return false;result.profile.transfer_ms+=std::chrono::duration<double,std::milli>(Clock::now()-d0).count();*out=std::sqrt(std::max(0.0f,scalar_host[0]));return true;}if(!comm->allreduce_device(scalar_local.data,scalar_global.data,1,e))return false;const auto d0=Clock::now();if(!ops.download(scalar_global,&scalar_host,e))return false;result.profile.transfer_ms+=std::chrono::duration<double,std::milli>(Clock::now()-d0).count();*out=std::sqrt(std::max(0.0f,scalar_host[0]));return true;};
 auto global_dot=[&](const DeviceVector&a,const DeviceVector&bb,float*out){if(!ops.dot(a,bb,&scalar_local,&result.profile,e))return false;if(single_rank){const auto d0=Clock::now();if(!ops.download(scalar_local,&scalar_host,e))return false;result.profile.transfer_ms+=std::chrono::duration<double,std::milli>(Clock::now()-d0).count();*out=scalar_host[0];return true;}if(!comm->allreduce_device(scalar_local.data,scalar_global.data,1,e))return false;const auto d0=Clock::now();if(!ops.download(scalar_global,&scalar_host,e))return false;result.profile.transfer_ms+=std::chrono::duration<double,std::milli>(Clock::now()-d0).count();*out=scalar_host[0];return true;};
  float bn=0;if(!norm(db,&bn))return result;bn=std::max(bn,1e-30f);
  std::vector<float>h(static_cast<std::size_t>(restart+1)*restart),cs(restart),sn(restart),g(restart+1);
  // True relative residual of the starting guess, computed once before the
  // outer restart loop; beta stays persistent across restart cycles.
  float beta=0;if(!gather_spmv(dx,&ax)||!ops.subtract(db,ax,&residual,&result.profile,e))return result;if(!norm(residual,&beta))return result;result.residual=beta/bn;if(!std::isfinite(result.residual)){if(e)*e="non-finite relative residual";return result;}if(result.residual<=o.tolerance){result.converged=true;}
  while(result.iterations<o.max_iterations&&!result.converged){if(!ops.copy(residual,&basis[0],e)||!ops.scale(1/beta,&basis[0],&result.profile,e))return result;std::fill(h.begin(),h.end(),0);std::fill(g.begin(),g.end(),0);g[0]=beta;int used=0;
  for(int j=0;j<restart&&result.iterations<o.max_iterations;++j){auto&w=basis[static_cast<std::size_t>(j+1)];if(!gather_spmv(basis[static_cast<std::size_t>(j)],&w))return result;for(int i=0;i<=j;++i){float coefficient=0;if(!global_dot(w,basis[static_cast<std::size_t>(i)],&coefficient))return result;h[hi(i,j,restart)]=coefficient;if(!ops.axpy(-coefficient,basis[static_cast<std::size_t>(i)],&w,&result.profile,e))return result;}float next=0;if(!norm(w,&next))return result;h[hi(j+1,j,restart)]=next;if(next>1e-30f&&!ops.scale(1/next,&w,&result.profile,e))return result;auto q0=Clock::now();for(int i=0;i<j;++i)rotate(&h[hi(i,j,restart)],&h[hi(i+1,j,restart)],cs[i],sn[i]);make_rotate(h[hi(j,j,restart)],h[hi(j+1,j,restart)],&cs[j],&sn[j]);rotate(&h[hi(j,j,restart)],&h[hi(j+1,j,restart)],cs[j],sn[j]);rotate(&g[j],&g[j+1],cs[j],sn[j]);result.profile.givens_ms+=std::chrono::duration<double,std::milli>(Clock::now()-q0).count();++result.iterations;used=j+1;result.residual=std::fabs(g[j+1])/bn;if(result.residual<=o.tolerance||next<=1e-30f)break;}
  std::vector<float>y(static_cast<std::size_t>(used));for(int i=used-1;i>=0;--i){float z=g[i];for(int j=i+1;j<used;++j)z-=h[hi(i,j,restart)]*y[j];y[i]=std::fabs(h[hi(i,i,restart)])>1e-30f?z/h[hi(i,i,restart)]:0;}for(int i=0;i<used;++i)if(!ops.axpy(y[i],basis[static_cast<std::size_t>(i)],&dx,&result.profile,e))return result;
  // Mirror the CPU path: after every dx update, re-evaluate the true relative
  // residual on Device (CSR/Krylov vectors stay on Device, only the scalar
  // norm returns to Host), check finiteness and set converged, so convergence
  // is still reported when this cycle consumes exactly max_iterations. The
  // recomputed beta naturally carries into the next restart cycle.
  if(!gather_spmv(dx,&ax)||!ops.subtract(db,ax,&residual,&result.profile,e))return result;
  if(!norm(residual,&beta))return result;
  result.residual=beta/bn;
  if(!std::isfinite(result.residual)){if(e)*e="non-finite relative residual after solution update";return result;}
  if(result.residual<=o.tolerance){result.converged=true;}
 }
 const auto download0=Clock::now();if(!ops.download(dx,host_x,e))return result;result.profile.transfer_ms+=std::chrono::duration<double,std::milli>(Clock::now()-download0).count();result.profile.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-total0).count();const auto&s=comm->stats();result.profile.communication_ms=s.collective_ms;result.profile.transfer_ms+=s.transfer_ms;result.profile.synchronization_ms+=s.synchronization_ms;result.profile.allreduce_calls=s.allreduce_calls;result.profile.allgather_calls=s.allgather_calls;return result;
}
}
#endif

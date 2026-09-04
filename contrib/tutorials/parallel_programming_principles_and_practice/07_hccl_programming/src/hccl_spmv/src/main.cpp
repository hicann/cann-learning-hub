#include "hccl_context.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
namespace fs=std::filesystem;using namespace hccl_spmv;
struct Options{std::string matrix="U1",matrix_dir="matrices",rank_table;int warmup=2,repeat=10,device=-1,rank=-1,world=-1;};
static int env_i(const char*n,int d){const char*v=std::getenv(n);return v?std::atoi(v):d;}
static Options parse(int ac,char**av){Options o;o.device=env_i("DEVICE_ID",0);o.rank=env_i("RANK_ID",0);o.world=env_i("RANK_SIZE",1);for(int i=1;i<ac;++i){std::string a=av[i];auto need=[&](){if(i+1>=ac)throw std::runtime_error(a+" needs a value");return std::string(av[++i]);};if(a=="--matrix")o.matrix=need();else if(a=="--matrix-dir")o.matrix_dir=need();else if(a=="--rank-table")o.rank_table=need();else if(a=="--device")o.device=std::stoi(need());else if(a=="--rank")o.rank=std::stoi(need());else if(a=="--world-size")o.world=std::stoi(need());else if(a=="--warmup")o.warmup=std::stoi(need());else if(a=="--repeat")o.repeat=std::stoi(need());else if(a=="--help"){std::cout<<"--matrix U1|U2|L1|L2|B1|B2 --rank-table FILE --world-size N --rank R --device D --warmup N --repeat N\n";std::exit(0);}else throw std::runtime_error("unknown option "+a);}if(o.world<1||o.rank<0||o.rank>=o.world||o.repeat<1)throw std::runtime_error("invalid rank or iteration count");return o;}
int main(int ac,char**av){
 try{
  const Options o=parse(ac,av);fs::create_directories(o.matrix_dir);CSRMatrix matrix;std::string error;const auto path=(fs::path(o.matrix_dir)/(o.matrix+".csrbin")).string();
  if(!CSRMatrix::load(path,&matrix,&error)){if(o.rank==0){matrix=generate_matrix(o.matrix);if(!matrix.save(path,&error))throw std::runtime_error(error);}else{bool loaded=false;for(int attempt=0;attempt<600&&!loaded;++attempt){loaded=CSRMatrix::load(path,&matrix,&error);if(!loaded&&attempt<599)std::this_thread::sleep_for(std::chrono::milliseconds(100));}if(!loaded)throw std::runtime_error("matrix cache is not ready: "+path);}}error.clear();
  const auto x=generate_x(matrix.cols);std::vector<float> reference;spmv_cpu(matrix,x,&reference);
  HcclContext context;if(!context.initialize(o.device,o.rank,o.world,o.rank_table,&error)||!context.real_hccl())throw std::runtime_error(error.empty()?"real ACL+HCCL backend is required":error);
  double total=0,communication=0,transfer=0,launch_overhead=0,synchronization=0,launch_to_complete=0;std::vector<float> output(static_cast<size_t>(matrix.rows));
  {
   const int chunk=(matrix.rows+o.world-1)/o.world;const int first=std::min(matrix.rows,o.rank*chunk),last=std::min(matrix.rows,first+chunk);
   DeviceSpmv device_spmv;if(!device_spmv.prepare(context,matrix,first,last,chunk,&error))throw std::runtime_error(error);
   for(int i=0;i<o.warmup;++i){Timings t;if(!distributed_spmv(context,device_spmv,x,&output,&t,&error))throw std::runtime_error(error);}
   for(int i=0;i<o.repeat;++i){Timings t;if(!distributed_spmv(context,device_spmv,x,&output,&t,&error))throw std::runtime_error(error);total+=t.total_ms;communication+=t.communication_ms;transfer+=t.transfer_ms;launch_overhead+=t.kernel_launch_overhead_ms;synchronization+=t.synchronization_ms;launch_to_complete+=t.local_spmv_launch_to_complete_ms;}
  }
  const double scale=1.0/o.repeat,error_value=max_relative_error(reference,output);
  const double local_total=total*scale,local_comm=communication*scale,local_transfer=transfer*scale;
  const double local_launch=launch_overhead*scale,local_sync=synchronization*scale,local_ltc=launch_to_complete*scale;
  std::cout<<"Rank="<<o.rank<<" Device ID="<<o.device<<" World Size="<<o.world<<"\n";
  // Per-rank local summary: every rank reports the timings it observed itself.
  std::cout<<"Local Total Time="<<std::fixed<<std::setprecision(6)<<local_total<<" ms\n"
           <<"Local Kernel launch overhead="<<local_launch<<" ms\n"
           <<"Local SpMV launch-to-complete="<<local_ltc<<" ms\n"
           <<"Local HCCL Communication="<<local_comm<<" ms\n"
           <<"Local Data Transfer="<<local_transfer<<" ms\n"
           <<"Local Synchronization="<<local_sync<<" ms\n";
  // Wall-time aggregation: MAX across ranks gives the distributed critical-path
  // time; a SUM/world_size rank mean would understate the true wall time.
  const std::vector<float> local={static_cast<float>(local_total),static_cast<float>(local_launch),static_cast<float>(local_ltc),static_cast<float>(local_comm),static_cast<float>(local_transfer),static_cast<float>(local_sync)};
  std::vector<float> global;
  if(!context.allreduce_max(local,&global,&error))throw std::runtime_error(error);
  if(o.rank==0)std::cout<<"Matrix="<<o.matrix<<" rows="<<matrix.rows<<" cols="<<matrix.cols<<" nnz="<<matrix.nnz()<<"\nActual Compute Backend=Ascend C FP32 RTC\nCommunication Backend=ACL+HCCL\nTotal Time (MAX across ranks)="<<std::fixed<<std::setprecision(6)<<global[0]<<" ms\nKernel launch overhead (MAX)="<<global[1]<<" ms\nLocal SpMV launch-to-complete (MAX)="<<global[2]<<" ms\nHCCL Communication (MAX)="<<global[3]<<" ms\nData Transfer (MAX)="<<global[4]<<" ms\nSynchronization (MAX)="<<global[5]<<" ms\nCPU Reference Error="<<std::scientific<<std::setprecision(9)<<error_value<<"\nCorrectness="<<(error_value<1e-6?"PASS":"FAIL")<<"\n";
  context.finalize();return error_value<1e-6?0:1;
 }catch(const std::exception&e){std::cerr<<"hccl_spmv: "<<e.what()<<"\n";return 2;}
}

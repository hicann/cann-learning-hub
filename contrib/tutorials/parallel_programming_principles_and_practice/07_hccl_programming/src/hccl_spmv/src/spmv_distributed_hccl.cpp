#include "hccl_context.hpp"
#include <algorithm>
#include <chrono>
namespace hccl_spmv {
bool distributed_spmv(HcclContext& ctx,DeviceSpmv& device_spmv,const std::vector<float>& input,
                      std::vector<float>* output,Timings* timings,std::string* error){
 if(!output||input.empty()||output->empty()){if(error)*error="input/output size mismatch";return false;}
  const std::size_t rows=output->size();const auto total_begin=std::chrono::steady_clock::now();void* device_x=nullptr;
  if(!ctx.broadcast_device(input,&device_x,error))return false;
  void* device_local_y=nullptr;double launch_overhead_ms=0.0,sync_ms=0.0;
  if(!device_spmv.run(device_x,&device_local_y,&launch_overhead_ms,&sync_ms,error))return false;
  const int chunk=(static_cast<int>(rows)+ctx.world()-1)/ctx.world();
  std::vector<float> gathered;if(!ctx.allgather_device(device_local_y,static_cast<std::size_t>(chunk),&gathered,error))return false;
  output->assign(gathered.begin(),gathered.begin()+std::min(rows,gathered.size()));
  if(timings){timings->communication_ms=ctx.last_collective_ms();timings->transfer_ms=ctx.last_transfer_ms();timings->kernel_launch_overhead_ms=launch_overhead_ms;timings->local_spmv_launch_to_complete_ms=launch_overhead_ms+sync_ms;timings->synchronization_ms=sync_ms;timings->total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-total_begin).count();}
 return true;
}
}

#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"
#include "kernel_operator_dump_tensor_intf_impl.h"
constexpr int32_t BUFFER_NUM = 2;

class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init() 
    {
     
    }
    __aicore__ inline void Process()
    {
  
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
   
    }
    __aicore__ inline void Compute(int32_t progress)
    {
      
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
    
    }

};



 __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingDataTemplate);
    GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, tiling_data, tiling);

}

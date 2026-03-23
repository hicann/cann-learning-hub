# AICPU Tiling下沉编程

## 基础知识准备

本文内容基于Ascend C算子开发衍生而来，对于算子开发还不了解的读者可以通过以下资源进行学习：

[《Ascend C算子开发文档手册》](https://www.hiascend.com/document/redirect/CannCommunityOpdevAscendC)

[《Ascend C算子开发系列课程》](https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development)



## 背景介绍

**Host Bound**一直是算子调用的显著性能瓶颈，造成**Host Bound**的核心原因就在于算子在Kernel执行前都需要计算出TilingData，而TilingData的计算通常是在Host侧完成再拷贝到Device侧的。针对这一问题我们推出了**AICPU Tiling下沉编程方式**，使用Device侧的AICPU计算TilingData，节省了Host侧拷贝TilingData到Device侧的步骤，降低算子执行耗时。

## 亮点介绍

- 通过减少Host与Device的交互，**提升算子执行性能**
- 通过<<<>>>调用AICPU的方式，**降低了编程成本**


## AICPU Tiling下沉编程使用详解

### 一、开发流程

1. **目录结构**

   以一个简单的**abs**算子的demo为例：

   ```
   📁 aicpu_demo/             # demo目录
   ├── 📄 main.cpp            # 算子入口，分别调用AICPU和AICORE
   ├── 📄 abs.aicpu           # 算子AICPU实现
   ├── 📄 abs.asc             # 算子AICORE实现
   ├── 📄 kernel_args.h       # 结构体定义
   ├── 📄 CMakeLists.txt      # cmake文件
   ```

2. **编写AICPU Tiling实现逻辑**

   - 定义AICPU Tiling的KernelArgs入参（对应**kernel_args.h**文件）  

     当前<<<>>>方式调用AICPU函数可以通过传入一个结构体指针的方式进行调用，如下，将算子需要的用于计算Tiling的入参和输出的TilingData地址定义在一个struct中。

     ```cpp
      
      // kernel_args.h
      struct TilingInfo {
        uint32_t data_size_per_block;
      };
      
      struct KernelArgs {
          uint32_t block_num;
          uint32_t data_size;
          TilingInfo *ti; // 与aicore共享的参数
      };
      
     ```

   - AICPU Tiling的实现

     将上一步定义的KernelArgs作为入参，实现AICPU Tiling的逻辑，将计算好的结果写入TilingData中。

     ```cpp
     // abs.aicpu
     __global__ __aicpu__ int32_t TemplateAicpuKernel(T *args)
     {
         // 计算每个核需要处理的数据量，将结果保存在tiling地址对应的device空间中
         args->ti->data_size_per_block = args->data_size / args->block_num;
         return 0;
     }
     ```

3. **编写AICORE实现逻辑**

   实现一个简单的abs算子示例：只使用一个核计算所有输入的abs结果，通过tiling地址来访问计算好的tiling数据。

     ```cpp
   // abs.asc
   template<typename T>
   __aicore__ void hello_world_impl(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR tiling_addr)
   {
       __gm__ struct TilingInfo *tiling = (__gm__ struct TilingInfo *)tiling_addr;
       uint64_t dataSize =  tiling->data_size_per_block;
   
       AscendC::printf("aicore get dataSize %d\n", dataSize);
   
       AscendC::GlobalTensor<float> inputGlobal; 
       AscendC::GlobalTensor<float> outputGlobal; 
       inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src_gm), dataSize);
       outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst_gm), dataSize);
       AscendC::TPipe pipe;
   
       AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
       pipe.InitBuffer(calcBuf, dataSize * sizeof(float));
       AscendC::LocalTensor<float> tempTensor1 = calcBuf.Get<float>();
   
       AscendC::DataCopy(tempTensor1, inputGlobal, dataSize);
   
       event_t eventID1 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
       AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID1);
       AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID1);
   
       AscendC::Abs(tempTensor1, tempTensor1, dataSize);
       
       event_t eventIdVToMte3 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
       AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
       AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
   
       AscendC::DataCopy(outputGlobal, tempTensor1, dataSize);
   }
   
     ```


4. **通过两条不同的流分别调用AICPU和AICORE任务**

   出于性能的考虑，需要使用不同的两条流来分别执行AICPU和AICORE任务，目的是在网络场景中让AICPU和AICORE的计算能够并行；同时对于单算子内部的实现，需要使用event机制，来保证AICPU的计算结束后再执行AICORE上的任务。

   ```cpp
   // main.cpp
   int32_t main(void) {
       CHECK_ACL(aclInit(nullptr));
       int32_t deviceId = 0;
       printf("acl init ok! \n");
       CHECK_ACL(aclrtSetDevice(deviceId));
       printf("set device ok! \n");
   
       aclrtStream aicpu_stream = nullptr;
       aclrtStream aicore_stream = nullptr;
       CHECK_ACL(aclrtCreateStream(&aicpu_stream));
       CHECK_ACL(aclrtCreateStream(&aicore_stream));
       printf("create stream ok! \n");
   
       aclrtEvent event;
       CHECK_ACL(aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC));
   
       void *srcDevice;
       void *dstDevice;
       void *ti;
       CHECK_ACL(aclrtMalloc((void **)&srcDevice, 4096, ACL_MEM_MALLOC_HUGE_FIRST));
       CHECK_ACL(aclrtMalloc((void **)&dstDevice, 4096, ACL_MEM_MALLOC_HUGE_FIRST));
       CHECK_ACL(aclrtMalloc((void **)&ti, 4096, ACL_MEM_MALLOC_HUGE_FIRST));
       void *zHost = malloc(4096);
       memset(zHost, 0, 4096);
       CHECK_ACL(aclrtMemcpy(srcDevice, 4096, zHost, 4096, ACL_MEMCPY_HOST_TO_DEVICE));
       CHECK_ACL(aclrtMemcpy(dstDevice, 4096, zHost, 4096, ACL_MEMCPY_HOST_TO_DEVICE));
       struct KernelArgs args = {0};
       args.block_num = 1;
       args.data_size = 10;
       args.ti = (TilingInfo *)ti;
   
       TemplateAicpuKernel_do(aicpu_stream, &args);
   
       CHECK_ACL(aclrtRecordEvent(event, aicpu_stream));
       CHECK_ACL(aclrtStreamWaitEvent(aicore_stream, event));
   
       hello_world_do(1, aicore_stream, (uint8_t *)srcDevice, (uint8_t *)dstDevice, (uint8_t *)ti);
       printf("launch ok! \n");
   
       CHECK_ACL(aclrtSynchronizeStreamWithTimeout(aicore_stream, 10000));
       printf("sync ok!\n");
   
       CHECK_ACL(aclrtFree(srcDevice));
       CHECK_ACL(aclrtFree(dstDevice));
       free(zHost);
       CHECK_ACL(aclrtDestroyStream(aicpu_stream));
       CHECK_ACL(aclrtDestroyStream(aicore_stream));
       CHECK_ACL(aclrtResetDevice(deviceId));
       CHECK_ACL(aclFinalize());
       return 0;
   }
   ```

   - AICPU Tiling入口

   ```cpp
   // abs.asc
   template<typename T>
   extern __global__ __aicpu__ int32_t TemplateAicpuKernel(T *args);
   
   template extern __global__ __aicpu__ int32_t TemplateAicpuKernel<KernelArgs>(KernelArgs *args);
   
   void TemplateAicpuKernel_do(void *stream, KernelArgs *args) // aicpu entrance
   {
       TemplateAicpuKernel<KernelArgs><<<1, nullptr, stream>>>(args, sizeof(KernelArgs));
   }
   ```

   - AICORE入口

   ```cpp
   // abs.asc
   template<typename T>
   __global__ __aicore__ void hello_world(GM_ADDR src, GM_ADDR dst, GM_ADDR tiling)
   {
       hello_world_impl<T>(src, dst, tiling);
   }
   extern "C" {
       void hello_world_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *dst, uint8_t *ti)  // aicore entrance
       {
           hello_world<int><<<1, nullptr, stream>>>(src, dst, ti);
       }
   }
   ```

5. **CMake编译**

   在`CMakeLists.txt`文件中分别使用不同的编译配置编译AICORE和AICPU，最终将结果打包成一个静态库。

   ```cpp
   // CMakeLists.txt
   cmake_minimum_required(VERSION 3.18)
   set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
   set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
   set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
   set(CMAKE_PREFIX_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
   set(ASCEND_LIB_DIR "$ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64")
   link_directories(${ASCEND_LIB_DIR})
   find_package(ASC REQUIRED)
   find_package(AICPU REQUIRED)
   add_library(my_kernel SHARED
       abs.aicpu
       abs.asc
   )
   set_target_properties(my_kernel PROPERTIES LINKER_LANGUAGE CXX)
   project(my_ops LANGUAGES ASC AICPU CXX)
   target_link_libraries(my_kernel PRIVATE
       ascendc_runtime
       profapi
       ascendalog
       ascendcl 
       runtime 
       c_sec 
       mmpa
       error_manager
       ascend_dump
       pthread
   )
   target_compile_options(my_kernel PRIVATE
       $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>
   )
   target_include_directories(my_kernel PUBLIC
       $ENV{ASCEND_HOME_PATH}/lib64
       $ENV{ASCEND_HOME_PATH}/x86_64-linux/include
       $ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64
       ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/aicpu_api
   )
   
   add_executable(main main.cpp)
   
   target_link_libraries(main PRIVATE
       my_kernel
       ascendcl
   )
   ```

##### 二. 代码调测

- Host侧
  可使用通用C++语言的维测手段，包括打印，GDB等。

- Device侧

  - AICORE 可直接使用`AscendC::printf`或`AscendC::DumpTensor`，打印变量调试。

  - AICPU 也可使用`AscendC::printf`打印变量调试。

    ```cpp
    __global__ __aicpu__ int32_t TemplateAicpuKernel(T *args)
    {
        int32_t var = 0;
        AscendC::printf("TemplateAicpuKernel inited! %d\n", var);
        ...
    }
    ```


##### 三. 性能调优

- 该方案中由于把Tiling计算移动到了AICPU上，因此Tilingkey无法在Host上获取，只能将原本的Tilingkey分发逻辑移动到AICORE Kernel中进行判断；在实际开发dequant_swiglu_quant算子时，初步性能测试时发现这一改动导致了额外的icache miss，算子整体性能下降5%。

```cpp
__global__ __aicore__ __attribute__((aiv)) void dequant_swiglu_quant(GM_ADDR x, GM_ADDR weight_scale, GM_ADDR activation_scale, GM_ADDR bias, GM_ADDR quant_scale, GM_ADDR quant_offset, GM_ADDR y, GM_ADDR scale, GM_ADDR tiling_data) {
      __gm__ struct DequantSwigluQuantTiling *tiling = (__gm__ struct DequantSwigluQuantTiling *)tiling_data;
    
      if (AscendC::GetBlockIdx() >= tiling->core_num) {
        return;
      }
    
      // 原本是在Host上进行判断
      if (tiling->tiling_key == 0) {
        swiglu_quant_impl<float16_t, 2>(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, y, scale, tiling_data);
      } else if (tiling->tiling_key == 1) {
        swiglu_quant_impl<bfloat16_t, 2>(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, y, scale, tiling_data);
      } else if (tiling->tiling_key == 2) {
        swiglu_quant_impl<float16_t, 1>(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, y, scale, tiling_data);
      } else if (tiling->tiling_key == 3) {
        swiglu_quant_impl<bfloat16_t, 1>(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, y, scale, tiling_data);
      } else if (tiling->tiling_key == 4) {
        dequant_swiglu_quant_impl<2>(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, y, scale, tiling_data);
      } else if (tiling->tiling_key == 5) {
        dequant_swiglu_quant_impl<1>(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, y, scale, tiling_data);
      }
  }
```

- 针对这一现象，可以使用Ascend C提供的ICachePreLoad接口将代码段预加载到ICache中，使得该算子整体性能相较于原本提升了15%。

 ```cpp
 template <int BufferNum>
 __aicore__  void dequant_swiglu_quant_impl(GM_ADDR x, GM_ADDR weight_scale, GM_ADDR activation_scale, GM_ADDR bias,
                            GM_ADDR quant_scale, GM_ADDR quant_offset, GM_ADDR y, GM_ADDR scale,
                            GM_ADDR tiling_data) {
  AscendC::ICachePreLoad(2); // 按照实际代码段长度根据接口文档来设置参数
  AscendC::TPipe pipe;
  DequantSwigluQuantKernel<BufferNum> op(&pipe);
  op.init(x, weight_scale, activation_scale, bias, quant_scale, quant_offset, y, scale, tiling_data);
  op.process();
 }
 ```

  - AICPU Tiling+ICachePreLoad耗时：

| 数据类型 | 大case(us) | 小case(us) |
| -------- | ---------- | ---------- |
| FP16     | 60.8       | 9.6        |
| BF16     | 62.38      | 9.2        |
| INT32    | 85         | 11.36      |

  - 原版耗时：

| 数据类型 | 大case(us) | 小case(us) |
| -------- | ---------- | ---------- |
| FP16     | 69.08      | 9.46       |
| BF16     | 69.48      | 8.56       |
| INT32    | 105        | 12.96      |




## 总结

AICPU Tiling下沉方案优化了算子在Host侧上动态计算Tiling场景的性能，同时通过<<<>>>的方式调用AICPU让开发者能轻松地完成方案的代码适配。此方案正在逐步应用到实际的商用业务场景中，成为解决算子**Host-Bound**问题的有效路径之一。


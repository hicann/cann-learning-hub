# HIXL FabricMem高性能KV Cache传输

## 背景

随着大语言模型（LLM）参数规模的指数级增长，推理部署面临着前所未有的内存压力。以 GPT-4 级别的模型为例，千亿级参数在 FP16 精度下仅权重就需要数百 GB 显存，而更大的内存消耗来自 KV Cache——在长序列推理场景下，KV Cache 的容量需求往往超过模型权重本身数倍。业界对此的解决方案是构建多级缓存架构，将 GPU 显存作为一级缓存，分布式 DRAM 作为二级缓存，SSD/NVMe作为三级缓存。以 Mooncake 为代表的分布式 KV Cache 系统逐渐成为主流选择，HIXL作为一个传输后端，也已经接入了Mooncake，如何高效地在超节点之间进行KV Cache传输成为核心挑战。在以DRAM构成的分布式内存池中，传统方案依赖RoCE（RDMA over Converged Ethernet）网络，其满载带宽约 20GB/s，在Atlas 800T A3 超节点的部署场景下会成为明显的性能瓶颈。为此，HIXL提供了 Fabric Mem模式，可以将Atlas 800T A3 超节点内的传输带宽提升至百 GB/s 级别。

## 整体方案

在Atlas 800T A3 超节点内，所有计算节点的 DRAM 内存被统一编址，NPU 可以通过 HCCS高速链路直接访问远程节点的内存。Fabric Mem 模式的核心价值在于：

- 超节点内 DRAM 统一编址：打破节点边界，实现内存资源池化

- D2RH/RH2D 高带宽传输：设备到远程主机、远程主机到设备的双向高速通道

- 无需 CPU 介入的单边通信：源端主动发起传输，对端零开销

## 基于 VMM 的内存管理

Fabric Mem 模式的底层依赖于CANN的Virtual Memory Manager机制, 实现了全局统一编址并支持各个进程直接访问，具体实现如下：

1. 每个进程申请自己的片上内存和DRAM内存: 先通过调用aclrtMallocPhysical申请物理内存，再通过调用aclrtReserveMemAddress申请虚拟内存，最后通过调用aclrtMapMem将物理内存映射到虚拟内存。

2. 进行物理地址的交换。

3. 将物理地址映射到访问进程的页表中。

4. 发起SDMA访问，可读写任何进程的片上内存和DRAM内存。

![fabricmem_unified_virtual_addressing](images/fabricmem_unified_virtual_addressing.jpeg)

从本地NPU的片上内存直接往远程的HOST内存写数据的数据流向：

![fabricmem_cross_machine_transfer_path](images/fabricmem_cross_machine_transfer_path.png)

另外HIXL Fabric Mem 模式已与 Mooncake Store 深度集成，在Mooncake系统中，HOST内存直接由Mooncake管理，且做了numa均衡的申请，这种复杂性用户无需感知。使用的VMM接口列表:

| 接口 | 说明 |
| --- | --- |
| aclrtReserveMemAddress | 预留虚拟内存 |
| aclrtMallocPhysical | 申请物理内存 |
| aclrtMapMem | 将虚拟内存映射到物理内存 |
| aclrtMemExportToShareableHandle | 导出共享物理内存Handle |
| aclrtMemImportFromShareableHandle | 导入共享物理内存Handle |
| aclrtMemRetainAllocationHandle | 获取物理内存Handle |

## 性能

| 传输方向 | 数据量 | Fabric Mem 带宽 | RoCE带宽 |
| --- | --- | --- | --- |
| put D2RH (Device to Remote Host) | 1GB | 64 GB/s | 20 GB/s |
| get RH2D (Remote Host to Device) | 1GB | 103 GB/s | 20 GB/s |

注：FabricMem使用HCCS高速链路，RoCE使用参数面网络。另外，在vLLM-Ascend Prefix Cache场景实测数据：50%命中率时，TTFT降低40%+，100%命中率时，TTFT降低85%。

## 使用示例

在Mooncake中，运行时使用export ASCEND_ENABLE_USE_FABRIC_MEM=1启用Fabric Mem模式。
使用Mooncake的示例请参考:
https://gitcode.com/cann/hixl/tree/master/examples/third_parties/mooncake_store/python

直接使用HIXL接口的示例请参考:
https://gitcode.com/cann/hixl/blob/master/examples/cpp/fabric_mem_d2d.cpp

## 总结

对于使用Mooncake在昇腾超节点构建KV Cache池化系统的开发者，Fabric Mem 模式提供了一种性能提升方案：无需修改业务逻辑，仅需几行配置即可使用HCCS高速链路。HIXL Fabric Mem 模式的推出，标志着昇腾在KVCache池化技术方面的进一步探索。未来发展方向包括：

1. 下一代超节点支持：支持下一代超节点内的多种传输模式

2. 异构互联优化：支持 A2/A3 系列芯片的混合部署场景, 支持通用服务器异构部署场景

3. 开源社区持续集成：和Mooncake、vLLM、SGLang等社区持续集成

欢迎广大开发者来了解并共同建设。

## 相关链接

HIXL仓库：
https://gitcode.com/cann/hixl

Mooncake仓库：
https://github.com/kvcache-ai/Mooncake

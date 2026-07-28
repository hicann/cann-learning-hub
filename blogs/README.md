## 介绍
这里将介绍CANN在实际业务场景中基于最新技术特性以及实践成果形成的文章博客，帮助大家了解和掌握CANN最新的行业技术动态。


## 算子
| 案例名称 | 案例介绍 | 发布时间 |
| --- | --- | --- |
| [多Vendor自定义算子并行编译实践](./operator/multi_vendor_operator_parallel_compilation/multi_vendor_operator_parallel_compilation.md) | 通过顶层构建入口统一调度多个独立 Vendor 子工程，并行完成自定义算子编译、打包和产物隔离，降低多算子包的维护与交付成本。 | 2026.7 |
| [TileLang与xLLM驱动Qwen3.5昇腾算子适配](./operator/tilelang_xllm_qwen35_operator_adaptation/tilelang_xllm_qwen35_operator_adaptation.md) | 以 TileLang Python DSL 为算子开发入口，结合 xLLM 完成 Qwen3.5 昇腾算子适配，形成可复用的模型算子接入流程。 | 2026.7 |
| [昇腾950 AIV直驱URMA的SHMEM跨PE通信实践](./operator/ascend950_aiv_urma_shmem_communication/ascend950_aiv_urma_shmem_communication.md) | 介绍 AIV 在 Device 侧构造 WQE，并借助 URMA 与 SHMEM 完成跨 PE 远端写和通知，减少 Host 往返与阶段同步开销。 | 2026.7 |
| [DSA算子开源贡献实践](./operator/dsa_operator_open_source_contribution/dsa_operator_open_source_contribution.md) | 围绕 DSA 长序列训练算子的 FP32 支持，展示从业务需求、算子定制和调试验证到反向贡献 ops-transformer 社区的完整路径。 | 2026.6 |
| [Scalar对昇腾NPU算子性能的影响与优化](./operator/scalar_npu_operator_performance_optimization/scalar_npu_operator_performance_optimization.md) | 从寄存器 Spill、I-Cache、指针解引用和常量传播等角度分析 ScalarBound，并给出可复用的诊断与编码优化方法。 | 2026.5 |
| [Ascend 950 RmsNormQuant算子分阶段优化实践](./operator/ascend950_rmsnormquant_optimization/ascend950_rmsnormquant_optimization.md) | 通过 Gamma 预加载、多核并行、寄存器数据流、Double Buffer、UB 批处理和二分累加，系统展示 Vector 算子的性能与精度优化过程。 | 2026.5 |
| [MX量化矩阵乘性能优化实践](./operator/mx_quantized_matmul_optimization/mx_quantized_matmul_optimization.md) | 针对 MX 量化矩阵乘的数据搬运和 Cube 利用率，结合 SWAT、尾轮负载均衡和 UnitFlag 提升流水并行效率。 | 2026.5 |
| [TileLang-Ascend算子性能优化实践](./operator/tilelang_ascend_operator_optimization/tilelang_ascend_operator_optimization.md) | 总结 TileLang-Ascend 算子在流水级数、核间同步、数据切分和调试分析方面的性能优化方法。 | 2026.5 |
| [面向MoE的Dispatch与Combine算子优化](./operator/moe_dispatch_combine_optimization/moe_dispatch_combine_optimization.md) | 通过跨 Rank 通信去重、本地加权合并、AIV 直驱 RDMA 与多流并行，降低 MoE Dispatch/Combine 算子的通信和调度开销。 | 2026.5 |
| [使用DumpTensor定位算子计算结果异常](./operator/dumptensor_operator_debugging/dumptensor_operator_debugging.md) | 利用 DumpTensor 观察 GM、UB、L1 中间数据，沿算子数据流逐步定位输入搬运、计算或回写阶段的结果异常。 | 2026.4 |
| [基于AICPU引擎的HCCL点对点通信算子开发](./operator/hccl_custom_operator_aicpu_p2p/基于AICPU引擎的HCCL点对点通信算子开发.md) | 介绍基于 AICPU+TS 引擎实现 HCCL 自定义 Send/Recv 点对点通信算子，满足 pipeline 并行等灵活通信编排需求。 | 2026.2 |
| [AICPU Tiling下沉编程](./operator/aicpu_tiling_sink/AICPU%20Tiling下沉编程.md) |AICPU Tiling下沉编程 将 Tiling 计算下沉到 AICPU，减少 Host 与 Device 交互及拷贝，降低 Host Bound 并提升算子执行效率。 | 2025.12 |
| [自定义算子开发系列：Ascend C RTC即时编译](./operator/ascendc_rtc_compilation/自定义算子开发系列：Ascend%20C%20RTC即时编译.md) | Ascend C RTC 通过运行时按实际 shape 即时编译算子，兼顾更优执行性能、更快编译速度和更灵活的算子迭代维护。 | 2025.12 |
| [基于昇腾的DeepXTrace推理集群快慢卡在线检测](./operator/deepxtrace_moe_slow_card_detection/基于昇腾的DeepXTrace推理集群快慢卡在线检测.md) | DeepXTrace在昇腾设备面向 MOE 推理集群提供轻量级快慢卡在线诊断能力，支持分钟级精准定位通信 slow 问题，缩短排障时间。 | 2025.12 |
| [HCCL ReduceScatter精度优化](./operator/hccl_reducescatter_high_precision_redevelopment/HCCL%20ReduceScatter精度优化.md) | 基于开源 ReduceScatter 进行精度增强改造，在尽量保持通信性能的同时提升分布式计算结果精度。 | 2025.12 |
| [transformer仓experimental路径MIX算子开发贡献](./operator/transformer_experimental_mix_operator/transformer仓experimental路径MIX算子开发贡献.md) | 以矩阵化方式重构 RoPE 并落地首个开源 MIX 算子，在单算子和整网层面同时获得可观性能收益。 | 2025.12 |
| [CrossEntropyLoss与Zloss融合算子开发](./operator/cross_entropy_zloss_fusion/CrossEntropyLoss与Zloss融合算子开发.md) | CrossEntropyLoss和Zloss融合算子通过损失函数融合消除串行小算子开销，解决训练尾部瓶颈，在 MoE 场景中实现整网端到端 5.2% 效率提升。 | 2025.11 |
| [算子Kernel直调编程](./operator/kernel_direct_call_programming/算子Kernel直调编程.md) | 通过 Kernel 直调、异构混合编程和模板化能力，简化算子编译部署流程，降低开发实现门槛。 | 2025.11 |
| [TilingKey模板化编程](./operator/tilingkey_template_programming/TilingKey模板化编程.md) | 借助 TilingKey 模板化编程统一多场景算子开发与管理，同时减少 icache miss 和 scalar 开销，提升调用性能。 | 2025.11 |

<br>

## 推理
| 案例名称 | 案例介绍 | 发布时间 |
| --- | --- | --- |
| [AOT SuperKernel图执行优化](./inference/aot_superkernel_graph_execution/aot_superkernel_graph_execution.md) | 从 Aclgraph 的 Stream 与 Task 调度出发，介绍 AOT SuperKernel 如何融合任务并减少启动、调度等待和算子间流水开销。 | 2026.7 |
| [LongCat-Flash-Lite昇腾推理优化实践](./inference/longcat_flash_lite_inference_optimization/longcat_flash_lite_inference_optimization.md) | 围绕 N-gram Embedding 与 EAGLE3 投机解码，结合 ComputeNGramIds 算子优化和 SuperKernel 提升 LongCat-Flash-Lite 推理效率。 | 2026.6 |
| [星火大模型昇腾算力集群吞吐优化实践](./inference/spark_model_cluster_throughput_optimization/spark_model_cluster_throughput_optimization.md) | 依托 CANN DSA 系列算子将长上下文 Attention 动态稀疏化，在星火大模型 128K 场景实现最高约 4.5 倍 Decode 吞吐提升。 | 2026.6 |
| [TensorFlow与AutoFuse推荐模型算子融合实践](./inference/tensorflow_autofuse_recommendation_fusion/tensorflow_autofuse_recommendation_fusion.md) | 打通 TensorFlow 到 Ascend IR 与 AutoFuse 后端的自动融合链路，通过 Schedule、Auto Tiling 和代码生成提升推荐模型性能。 | 2026.5 |
| [NPU DeepSeek-V4推理优化实践](./inference/deepseek_v4_npu_inference_optimization/deepseek_v4_npu_inference_optimization.md) | 针对 DeepSeek V4 的稀疏 Attention 与 mHC 结构，结合融合算子、上下文并行、量化和多流并行实现高性能 NPU 推理。 | 2026.4 |
| [AutoFuse与TorchInductor的DeepSeek算子融合实践](./inference/autofuse_torchinductor_deepseek_fusion/autofuse_torchinductor_deepseek_fusion.md) | 扩展 TorchInductor NPU Codegen，将 Loop IR 转换为 Ascend IR 并接入 AutoFuse，为 DeepSeek 模型带来自动算子融合收益。 | 2026.4 |
| [DeepSeek V4昇腾超节点支持](./inference/deepseek_v4_supernode_support/deepseek_v4_supernode_support.md) | 介绍昇腾 950 与 A3 超节点对 DeepSeek V4 推理和续训练的适配，包括融合 Kernel、多流并行、量化与自动融合能力。 | 2026.4 |
| [xLLM大模型推理性能优化](./inference/xllm_inference_performance_optimization/xllm_inference_performance_optimization.md) | 结合 xLLM 的图融合、多流并行、投机推理和动态负载均衡，优化 Qwen、DeepSeek 与 ChatGLM 等模型的昇腾推理性能。 | 2026.4 |
| [SALS长序列推理优化](./inference/sals_long_sequence_inference/sals_long_sequence_inference.md) | 从稀疏 Token 选择、SFAA 计算、离散访存、Preload 流水和 AICPU Tiling 等方面优化长序列推理。 | 2026.4 |
| [HIXL快速适配NIXL昇腾后端](./inference/hixl_nixl_ascend_backend/hixl_nixl_ascend_backend.md) | 基于 NIXL 插件架构和 SouthBound API，将 HIXL 点对点通信能力映射为昇腾后端，简化上层框架接入。 | 2026.3 |
| [Overlap Scheduling吞吐优化](./inference/overlap_scheduling_throughput_optimization/Overlap%20Scheduling吞吐优化.md) | 通过 CPU 调度与 NPU 执行重叠隐藏下发时延，提升设备利用率，在 LongCat-Flash 场景中带来约 70% 的 TPS 提升。 | 2026.3 |
| [第三方框架集成npugraph_ex](./inference/npugraph_ex_third_party_framework_integration/第三方框架集成npugraph_ex.md) | 介绍第三方框架如何接入 npugraph_ex 的图编译与编译缓存能力，进一步降低模型推理冷启动和端到端耗时。 | 2026.2 |
| [HIXL FabricMem高性能KV Cache传输](./inference/hixl_fabricmem_kv_cache_transfer/hixl_fabricmem_kv_cache_transfer.md) | 通过超节点 DRAM 统一编址、VMM 映射和 HCCS/SDMA 单边传输，为 Mooncake 等 KV Cache 池化系统提供高带宽 FabricMem 通道。 | 2026.2 |
| [基于Atlas 900 A3 SuperPoD推理部署Deepseek-R1性能优化实践](./inference/deepseek_r1_superpod_inference_optimization/基于Atlas%20900%20A3%20SuperPoD推理部署Deepseek-R1性能优化实践.md) | 结合 Omni-Infer 与 CANN 全栈协同优化，在满足 TTFT<2s、TPOT<50ms 的前提下实现 608 QPM 高吞吐推理。 | 2025.12 |
| [HIXL、Mooncake与vLLM的KV Cache池化与传输](./inference/hixl_mooncake_vllm_kv_cache_pooling/HIXL、Mooncake与vLLM的KV%20Cache池化与传输.md) | 通过 HIXL、Mooncake和vLLM实现KV Cache 池化和高性能 D2D/H2H 传输提升前缀缓存命中率，降低 TTFT 并减少大集群推理成本。 | 2025.12 |
| [HIXL在RL推理中的长尾时延优化](./inference/hixl_rl_tail_latency_optimization/HIXL在RL推理中的长尾时延优化.md) | 利用 HIXL 支撑 RL 推理阶段的 PD 分离与高效数据传输，缓解长尾拖慢问题并提升千卡集群资源利用率。 | 2025.12 |
| [基于Atlas 900 A3 SuperPoD的LongCat-Flash模型推理性能优化实践](./inference/longcat_flash_superpod_inference_optimization/基于Atlas%20900%20A3%20SuperPoD的LongCat-Flash模型推理性能优化实践.md) | 结合多流并发、控核与 SuperKernel 等优化手段，显著提升 LongCat-Flash 推理效率，并将 TPOT 优化到 10ms。 | 2025.12 |
| [CANN npugraph_ex图模式优化](./inference/npugraph_ex_aclgraph_graph_mode/CANN%20npugraph_ex图模式优化.md) | npugraph_ex基于 aclGraph 图捕获与重放能力降低 Host 下发开销，并提供亲和 NPU 的图优化，帮助推理框架获得更低时延。 | 2025.12 |
| [基于torch_npu的IPC特性介绍](./inference/torch_npu_ipc/基于torch_npu的IPC特性介绍.md) | IPC支持跨进程直接共享设备内存，减少显式拷贝开销，在分布式训练和强化学习场景中提升通信效率并节省显存。 | 2025.12 |
| [TorchAir自定义FX Pass](./inference/torchair_fx_pass_multi_stream/TorchAir自定义FX%20Pass.md) | 用自定义 FX Pass 将多流并行等优化从手动脚本改造成自动图变换，减少重复适配代码并提升开发效率。 | 2025.12 |
| [SGLang、Mooncake与CANN HIXL的PD分离D2D部署](./inference/sglang_mooncake_hixl_pd_separation_d2d/SGLang、Mooncake与CANN%20HIXL的PD分离D2D部署.md) | 打通 SGLang、Mooncake 与 HIXL 的协同链路，加速 PD 分离 D2D 特性落地，提升 KV Cache 传输效率与部署灵活性。 | 2025.11 |
| [SuperKernel技术综述](./inference/superkernel_inference_acceleration/SuperKernel技术综述.md) | 通过将整网重新编译为大算子减少调度与访存开销，在现有优化基础上进一步带来 10% 到 20% 的性能提升。 | 2025.11 |
| [vLLM-Ascend推理优化](./inference/vllm_ascend_inference_optimization/vLLM-Ascend推理优化.md) | vLLM-Ascend 基于 PagedAttention 和昇腾适配优化 KV Cache 管理与推理执行，提升大模型服务吞吐量并降低内存浪费。 | 2025.11 |

<br>

## 训练
| 案例名称 | 案例介绍 | 发布时间 |
| --- | --- | --- |
| [基于昇腾的AReaL全异步RL训练](./training/areal_async_rl_training/基于昇腾的AReaL全异步RL训练.md) | 基于全异步 RL、Single Controller 和解耦式 Agentic RL 架构提升训练效率与可靠性，并完成昇腾平台开箱即用适配。 | 2026.3 |
| [大模型训练故障恢复方案FlashRecovery](./training/flashrecovery_training_fault_recovery/大模型训练故障恢复方案FlashRecovery.md) | FlashRecovery 面向大模型长周期训练降低故障恢复成本，减少检查点 I/O 与回滚重算损失，让训练任务更快恢复到正常执行。 | 2025.12 |
| [基于昇腾的SAM投机解码长序列强化学习训练](./training/sam_speculative_decoding_rl_training/基于昇腾的SAM投机解码长序列强化学习训练.md) | 以无辅助模型的 SAM 投机解码降低 RL 训练 Rollout 延迟，在保证精度无损前提下带来超过 35% 的长尾阶段加速收益。 | 2025.12 |

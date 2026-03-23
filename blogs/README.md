## 介绍
这里将介绍CANN在实际业务场景中基于最新技术特性以及实践成果形成的文章博客，帮助大家了解和掌握CANN最新的行业技术动态。


## 算子
| 案例名称 | 案例介绍 | 发布时间 |
| --- | --- | --- |
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
| [Overlap Scheduling吞吐优化](./inference/overlap_scheduling_throughput_optimization/Overlap%20Scheduling吞吐优化.md) | 通过 CPU 调度与 NPU 执行重叠隐藏下发时延，提升设备利用率，在 LongCat-Flash 场景中带来约 70% 的 TPS 提升。 | 2026.3 |
| [第三方框架集成npugraph_ex](./inference/npugraph_ex_third_party_framework_integration/第三方框架集成npugraph_ex.md) | 介绍第三方框架如何接入 npugraph_ex 的图编译与编译缓存能力，进一步降低模型推理冷启动和端到端耗时。 | 2026.2 |
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

# 原子课程目录规划（directory）

> 本目录规划基于《原子课程设计参考》（`ref/atomic_courses.md`）生成，课程清单与编号（Lx-yy，共 **135 门**）对齐同目录总纲 [README.md](./README.md)（其框架层在参考文件 21 门基础上规范化重组并扩充至 29 门，其余三层与参考文件一致）。
>
> **落位规则**：全部原子课程统一规划到 `cann-learning-hub_1799/tutorials/` 目录下，按技术架构四层建立编号子目录；每个子方向（子课程）建立单独目录，目录内**每节课对应一个独立文件**（45–90 分钟/节，Notebook 为主）。
>
> **责任人列**：当前统一为"待定"，待各方向课程负责人确认后填写。

---

## 一、规划原则

| 项目 | 规则 |
|------|------|
| 一级目录 | 技术架构四层：`01_applications`（应用层）/ `02_ai_frameworks`（框架层）/ `03_acceleration_libs`（加速库）/ `04_ops_programming`（编程、编译、NPU 架构） |
| 二级目录 | 层内子方向，带两位编号（如 `01_deep_learning`），对应总纲 1.1~4.6 小节 |
| 三级目录 | 每门原子课程一个目录，带两位编号（如 `01_math_and_tensor_basics/`），对应总纲一个 `Lx-yy` 编号 |
| 课程文件 | 每节课一个 Notebook 文件，命名 `NN.xx_<topic>.ipynb`（`NN`=章号，`xx`=节号），与现有 ascendc 教程风格一致 |
| 配套目录 | 每门课程目录内可含 `answer/`（练习答案）、`images/`（配图）、`src/`（源码）、`slides/`（课件 PDF） |
| 难度三级 | 初级（认知/首跑）→ 中级（系统实践）→ 高级（深度优化与扩展），目录内按难度顺序编号 |
| 双硬件路线 | L4 层 Ascend C 课程名含 `A2/A3` 或 `Ascend 950`，两条硬件路线独立成课、目录并列 |
| 状态标记 | ✅ 已建设（现有教程完整覆盖）/ 🟡 部分覆盖（现有教程覆盖部分内容）/ ❌ 待建设 |

---

## 二、目录结构总览

> **四层结构**：**第 1 层** 技术架构层（4 个）→ **第 2 层** 子方向（22 个）→ **第 3 层** 原子课程目录（135 门 + 4 门规划外补充，对应总纲 `Lx-yy` 编号）→ **第 4 层** 课程文件（**一节课一个 Notebook 文件**，45~90 分钟/节）。
>
> 下方目录树完整列出第 1~3 层全部目录；第 4 层文件每层各选 1 门代表课程展开示例（其余课程的第 4 层遵循同一模板，完整规范见第三节）。

```text
tutorials/                                                # 教程根目录（CANN-Learning-Hub）
├── 01_applications/                                      # 第 1 层 · L1 应用层（44 门）
│   ├── 01_deep_learning/                                 # 第 2 层 · 1.1 深度学习（8 门）
│   │   ├── 01_math_and_tensor_basics/                    # L1-01 深度学习数学与张量基础
│   │   │   ├── README.md                                 # 第 4 层 · 课程说明与索引
│   │   │   ├── 01.01_chapter_intro.ipynb                 # 第 4 层 · 每节课一个文件（45~90 分钟/节）
│   │   │   ├── 01.02_linear_algebra.ipynb
│   │   │   ├── 01.03_calculus_and_gradient.ipynb
│   │   │   ├── 01.0x_chapter_practice.ipynb              # 章节实践（易/中/难三档练习）
│   │   │   ├── answer/                                   # 练习参考答案
│   │   │   └── images/                                   # 章节配图（src/、slides/ 可选）
│   │   ├── 02_dl_framework_introduction/                 # L1-02 深度学习框架入门
│   │   ├── 03_cnn_basics/                                # L1-03 卷积神经网络基础
│   │   ├── 04_sequence_models_rnn/                       # L1-04 序列模型与循环网络
│   │   ├── 05_transformer_architecture/                  # L1-05 Transformer 架构与自注意力机制
│   │   ├── 06_training_optimization/                     # L1-06 深度学习训练优化
│   │   ├── 07_inference_optimization/                    # L1-07 深度学习推理优化
│   │   └── 08_distributed_training_basics/               # L1-08 分布式训练基础
│   ├── 02_computer_vision/                               # 第 2 层 · 1.2 计算机视觉（6 门）
│   │   ├── 01_cv_basics_image_preprocessing/             # L1-09 计算机视觉基础与图像预处理
│   │   ├── 02_object_detection/                          # L1-10 目标检测算法
│   │   ├── 03_image_segmentation/                        # L1-11 图像分割算法
│   │   ├── 04_visual_understanding_temporal/             # L1-12 视觉理解与时序建模
│   │   ├── 05_cv_deployment_optimization/                # L1-13 计算机视觉部署与优化
│   │   └── 06_3d_vision_point_cloud/                     # L1-14 三维视觉与点云处理
│   ├── 03_generative_ai/                                 # 第 2 层 · 1.3 生成式 AI（7 门）
│   │   ├── 01_genai_basics_probabilistic_models/         # L1-15 生成式 AI 基础与概率模型
│   │   ├── 02_diffusion_models/                          # L1-16 扩散模型原理
│   │   ├── 03_text_to_image/                             # L1-17 文本到图像生成
│   │   ├── 04_gan/                                       # L1-18 生成对抗网络
│   │   ├── 05_speech_generation_tts/                     # L1-19 语音生成与 TTS
│   │   ├── 06_video_generation/                          # L1-20 视频生成模型
│   │   └── 07_genai_deployment_inference/                # L1-21 生成式 AI 部署与推理优化
│   ├── 04_large_language_models/                         # 第 2 层 · 1.4 大语言模型（11 门）
│   │   ├── 01_llm_fundamentals_history/                  # L1-22 大语言模型基础与发展历程
│   │   ├── 02_prompt_engineering_icl/                    # L1-23 提示工程与上下文学习
│   │   ├── 03_llm_fine_tuning/                           # L1-24 大模型微调技术
│   │   ├── 04_rag/                                       # L1-25 RAG 检索增强生成
│   │   ├── 05_llm_evaluation_safety/                     # L1-26 大模型评估与安全
│   │   ├── 06_llm_pretraining/                           # L1-27 大模型预训练
│   │   ├── 07_alignment_rlhf/                            # L1-28 对齐与 RLHF
│   │   ├── 08_kv_cache_inference_optimization/           # L1-29 KV Cache 与推理优化
│   │   ├── 09_llm_distributed_training/                  # L1-30 大模型分布式训练
│   │   ├── 10_agents_multimodal_llm/                     # L1-31 智能体与多模态大模型
│   │   └── 11_llm_quantization_compression/              # L1-32 模型量化与压缩
│   ├── 05_scientific_computing/                          # 第 2 层 · 1.5 科学计算（9 门）
│   │   ├── 01_hpc_overview/                              # L1-33 科学计算与高性能计算概述
│   │   ├── 02_parallel_programming_basics/               # L1-34 并行编程基础
│   │   ├── 03_parallel_patterns/                         # L1-35 科学计算的并行模式
│   │   ├── 04_dense_linear_algebra/                      # L1-36 稠密线性代数
│   │   ├── 05_sparse_linear_algebra/                     # L1-37 稀疏线性代数
│   │   ├── 06_fft_spectral_methods/                      # L1-38 傅里叶变换与谱方法
│   │   ├── 07_sparse_matrix_graph_computing/             # L1-39 稀疏矩阵与图计算
│   │   ├── 08_performance_optimization_cases/            # L1-40 科学计算性能优化案例
│   │   └── 09_supercomputing_large_scale_parallel/       # L1-41 超算系统与大规模并行
│   └── 06_recommendation_system/                         # 第 2 层 · 1.6 推荐系统（3 门）
│       ├── 01_ctr_din_dssm/                              # L1-42 CTR 预测、DIN、DSSM
│       ├── 02_autofusion_dynamic_graph_optimization/     # L1-43 推荐系统自动融合与动态图性能优化
│       └── 03_multi_card_training_custom_ops/            # L1-44 推荐系统多卡训练调优与自定义算子开发
├── 02_ai_frameworks/                                     # 第 1 层 · L2 框架层（29 门）
│   ├── 01_framework_basics/                              # 第 2 层 · 2.1 框架基础（4 门）
│   │   ├── 01_pytorch_npu_quickstart/                    # L2-01 PyTorch NPU 快速入门
│   │   │   ├── README.md                                 # 第 4 层 · 课程文件示例（完整模板见第三节）
│   │   │   ├── 01.01_chapter_intro.ipynb
│   │   │   └── 01.0x_chapter_practice.ipynb
│   │   ├── 02_tensorflow_npu_quickstart/                 # L2-02 TensorFlow NPU 快速入门
│   │   ├── 03_ge_framework_introduction/                 # L2-03 GE 框架入门
│   │   └── 04_ascend_ecosystem_cann_architecture/        # L2-29 昇腾 AI 产业生态与 CANN 架构基础
│   ├── 02_training_techniques/                           # 第 2 层 · 2.2 训练技术（10 门）
│   │   ├── 01_dataloader_training_pipeline/              # L2-04 PyTorch NPU 数据加载与训练流程
│   │   ├── 02_llm_training_introduction/                 # L2-05 大模型训练入门
│   │   ├── 03_torchnpu_practice/                         # L2-06 TorchNPU 实践
│   │   ├── 04_torchair/                                  # L2-07 TorchAir
│   │   ├── 05_receipt_llm_training/                      # L2-08 基于 Receipt 的 Torch NPU 大模型训练实践
│   │   ├── 06_aclgraph/                                  # L2-09 AclGraph
│   │   ├── 07_distributed_training_framework/            # L2-10 分布式训练框架
│   │   ├── 08_cann_sft_rl_basics/                        # L2-20 基于 CANN 如何训练大模型：SFT、RL 训练基础
│   │   ├── 09_cann_sft_rl_advanced_varlen_cp/            # L2-21 基于 CANN 训练大模型进阶：SFT、RL，varlen+CP 融合
│   │   └── 10_cann_sft_rl_expert/                        # L2-22 大模型训练高阶（SFT、RL）：图优化、显存调优、端到端吞吐
│   ├── 03_inference_techniques/                          # 第 2 层 · 2.3 推理技术（8 门）
│   │   ├── 01_llm_inference_deployment/                  # L2-11 大模型推理部署
│   │   ├── 02_receipt_llm_deployment/                    # L2-12 基于 Receipt 的 Torch NPU 大模型部署实践
│   │   ├── 03_receipt_inference_optimization/            # L2-13 基于 Receipt 的大模型推理优化
│   │   ├── 04_inference_serving_benchmark/               # L2-14 推理服务部署与压测
│   │   ├── 05_model_quantization/                        # L2-15 模型量化技术
│   │   ├── 06_cann_deployment_inference_basics/          # L2-23 基于 CANN 如何部署和推理
│   │   ├── 07_cann_llm_inference_optimization/           # L2-24 基于 CANN 大模型推理优化和最佳实践
│   │   └── 08_cann_llm_inference_advanced/               # L2-25 基于 CANN 大模型推理优化高级实践
│   ├── 04_advanced_topics/                               # 第 2 层 · 2.4 进阶主题（4 门）
│   │   ├── 01_torchnpu_graph_fusion/                     # L2-16 TorchNPU 图融合
│   │   ├── 02_ai_infrastructure/                         # L2-17 AI 基础设施底层
│   │   ├── 03_reinforcement_learning_advanced/           # L2-18 强化学习进阶
│   │   └── 04_multimodal_llm/                            # L2-19 多模态大模型
│   └── 05_graph_frameworks/                              # 第 2 层 · 2.5 图框架（3 门）
│       ├── 01_ge_torchair_autofusion_basics/             # L2-26 GE 图引擎、TorchAir、AutoFusion
│       ├── 02_graph_compilation_custom_op_integration/   # L2-27 图编译与自定义算子入图
│       └── 03_autofusion_superkernel/                    # L2-28 自动融合与 SuperKernel
├── 03_acceleration_libs/                                 # 第 1 层 · L3 加速库（13 门 + 2 门规划外补充）
│   ├── 01_aclnn/                                         # 第 2 层 · 3.1 Aclnn 算子库（2 门）
│   │   ├── 01_aclnn_introduction_usage/                  # L3-01 Aclnn 算子库介绍与调用
│   │   │   ├── README.md                                 # 第 4 层 · 课程文件示例（完整模板见第三节）
│   │   │   ├── 01.01_chapter_intro.ipynb
│   │   │   └── 01.0x_chapter_practice.ipynb
│   │   └── 02_aclnn_custom_extension/                    # L3-02 Aclnn 算子库自定义扩展指南
│   ├── 02_ge/                                            # 第 2 层 · 3.2 GE 加速库（3 门）
│   │   ├── 01_operator_ge_integration/                   # L3-03 算子接入 GE 入图
│   │   ├── 02_custom_graph_construction/                 # L3-04 自定义构建图指南
│   │   └── 03_custom_graph_pass_optimization/            # L3-05 自定义扩展图 PASS 优化指南
│   ├── 03_communication/                                 # 第 2 层 · 3.3 通信算子库（2 门 + 2 门补充）
│   │   ├── 01_hccl_usage_guide/                          # L3-06 HCCL 通信库使用指南
│   │   ├── 02_asc_custom_communication_ops/              # L3-07 基于 ASC 自定义扩展开发通信算子指南
│   │   ├── 03_hixl_development/                          # 补充① HiXL 单边通信库开发
│   │   └── 04_mc2_fused_operator/                        # 补充② MC2 融合算子开发（支撑 L4-12/L4-30）
│   ├── 04_thrust/                                        # 第 2 层 · 3.4 C++ 并行算法库 Thrust（2 门）
│   │   ├── 01_thrust_introduction/                       # L3-08 C++ 并行算法库入门
│   │   └── 02_thrust_algorithm_practice/                 # L3-09 典型算法的 C++ 并行算法实践
│   └── 05_python_acceleration/                           # 第 2 层 · 3.5 Python 加速库（4 门）
│       ├── 01_python_parallel_introduction/              # L3-10 Python 并行计算入门
│       ├── 02_asnumpy_practice/                          # L3-11 AsNumpy 编程实践
│       ├── 03_math_python_practice/                      # L3-12 Math-Python 编程实践
│       └── 04_thrust_python_practice/                    # L3-13 并行算法实践：Thrust-Python
└── 04_ops_programming/                                   # 第 1 层 · L4 编程、编译、NPU 架构（49 门 + 2 门规划外补充）
    ├── 01_ascendc/                                       # 第 2 层 · 4.1 Ascend C 算子编程（30 门 + 1 门补充）
    │   ├── 01_introduction/                              # L4-01 异构计算与 Ascend C SIMD 算子编程导论
    │   │   ├── README.md                                 # 第 4 层 · 课程文件示例（完整模板见第三节）
    │   │   ├── 01.01_chapter_intro.ipynb
    │   │   └── 01.0x_chapter_practice.ipynb
    │   ├── 02_a2a3_simd_programming_model/               # L4-02 A2/A3 Ascend C SIMD 编程模型
    │   ├── 03_a2a3_simd_memory_vector/                   # L4-03 A2/A3 Ascend C SIMD Memory 矢量编程
    │   ├── 04_a2a3_simd_matmul/                          # L4-04 A2/A3 Ascend C SIMD 矩阵算子编程
    │   ├── 05_a2a3_simd_fused_operator/                  # L4-05 A2/A3 Ascend C SIMD 融合算子编程
    │   ├── 06_a2a3_typical_matmul_practice/              # L4-06 A2/A3 典型矩阵算子开发实践
    │   ├── 07_a2a3_typical_vector_practice/              # L4-07 A2/A3 典型矢量算子开发实践
    │   ├── 08_a2a3_debug_tuning/                         # L4-08 A2/A3 Ascend C 算子调试调优与最佳实践
    │   ├── 09_950_simd_programming_model/                # L4-09 Ascend 950 Ascend C SIMD 编程模型
    │   ├── 10_950_simd_reg_vector/                       # L4-10 Ascend 950 Ascend C SIMD Reg 矢量编程
    │   ├── 11_950_simd_matmul/                           # L4-11 Ascend 950 Ascend SIMD 矩阵算子编程
    │   ├── 12_950_simd_fused_operator/                   # L4-12 Ascend 950 Ascend SIMD 融合算子编程
    │   ├── 13_950_simt_programming/                      # L4-13 Ascend 950 Ascend C SIMT 算子编程
    │   ├── 14_950_typical_matmul_practice/               # L4-14 Ascend 950 典型矩阵算子开发实践
    │   ├── 15_950_typical_vector_practice/               # L4-15 Ascend 950 典型矢量（SIMD&SIMT）算子开发实践
    │   ├── 16_950_simd_debug_tuning/                     # L4-16 Ascend 950 Ascend C SIMD 算子调试调优与最佳实践
    │   ├── 17_950_simt_debug_tuning/                     # L4-17 Ascend 950 Ascend C SIMT 算子调试调优与最佳实践
    │   ├── 18_pytorch_single_operator_call/              # L4-18 Ascend C PyTorch 单算子调用
    │   ├── 19_ecosystem_libraries/                       # L4-19 Ascend C 开发生态库：asc-stl、asc-mathdx、高阶 API
    │   ├── 20_a2a3_vector_extreme_performance/           # L4-20 A2/A3 Ascend C 矢量算子极致性能实践
    │   ├── 21_a2a3_matmul_extreme_performance/           # L4-21 A2/A3 Ascend C 矩阵算子极致性能实践
    │   ├── 22_a2a3_fused_extreme_performance/            # L4-22 A2/A3 Ascend C 融合算子极致性能实践
    │   ├── 23_950_vector_extreme_performance/            # L4-23 Ascend 950 Ascend C 矢量算子极致性能实践
    │   ├── 24_950_matmul_extreme_performance/            # L4-24 Ascend 950 Ascend C 矩阵算子极致性能实践
    │   ├── 25_950_fused_extreme_performance/             # L4-25 Ascend 950 Ascend C 融合算子极致性能实践
    │   ├── 26_950_simd_simt_hybrid_programming/          # L4-26 Ascend 950 SIMD&SIMT 混合编程
    │   ├── 27_950_simd_simt_hybrid_practice/             # L4-27 Ascend 950 SIMD&SIMT 混合算子开发实践
    │   ├── 28_operator_graph_integration/                # L4-28 Ascend C 算子入图（PyTorch/AclGraph/GE）
    │   ├── 29_aclnn_engineering_development/             # L4-29 Aclnn 算子工程化开发
    │   └── 30_custom_communication_ops/                  # L4-30 Ascend C 通信算子自定义开发
    ├── 02_pypto/                                         # 第 2 层 · 4.2 PyPTO 算子编程（5 门 + 1 门补充）
    │   ├── 01_pypto_introduction/                        # L4-31 PyPTO 算子编程导论
    │   ├── 02_pypto_vector_development_tuning/           # L4-32 PyPTO 矢量算子开发与调优实践
    │   ├── 03_pypto_matmul_development_tuning/           # L4-33 PyPTO 矩阵算子开发与调优实践
    │   ├── 04_pypto_fused_development_tuning/            # L4-34 PyPTO 融合算子开发与调优实践
    │   ├── 05_pypto_custom_extension_pass/               # L4-35 PyPTO 自定义扩展优化 PASS 实践
    │   └── 06_pypto_ai_agent_dev/                        # 补充④ PyPTO AI Agent 算子开发
    ├── 03_tilelang/                                      # 第 2 层 · 4.3 TileLang 算子编程（5 门）
    │   ├── 01_tilelang_introduction/                     # L4-36 TileLang 算子编程导论
    │   ├── 02_tilelang_vector_development_tuning/        # L4-37 TileLang 矢量算子开发与调优实践
    │   ├── 03_tilelang_matmul_development_tuning/        # L4-38 TileLang 矩阵算子开发与调优实践
    │   ├── 04_tilelang_fused_development_tuning/         # L4-39 TileLang 融合算子开发与调优实践
    │   └── 05_tilelang_expert_mode/                      # L4-40 TileLang 专家模式与高级特性
    ├── 04_pyasc/                                         # 第 2 层 · 4.4 PyAsc 算子编程（4 门）
    │   ├── 01_pyasc_introduction/                        # L4-41 PyAsc 算子编程导论
    │   ├── 02_pyasc_tensor_development_tuning/           # L4-42 PyAsc Tensor 编程开发与调优实践
    │   ├── 03_pyasc_simt_development_tuning/             # L4-43 PyAsc SIMT 编程开发与调优实践
    │   └── 04_pyasc_advanced_features/                   # L4-44 PyAsc 高级特性编程实践
    ├── 05_cannbot/                                       # 第 2 层 · 4.5 基于 CANNBot 开发算子实践（2 门）
    │   ├── 01_cannbot_introduction_practice/             # L4-45 CANNBot 介绍与编程实践
    │   └── 02_cannbot_knowledge_base_skills/             # L4-46 CANNBot 知识库与 Skills 贡献实践
    └── 06_npu_architecture/                              # 第 2 层 · 4.6 NPU 体系架构（3 门）
        ├── 01_architecture_challenges_ai_era/            # L4-47 AI 时代对体系架构的挑战
        ├── 02_npu_architecture_response/                 # L4-48 NPU 如何应对 AI 时代架构挑战
        └── 03_single_chip_optimization_supernode/        # L4-49 NPU 的单芯片性能优化以及超节点演进
```

---

## 三、标准课程目录与文件模板

每门原子课程目录内**一节课一个文件**，命名与现有 ascendc 教程体系保持一致：

```text
tutorials/<层>/<方向>/<课程>/
├── README.md                     # 课程简介、软硬件配套、课程内容索引
├── NN.01_chapter_intro.ipynb     # 第 1 节：章节介绍
├── NN.02_<topic_1>.ipynb         # 第 2 节：主题 1（45~90 分钟一节）
├── NN.03_<topic_2>.ipynb         # 第 3 节：主题 2
├── NN.0x_chapter_practice.ipynb  # 章节实践（含易/中/难三档练习）
├── NN.0x_chapter_test.ipynb      # 章节自测（可选）
├── answer/                       # 练习参考答案
├── images/                       # 章节配图
├── src/                          # 配套源码（可选）
└── slides/                       # 授课课件 PDF（可选，纯讲授型课程必配）
```

**文件级规划样例**（其余课程遵循同样模板）：

<details>
<summary>样例 1：L4-02 A2/A3 Ascend C SIMD 编程模型（对应现有 ascendc_operator_development/02 章）</summary>

```text
04_ops_programming/01_ascendc/02_a2a3_simd_programming_model/
├── README.md
├── 02.01_chapter_intro.ipynb
├── 02.02_a2a3_architecture_overview.ipynb
├── 02.03_kernel_function_definition.ipynb
├── 02.04_memory_hierarchy.ipynb
├── 02.05_synchronization_and_stream.ipynb
├── 02.06_operator_compilation.ipynb
├── 02.07_vector_add_complete_impl.ipynb
├── 02.08_chapter_practice.ipynb
├── answer/
├── images/
└── src/
```

</details>

<details>
<summary>样例 2：L1-01 深度学习数学与张量基础（新建课程）</summary>

```text
01_applications/01_deep_learning/01_math_and_tensor_basics/
├── README.md
├── 01.01_chapter_intro.ipynb
├── 01.02_linear_algebra.ipynb
├── 01.03_calculus_and_gradient.ipynb
├── 01.04_tensor_representation_ops.ipynb
├── 01.05_npu_tensor_layout_nd_nz.ipynb
├── 01.06_chapter_practice.ipynb
├── answer/
└── images/
```

</details>

---

## 四、现有教程迁移映射

现有 15 个 tutorials 教程目录迁入新目录结构的建议映射（迁移期间可在原目录留 README 跳转链接）。「迁入目标目录」为教程的物理落位；「内容级素材支撑」为同一教程额外可作为其他原子课程素材的交叉引用（与第二节目录树及第五节校验清单逐条对账）：

| 现有教程（tutorials/） | 迁入目标目录 | 内容级素材支撑（交叉引用） |
|------------------------|-------------|---------------------------|
| llm_inference | `02_ai_frameworks/03_inference_techniques/06_cann_deployment_inference_basics/` 与 `07_cann_llm_inference_optimization/`（qwen3_1.7B/8B 实践分别支撑两课） | 课件 01（LLM 基础）→ L1-22；课件 02（CANN 开源推理仓）→ L2-12/L2-13；课件 04 + qwen3_8B 量化实践 → L1-32/L2-15/L2-25；NPU 优化实践 → L1-29；qwen3_8B/06 自定义量化 MatMul 算子开发与模型接入 → L4-18/L4-28 |
| sft_training_pipeline | `02_ai_frameworks/02_training_techniques/08_cann_sft_rl_basics/`（SFT 部分） | L1-24（大模型微调技术，Wordle 场景实践素材） |
| rl_training_pipeline | `02_ai_frameworks/02_training_techniques/08~10_cann_sft_rl_*/`（第 1~4 章支撑 08，第 5~8 章支撑 09/10） | L1-28（对齐与 RLHF，GRPO 素材）；L2-18（强化学习进阶，GRPO/PPO 部分） |
| ascendc_operator_development | `04_ops_programming/01_ascendc/01~08、18、29/`（章节级拆分迁入） | — |
| ascendc_operator_development_light | `04_ops_programming/01_ascendc/00_ascendc_light/`（整体保留为轻量平行版） | 02.06 kernel_pytorch_call → L4-18 素材 |
| conv_operator_development | `04_ops_programming/01_ascendc/06_a2a3_typical_matmul_practice/conv_operator_development/`（Conv 专项子目录） | 同时支撑 L4-04（`04_a2a3_simd_matmul/`） |
| MC2_fused_operator_development | `03_acceleration_libs/03_communication/04_mc2_fused_operator/` | L4-12/L4-30（Matmul+MTE 通信融合素材） |
| hccl_development | `03_acceleration_libs/03_communication/01_hccl_usage_guide/`（初级）与 `02_asc_custom_communication_ops/`（中级） | 中级 AICPU_TS/CCU 算子开发 → L4-30 素材 |
| hixl_development | `03_acceleration_libs/03_communication/03_hixl_development/` | — |
| autofusion_development | `02_ai_frameworks/05_graph_frameworks/01_ge_torchair_autofusion_basics/`（基础）与 `03_autofusion_superkernel/`（原理） | — |
| ge_development | `02_ai_frameworks/01_framework_basics/03_ge_framework_introduction/`（01 基础概念章节）、`05_graph_frameworks/02_graph_compilation_custom_op_integration/`（03 图编译章节）与 `03_acceleration_libs/02_ge/01~02/`（02/04 章构图·执行优化） | L2-03（GE 框架入门）；L2-26（基础概念部分） |
| TorchAir_development | `02_ai_frameworks/02_training_techniques/04_torchair/` | 内容偏图模式，可在 `05_graph_frameworks/01/` 建交叉引用（L2-26） |
| pyasc_operator_development | `04_ops_programming/04_pyasc/01~02/`（后续 03/04 章按规划补齐） | — |
| pypto_development | `04_ops_programming/02_pypto/01~06/`（章节对应迁入） | — |
| CANNBot | `04_ops_programming/05_cannbot/01~02/` | — |
| （tutorials/README.md） | 保留为 tutorials 总索引，按新四层结构重排教程列表 | — |

**迁移映射校验结论**：

- **目录级**：tutorials 现有 15 个教程目录全部有迁移行（15/15），无目录遗漏；
- **内容级**：本轮复核补注 6 处遗漏——① ge_development 的 01 基础概念章节落位 `01_framework_basics/03_ge_framework_introduction/`（原表遗漏 L2-03 目标目录）；② llm_inference 课件与实践向 L1-22/L1-29/L1-32/L2-12/L2-13/L2-15/L2-25 及 qwen3_8B 自定义算子实践向 L4-18/L4-28 的素材分流；③ sft_training_pipeline → L1-24；④ rl_training_pipeline → L1-28/L2-18；⑤ hccl_development 中级 → L4-30；⑥ conv_operator_development → L4-04；
- ascendc_operator_development（01~08、18、29）、TorchAir、autofusion、pypto、pyasc、CANNBot、hixl、MC2、light 九项映射经逐条核对一致，无遗漏。

---

## 五、与 tutorials 现有内容校验结果

### 5.1 覆盖统计

| 技术栈层 | 课程数 | ✅ 已建设 | 🟡 部分覆盖 | ❌ 待建设（遗漏） | 覆盖率（含部分） |
|---------|--------|----------|------------|------------------|------------------|
| L1 应用层 | 44 | 0 | 5 | 39 | 11.4% |
| L2 框架层 | 29 | 3 | 10 | 16 | 44.8% |
| L3 加速库 | 13 | 1 | 3 | 9 | 30.8% |
| L4 编程·编译·NPU 架构 | 49 | 13 | 7 | 29 | 40.8% |
| **合计** | **135** | **17** | **25** | **93** | **31.1%** |

> 说明：🟡 部分覆盖 = 现有教程覆盖该课部分内容（如仅实践场景、仅部分章节），仍需按原子课程大纲补齐成独立课程目录。

### 5.2 遗漏清单（规划有、tutorials 无 —— 按建设优先级排序）

**P0（总纲明示优先 + 学习路径入口）：**

1. **TileLang 全线 5 门**（L4-36~40）——tutorials README 明确"规划中教程：优先覆盖 TileLang"，目前仅 CANNBot 第 4 课涉及体验路径；
2. **应用层 44 门接近空白**——其中 39 门完全无教程，深度学习（L1-01~08）、计算机视觉（L1-09~14）、生成式 AI（L1-15~21）、大语言模型应用侧（L1-22~32）、科学计算（L1-33~41）、推荐系统（L1-42~44）均缺独立课程（仅 L1-22/24/28/29/32 被 LLM 推理/训练教程部分触及），建议按"深度学习 → LLM → 推荐系统"顺序优先启动；
3. **框架基础入口 4 门**（L2-01 PyTorch NPU 快速入门、L2-02 TF NPU、L2-03 GE 入门、L2-29 昇腾生态与 CANN 架构）——L2-29 可基于仓库 `quick_start/cann_basics` 内容整合迁入 tutorials。

**P1（现有主线的关键缺口）：**

4. **Ascend 950 路线中级 8 门**（L4-09~11、13~17）——现有 ascendc 教程仅覆盖 A2/A3 路线，950 仅 MC2 融合算子一门部分覆盖；
5. **Ascend C 极致性能 8 门**（L4-20~27，含 950 SIMD&SIMT 混合 2 门）；
6. **NPU 体系架构 3 门**（L4-47~49）——总纲"体系结构三幕剧"传统课程的教学化载体；
7. **Aclnn 算子库 2 门**（L3-01/02）——加速库唯一零覆盖的入门线；
8. **Python/C++ 加速库 6 门**（L3-08~13：Thrust ×2、Python 并行/AsNumpy/Math-Python/Thrust-Python）。

**P2（框架层进阶缺口）：**

9. **Receipt 训练/部署实践 3 门**（L2-08/12/13）；
10. **推理服务压测与模型量化专项 2 门**（L2-14/15）；
11. **TorchNPU 实践/图融合、AclGraph、分布式训练框架等 8 门**（L2-04/05/06/09/10/11/16/17）；
12. **多模态大模型（L2-19）、强化学习进阶补齐（L2-18）**；
13. **GE 图 PASS 优化（L3-05）、算子入图（L4-28）、开发生态库（L4-19）、PyPTO PASS（L4-35）、PyAsc SIMT/高级（L4-43/44）、CANNBot Skills 贡献补齐（L4-46）**。

### 5.3 规划外内容（tutorials 有、135 门规划未含 —— 建议补入规划）

校验同时发现 4 项现有教程内容超出原子课程规划范围，已在目录规划中以"规划外补充"显式承接，建议总纲后续升版时正式纳入编号：

| # | 现有教程 | 内容 | 承接方式 |
|---|---------|------|---------|
| ① | hixl_development | HiXL 单边通信库开发（基础/API/传输模式/调优） | 挂 `03_acceleration_libs/03_communication/03_hixl_development/`，建议总纲 3.3 节增设"HiXL 单边通信库开发"课程 |
| ② | MC2_fused_operator_development | MC2 融合算子开发（Matmul+MTE，Ascend 950 验证） | 挂 `03_acceleration_libs/03_communication/04_mc2_fused_operator/`，同时作为 L4-12/L4-30 实践素材 |
| ③ | ascendc_operator_development_light | Ascend C Kernel 直调轻量版（与主线并行的教学法变体） | 挂 `04_ops_programming/01_ascendc/00_ascendc_light/`，作为 L4-01~04 的平行版本目录 |
| ④ | pypto_development 第 05 章 | PyPTO AI Agent 算子开发（AI Agent 辅助算子开发新范式） | 挂 `04_ops_programming/02_pypto/06_pypto_ai_agent_dev/`，建议总纲 4.2 节增设对应课程，并与 L4-45/46（CANNBot）建立交叉引用 |

### 5.4 校验结论

- **135 门规划课程中，93 门在 tutorials 中尚无对应内容（68.9%）**，其中应用层缺口最大（44 门中 39 门完全空白）；
- 现有 15 个教程高度集中在 **L4 算子编程（A2/A3 Ascend C 主线、PyPTO、PyAsc 入门、CANNBot）与 L2 训练/推理管线实践**，可作为对应原子课程的拆分与迁移素材；
- 4 项规划外内容已反向补入目录规划（见 5.3），规划与现状双向无遗漏；
- 后续建设建议按 5.2 的 P0 → P1 → P2 优先级推进，责任人确认后补充回填。

---

*文档生成时间：2026-09-14；数据来源：`ref/atomic_courses.md`（课程内容与目录规划要求）、`standard_course/00_atomic_courses/README.md`（135 门总纲）、`tutorials/` 现有 15 个教程目录实地盘点。*

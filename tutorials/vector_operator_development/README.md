# Vector 算子开发课程

本课程专注于华为昇腾 NPU 的 Vector 计算单元，从芯片架构、SIMD/SIMT 编程模型，到实际的算子开发和优化，提供全面的学习路径。

课程以 **FastGelu**（标准 GELU 的 sigmoid 近似：`y = x / (1 + exp(-1.702 * x))`，其中 `1.702` 为经验系数）作为贯穿全课程的示例算子。

## 课程结构

### 第一章：基础掌握
介绍昇腾 AI 处理器的 Vector 计算架构，包括 A2/A3/A5 芯片架构、SIMD Membase/Regbase 编程、SIMT 编程模型等基础知识。

### 第二章：加速库基础认知
了解算子执行流程、算子库的目录结构、Tiling 原理以及开发环境的搭建。

### 第三章：Ascend 算子开发
深入学习 SIMD 和 SIMT 算子开发，包括 membase/regbase 编程、aclnn 和 PTA 接口调用、性能优化和调试技巧。

## 前置知识要求

- 具备 C/C++ 编程基础
- 了解深度学习框架（PyTorch/TensorFlow）基本概念
- 具备一定的并行编程基础
- 已完成 Ascend C 算子开发系列教程基础部分

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Ascend 950PR/Ascend 950DT |
| CANN 版本 | 8.0.0 及以上 |
| Python | 3.8+ |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 950尝鲜 | cann_9.0.0-beta.2-py3.12-a5 | Python 3.12 |参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)创建CANNLab环境运行notebook |

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)。

## 章节目录

### 第一章：基础掌握

| 编号 | 标题 | 内容 |
|------|------|------|
| 1.1 | 章节介绍 | 章节概述、前置要求、学习目标 |
| 1.2 | 芯片架构简介 | A2/A3/A5 芯片 AI Core 架构 |
| 1.3 | SIMD Membase 编程基础 | Membase 编程范式与常见指令 |
| 1.4 | SIMD Regbase 编程基础 | Regbase 编程模型、VF 寄存器与指令 |
| 1.5 | SIMT 编程基础 | SIMT 线程架构与内存层级 |
| 1.6 | 章节实践 | FastGelu regbase 完整实现 |

### 第二章：加速库基础认知

| 编号 | 标题 | 内容 |
|------|------|------|
| 2.1 | 章节介绍 | 章节概述、前置要求、学习目标 |
| 2.2 | 算子执行流程 | aclnn 单算子 API、GE 图模式、Kernel 直调 |
| 2.3 | 算子目录结构 | FastGelu 工程目录布局与 Tiling 原理 |
| 2.4 | 开发环境部署 | CANN Toolkit/Ops 安装与环境配置 |
| 2.5 | 章节实践 | FastGelu 算子工程搭建 |

### 第三章：Ascend 算子开发

| 编号 | 标题 | 内容 |
|------|------|------|
| 3.1 | 章节介绍 | 章节概述、前置要求、学习目标 |
| 3.2 | SIMD 算子开发 | membase 与 regbase 模式 FastGelu 完整开发 |
| 3.3 | SIMT 算子开发 | SIMT 算子开发实战 |
| 3.4 | 调用接口开发 | aclnn 两段式接口与 PTA 调用模式 |
| 3.5 | 算子调试 | 调试工具与调试方法 |
| 3.6 | 算子性能优化 | Double Buffer、UB 优化、Scalar 优化等 |
| 3.7 | 章节实践 | FastGelu 完整开发 + 性能优化 |

---

本课程从基础概念到实际开发，循序渐进地帮助您掌握 Vector 算子开发的完整流程。

## 参考资料

- [算子工程入门](../ascendc_operator_development/03_intermediate_vector_operator_development/03.02_operator_engineering_intro.ipynb) - InferShape / InferDtype 实现
- [开源仓算子开发](../ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.03_operator_development_based_on_opensource_repo.ipynb) - InferDataType 实现与注册
- [SIMT 同步机制详解](../ascendc_operator_development_V2/03_programming_model/03.04.05_simt_synchronization_mechanism.ipynb) - asc_syncthreads、asc_threadfence
- [Regbase 流水线同步](../../blogs/operator/regbase_vec_add/从一个向量加法出发，深入理解Regbase编程范式.md) - LocalMemBar 与同步控制
- [动态 Shape 执行](../ge_development/04_model_execution_optimization/04.03_dynamic_shape_execution.ipynb) - Unknown Shape 运行时调度
- [动态 Shape 优化](../ge_development/04_model_execution_optimization/04.05_dynamic_shape_optimization.ipynb) - Dynamic Gear 分档策略
- [开源仓算子交付件](../ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.02_opensource_repo_intro_and_verification.ipynb) - op_proto、op_impl、op_tiling 等
- [aclnn pybind 调用](../ascendc_operator_development/03_intermediate_vector_operator_development/03.03_acl_pybind_call.ipynb) - aclnn 封装与 Python 调测
- [aclnn 算子工程](../ascendc_operator_development_V2/06_advanced_features/06.03_aclnn_operator_engineering_development.ipynb) - OpDef、自动生成、编译流程
- [开源仓贡献流程](../ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.01_chapter_intro.ipynb) - 算子上库完整流程
- [MIX算子贡献实战](../../blogs/operator/transformer_experimental_mix_operator/transformer仓experimental路径MIX算子开发贡献.md) - 端到端贡献经验
- [GE 图构建与框架适配](../ge_development/03_graph_compilation/03.02_graph_build_and_input.ipynb) - PyTorch/TensorFlow/ONNX 差异
- [Kernel 直调编程](../../blogs/operator/kernel_direct_call_programming/算子Kernel直调编程.md) - 异构编程、AscendOps 模板
- [Scalar 高性能编码](../../blogs/operator/scalar_npu_operator_performance_optimization/scalar_npu_operator_performance_optimization.md) - 7 条编码原则
- [泛化 Tiling 设计](../ascendc_operator_development/03_intermediate_vector_operator_development/03.04_generalized_tiling_design.ipynb) - 算子开发规范

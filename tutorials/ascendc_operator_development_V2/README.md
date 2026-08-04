# Ascend C 算子开发系列教程

本教程面向昇腾 NPU，系统讲解 Ascend C 高性能算子开发方法，内容涵盖 CANN 与 NPU 基础、SIMD/SIMT 编程模型、多层级 API、典型算子实践、PyTorch 框架集成以及 GE 入图编译与运行，帮助开发者逐步掌握从核函数编写到框架调用的完整开发流程。

根据章节内容，教程目录中包含以下学习材料：

- **Notebooks**：包含课程知识点、示例代码与练习题，适用于自主学习或讲师引导式教学。
- **src**：包含部分实践课程使用的完整样例工程，可用于编译、运行和修改。

## 适用人群

- 希望了解并掌握 Ascend C 算子编程方法的开发者
- AI 算法工程师、C/C++ 开发者和算子开发工程师

## 前置知识要求

在学习本教程之前，建议具备以下基础：

- 掌握 C/C++ 编程基础
- 了解并行编程基本概念
- 熟悉 Linux 命令行、CMake 和基础编译流程

## 整体学习目标

完成本教程后，你将能够：

- 理解 CANN 与昇腾 NPU 基础，掌握 Ascend C 核函数开发流程及多层级 API 的基本使用方法
- 掌握 SIMD、SIMT 编程模型，并能够根据算子特点选择合适的编程方式
- 掌握矩阵、融合及离散访存等典型算子的开发、验证与基础优化方法
- 掌握自定义 Ascend C Kernel 在 PyTorch 框架下的调用方法
- 掌握自定义算子接入 GE 图的基本方法，理解 GE 图的编译与运行流程

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 主要支持硬件 | Ascend 950PR / Ascend 950DT（`dav-3510`） |
| 兼容硬件 | 部分 SIMD、CV 融合和 GE 教程支持 Atlas A2 / A3（`dav-2201`），以具体 Notebook 的说明为准 |
| CANN 版本 | 9.1.0 及以上 |
| Python | 3.11 |

> **注意：** SIMT、RegBase 和部分 Tensor API 能力仅支持 Ascend 950PR / Ascend 950DT。运行样例前，请先查看对应 Notebook 中的硬件兼容性和编译参数说明。

## 在线体验环境

本教程中的概念讲解可直接通过 Notebook 阅读。如需编译和运行 Ascend 950 专属样例，建议使用 CANNLab Ascend 950 配套环境，具体请参考 [CANNLab 环境体验指南](../../docs/CANNLab_env_experience_guide.md)。

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 快速安装](https://www.hiascend.com/cann/download)。

## 第一章：前置知识

| Notebook | 状态 |
| --- | --- |
| [1.1 章节介绍](01_basic_overview/01.01_chapter_intro.ipynb) | 🚧 开发中 |
| [1.2 人工智能与算子基础](01_basic_overview/01.02_ai_and_operator_basics.ipynb) | 🚧 开发中 |
| [1.3 CANN 架构与昇腾 NPU 原理](01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb) | 🚧 开发中 |
| [1.4 算子开发的基本概念](01_basic_overview/01.04_operator_basic_concepts.ipynb) | 🚧 开发中 |
| [1.5 章节测试](01_basic_overview/01.05_chapter_test.ipynb) | 🚧 开发中 |

## 第二章：算子入门

| Notebook | 状态 |
| --- | --- |
| [2.1 章节介绍](02_ascendc_operator_basics/02.01_chapter_intro.ipynb) | 🚧 开发中 |
| [2.2 SIMT 与 SIMD 介绍](02_ascendc_operator_basics/02.02_simt_simd_introduction.ipynb) | 🚧 开发中 |
| [2.3 对应典型算子结构](02_ascendc_operator_basics/02.03_typical_operator_structure.ipynb) | 🚧 开发中 |
| [2.4 Ascend C 的 Hello World](02_ascendc_operator_basics/02.04_hello_world.ipynb) | 🚧 开发中 |
| [2.5 SIMD 连续类矢量算子示例（C API Add 算子）](02_ascendc_operator_basics/02.05_simd_continuous_vector_c_api.ipynb) | 🚧 开发中 |
| [2.6 SIMD 矩阵算子示例（Tensor 编程）](02_ascendc_operator_basics/02.06_simd_matrix_tensor_api.ipynb) | 🚧 开发中 |
| [2.7 SIMT 离散类矢量算子示例（Gather 算子）](02_ascendc_operator_basics/02.07_simt_discrete_vector.ipynb) | 🚧 开发中 |
| [2.8 章节测试](02_ascendc_operator_basics/02.08_chapter_test.ipynb) | 🚧 开发中 |

## 第三章：Ascend C 编程模型

| Notebook | 状态 |
| --- | --- |
| [3.1 章节介绍](03_programming_model/03.01_chapter_intro.ipynb) | 🚧 开发中 |
| [3.2 编程模型概述](03_programming_model/03.02_programming_model_overview.ipynb) | 🚧 开发中 |
| [3.3.1 SIMD 抽象硬件架构简介](03_programming_model/03.03.01_simd_abstract_hardware_architecture_intro.ipynb) | 🚧 开发中 |
| [3.3.2 SIMD 核函数](03_programming_model/03.03.02_simd_ascendc_kernel_function.ipynb) | 🚧 开发中 |
| [3.3.3 SIMD 编程模型与接口概述](03_programming_model/03.03.03_simd_programming_api.ipynb) | 🚧 开发中 |
| [3.3.4 基于指针的 C 语言 SIMD 编程](03_programming_model/03.03.04_simd_c_programming_base_on_pointer.ipynb) | 🚧 开发中 |
| [3.3.5 基于 Tensor 的 C++ SIMD 编程](03_programming_model/03.03.05_simd_c++_programming_base_on_tensor.ipynb) | 🚧 开发中 |
| [3.3.6 基于 TPipe/TQue 框架的 SIMD 编程](03_programming_model/03.03.06_simd_programming_base_on_Tpipe_and_Tque.ipynb) | 🚧 开发中 |
| [3.3.7 基于静态 Tensor 框架的 SIMD 编程](03_programming_model/03.03.07_simd_programming_base_on_static_tensor.ipynb) | 🚧 开发中 |
| [3.3.8 SIMD 章节实践](03_programming_model/03.03.08_simd_practice.ipynb) | 🚧 开发中 |
| [3.4.1 SIMT 抽象硬件架构](03_programming_model/03.04.01_simt_abstract_hardware_architecture_intro.ipynb) | 🚧 开发中 |
| [3.4.2 SIMT 线程架构](03_programming_model/03.04.02_simt_thread_architecture_intro.ipynb) | 🚧 开发中 |
| [3.4.3 SIMT 核函数](03_programming_model/03.04.03_simt_kernel_function.ipynb) | 🚧 开发中 |
| [3.4.4 SIMT 内存层级](03_programming_model/03.04.04_simt_memory_hierarchy.ipynb) | 🚧 开发中 |
| [3.4.5 SIMT 同步机制](03_programming_model/03.04.05_simt_synchronization_mechanism.ipynb) | 🚧 开发中 |
| [3.4.6 SIMT 编程 API](03_programming_model/03.04.06_simt_programming_api.ipynb) | 🚧 开发中 |
| [3.4.7 SIMT 课后实践](03_programming_model/03.04.07_simt_pratice.ipynb) | 🚧 开发中 |

## 第四章：简单算子实践

| Notebook | 状态 |
| --- | --- |
| [4.1 章节介绍](04_simple_operator_practice/04.01_chapter_intro.ipynb) | 🚧 开发中 |
| [4.2 Softmax 算子教程](04_simple_operator_practice/04.02_softmax.ipynb) | 🚧 开发中 |
| [4.3 Tensor API 矩阵算子优化实践](04_simple_operator_practice/04.03_tensor_api_matmul.ipynb) | 🚧 开发中 |
| [4.4 基于静态 Tensor 的 CV 融合算子开发](04_simple_operator_practice/04.04_static_tensor_cv_fusion.ipynb) | 🚧 开发中 |
| [4.5 CV 融合算子教程](04_simple_operator_practice/04.05_cv_fusion.ipynb) | 🚧 开发中 |
| [4.6 SIMT Gather 算子开发](04_simple_operator_practice/04.06_simt_gather_operator.ipynb) | 🚧 开发中 |

## 第五章：框架集成

| Notebook | 状态 |
| --- | --- |
| [5.1 章节介绍](05_framework_integration/05.01_chapter_intro.ipynb) | 🚧 开发中 |
| [5.2 PyTorch 框架下 Kernel 直调](05_framework_integration/05.02_kernel_pytorch_call.ipynb) | 🚧 开发中 |

## 第六章：高级特性

| Notebook | 状态 |
| --- | --- |
| [6.1 章节介绍](06_advanced_features/06.01_chapter_intro.ipynb) | 🚧 开发中 |
| [6.3 GE 入图编译与运行](06_advanced_features/06.03_ge_compile_launch.ipynb) | 🚧 开发中 |

## 第七章：高级算子实践

| Notebook | 状态 |
| --- | --- |
| [7.1 章节介绍](07_advanced_operator_practice/07.01_chapter_intro.ipynb) | 🚧 开发中 |

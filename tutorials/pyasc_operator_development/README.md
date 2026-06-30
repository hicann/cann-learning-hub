# pyasc算子开发系列教程

本教程是Ascend C算子开发系列教程的姊妹篇，带你学习使用pyasc——一种基于Python原生语法的昇腾NPU高性能算子开发方式。pyasc编程接口与Ascend C类库接口一一对应，让你可以用熟悉的Python语法完成算子开发。

教程按章节划分，每个章节均包含以下内容：
- Notebooks：包含课程知识点与练习题，适用于自主学习或讲师引导式教学，可在 gitcode 提供的轻量级 notebook 上运行。也可自行搭建jupyter lab，在本地环境执行使用。
- SRC：包含课程中所有的源码，供开发者自行下载及修改。

> **注意：**
> - 本教程**前置依赖Ascend C算子开发系列教程**，建议先完成Ascend C课程初级前两章的学习，掌握算子基础概念和核函数开发方法后再学习本课程。
> - 本教程当前针对Atlas A2（910B）和Atlas A3（910C）系列产品进行验证。

## 适用对象

- 已学习Ascend C算子开发系列教程（至少完成初级前两章）。
- 熟悉Python编程，希望用Python原生语法开发昇腾算子。
- 了解算子开发基本概念（核函数、Tiling、数据搬运、流水同步等）。

## 整体学习目标

- 掌握pyasc核函数开发与调用方式，理解@asc.jit编译机制。
- 掌握基于TPipe/TQue框架的工程化开发方式。
- 掌握Matmul高阶API开发MIX/Cube模式矩阵算子。
- 掌握融合算子开发方式。
- 掌握pyasc调试接口（printf/dump_tensor）和性能调优工具（msprof op）。

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2（910B）、Atlas A3（910C） |
| CANN版本 | 8.5.0.alpha001及以上 |
| pyasc版本 | v1.1.0及以上 |
| Python | 3.9-3.12 |
| CPU架构 | aarch64 / x86_64 |

## 课程大纲

### pyasc算子开发系列（初级）

#### 第一章：pyasc概述与环境准备

| Notebook | 链接 | 状态 |
|--|--|--|
| 1.1 章节介绍 | [在线体验](./01_pyasc_overview/01.01_chapter_intro.ipynb) | 🚧 开发中 |
| 1.2 pyasc简介 | [在线体验](./01_pyasc_overview/01.02_pyasc_introduction.ipynb) | 🚧 开发中 |
| 1.3 环境准备 | [在线体验](./01_pyasc_overview/01.03_environment_setup.ipynb) | 🚧 开发中 |
| 1.4 首个样例执行 | [在线体验](./01_pyasc_overview/01.04_first_sample_execution.ipynb) | 🚧 开发中 |
| 1.5 章节实践 | [在线体验](./01_pyasc_overview/01.05_chapter_practice.ipynb) | 🚧 开发中 |

#### 第二章：pyasc核函数开发基础

| Notebook | 链接 | 状态 |
|--|--|--|
| 2.1 章节介绍 | [在线体验](./02_kernel_function_basic/02.01_chapter_intro.ipynb) | 🚧 开发中 |
| 2.2 pyasc编程范式 | [在线体验](./02_kernel_function_basic/02.02_pyasc_programming_paradigm.ipynb) | 🚧 开发中 |
| 2.3 核函数定义与JIT编译 | [在线体验](./02_kernel_function_basic/02.03_kernel_function_and_jit.ipynb) | 🚧 开发中 |
| 2.4 Tensor与数据搬运 | [在线体验](./02_kernel_function_basic/02.04_tensor_and_data_transfer.ipynb) | 🚧 开发中 |
| 2.5 手动流水同步 | [在线体验](./02_kernel_function_basic/02.05_manual_pipeline_sync.ipynb) | 🚧 开发中 |
| 2.6 核函数调用与验证 | [在线体验](./02_kernel_function_basic/02.06_kernel_launch_and_verification.ipynb) | 🚧 开发中 |
| 2.7 章节实践 | [在线体验](./02_kernel_function_basic/02.07_chapter_practice.ipynb) | 🚧 开发中 |

### pyasc算子开发系列（中级）

#### 第三章：Vector算子工程化开发

| Notebook | 链接 | 状态 |
|--|--|--|
| 3.1 章节介绍 | [在线体验](./03_engineering_development/03.01_chapter_intro.ipynb) | 🚧 开发中 |
| 3.2 TPipe/TQue框架编程 | [在线体验](./03_engineering_development/03.02_tpipe_tque_framework.ipynb) | 🚧 开发中 |
| 3.3 Python语法约束 | [在线体验](./03_engineering_development/03.03_python_syntax_constraints.ipynb) | 🚧 开发中 |
| 3.4 ConstExpr与Device函数 | [在线体验](./03_engineering_development/03.04_constexpr_and_device_functions.ipynb) | 🚧 开发中 |
| 3.5 工程化Vector算子开发 | [在线体验](./03_engineering_development/03.05_engineering_vector_operator.ipynb) | 🚧 开发中 |
| 3.6 章节实践 | [在线体验](./03_engineering_development/03.06_chapter_practice.ipynb) | 🚧 开发中 |

#### 第四章：Matmul高阶API算子开发

| Notebook | 链接 | 状态 |
|--|--|--|
| 4.1 章节介绍 | [在线体验](./04_matmul_high_level_api/04.01_chapter_intro.ipynb) | 🚧 开发中 |
| 4.2 Matmul高阶API介绍 | [在线体验](./04_matmul_high_level_api/04.02_matmul_api_introduction.ipynb) | 🚧 开发中 |
| 4.3 MIX模式Matmul开发 | [在线体验](./04_matmul_high_level_api/04.03_mix_mode_matmul_development.ipynb) | 🚧 开发中 |
| 4.4 Cube Only模式Matmul开发 | [在线体验](./04_matmul_high_level_api/04.04_cube_only_mode_matmul_development.ipynb) | 🚧 开发中 |
| 4.5 章节实践 | [在线体验](./04_matmul_high_level_api/04.05_chapter_practice.ipynb) | 🚧 开发中 |

### pyasc算子开发系列（高级）

#### 第五章：融合算子开发

| Notebook | 链接 | 状态 |
|--|--|--|
| 5.1 章节介绍 | [在线体验](./05_fused_operator_development/05.01_chapter_intro.ipynb) | 🚧 开发中 |
| 5.2 融合算子概念 | [在线体验](./05_fused_operator_development/05.02_fused_operator_concept.ipynb) | 🚧 开发中 |
| 5.3 Matmul+矢量融合算子开发 | [在线体验](./05_fused_operator_development/05.03_matmul_vector_fusion.ipynb) | 🚧 开发中 |
| 5.4 章节实践 | [在线体验](./05_fused_operator_development/05.04_chapter_practice.ipynb) | 🚧 开发中 |

#### 第六章：算子调试与性能调优

| Notebook | 链接 | 状态 |
|--|--|--|
| 6.1 章节介绍 | [在线体验](./06_debug_and_profiling/06.01_chapter_intro.ipynb) | 🚧 开发中 |
| 6.2 功能调试 | [在线体验](./06_debug_and_profiling/06.02_functional_debugging.ipynb) | 🚧 开发中 |
| 6.3 性能调优 | [在线体验](./06_debug_and_profiling/06.03_performance_profiling.ipynb) | 🚧 开发中 |
| 6.4 章节实践 | [在线体验](./06_debug_and_profiling/06.04_chapter_practice.ipynb) | 🚧 开发中 |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 适用硬件 | 说明 |
| --- | --- | --- |
| learning-hub notebook在线体验环境 | A2 / A3 | 点击各notebook的"在线体验"链接可直接打开运行 |
| CANNLab云开发环境 | 910B / 910C | 提供完整的NPU开发环境，支持上板运行 |

> **注意：** 如在本地环境离线体验，需自行安装配套的CANN软件和pyasc，具体请参考[环境准备](./01_pyasc_overview/01.03_environment_setup.ipynb)章节。

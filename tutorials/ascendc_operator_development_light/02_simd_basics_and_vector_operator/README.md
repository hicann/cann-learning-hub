# 第2章：Ascend C SIMD 编程基础与矢量算子开发（A2/A3）

## 章节概述

围绕“理解模型 → 写出算子 → 支持不同输入规模”展开。先建立 SIMD/SPMD 与接口体系认知，再用 Add 学习 TPipe/TQue 框架以及自主内存与同步管理，随后完成基础 Sigmoid、泛化 Tiling 与综合实践。

## 前置要求

- 完成第一章，能够编译运行 Hello World。
- 掌握基本的 C/C++ 内存操作，配置好 CANN 工具链与 NPU 运行环境。

## 学习目标

- 理解分层访存、SIMD/SPMD 和 Ascend C API 的分工。
- 用 TPipe/TQue 组织 CopyIn、Compute、CopyOut 流水任务。
- 自主规划与分配 UB 内存，显式管理流水同步。
- 完成核函数直调，并设计面向不同 shape、数据类型和尾块的泛化 Tiling。

## Notebook 与在线体验

| Notebook | 在线体验 |
| --- | --- |
| [02.01 章节介绍](02.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.01_chapter_intro.ipynb) |
| [02.02 SIMD编程概述与Ascend C编程接口概述](02.02_simd_programming_and_ascendc_api_overview.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.02_simd_programming_and_ascendc_api_overview.ipynb) |
| [02.03 TPipe/TQue编程框架与矢量算子开发](02.03_vector_operator_development_with_tpipe_tque.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.03_vector_operator_development_with_tpipe_tque.ipynb) |
| [02.04 自主分配内存与同步管理](02.04_manual_memory_allocation_and_sync_management.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.04_manual_memory_allocation_and_sync_management.ipynb) |
| [02.05 核函数实践](02.05_kernel_function_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.05_kernel_function_practice.ipynb) |
| [02.06 泛化Tiling设计](02.06_generalized_tiling_design.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.06_generalized_tiling_design.ipynb) |
| [02.07 Vector算子开发实践](02.07_vector_operator_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.07_vector_operator_practice.ipynb) |

## 文件与运行说明

Notebook 按文件名前缀顺序学习，`.01` 为章节介绍。请在本章目录启动或运行 Notebook。本章`images/` 保存课程图片，`src/` 保存配套源码，`answer/` 保存参考答案。运行生成的 `Sources/` 或 `output/` 与仓库配套资源分开保存。练习中的待补充代码需完成后再执行验证。

[返回教程总目录](../README.md)

# 矢量算子开发系列教程

本教程将带你学习面向昇腾NPU的矢量算子开发全流程，涵盖芯片架构理解、编程范式掌握、算子执行流程、SIMT编程实践和调试技巧等核心内容。

教程按章节划分，每个章节均包含以下内容：
- Notebooks：包含课程知识点与练习题，可在gitcode提供的轻量级notebook上运行，也可在本地jupyter lab环境中执行。
- Answer：包含课后练习和章节实践的答案。
- Images：包含教程配图，辅助理解架构和流程。

## 前置要求

为了充分掌握本教程内容，你应已具备以下能力：
- 具备 C/C++ 编程基础，包括变量声明、循环、条件语句、指针与数组操作。
- 了解深度学习基础概念，熟悉常见AI算法的基本原理。
- 具备基础的计算机体系结构知识，了解并行计算概念。
- 已完成 [Ascend C算子开发系列教程](../ascendc_operator_development/README.md)，掌握Ascend C编程范式和核函数基本概念。

## 环境要求

- **硬件**：昇腾 NPU 硬件（Atlas A2/A3/A5 系列）或云服务器。
- **软件**：已安装 CANN 开发环境（建议 CANN 8.0 以上版本）。
- **Python**：Python 3.8+ 环境。
- **工具**：Jupyter Lab / Notebook。

> **注意：**
> 本教程当前仅针对 Atlas A2 系列产品完成验证，A3/A5 系列产品验证正在进行中。不同型号芯片在计算能力和缓存容量上存在差异，具体说明见各章节内容。

## 课程章节规划

### 基础掌握（初级）
- **覆盖章节**：第1章
- **核心能力**：理解昇腾AI处理器架构，掌握SIMD/SIMT编程基础概念
- **典型任务**：能够阅读和理解基础算子代码，掌握核函数基本调用方式
- **考核方式**：选择题和判断题检验知识点掌握情况

### 加速库基础（中级）
- **覆盖章节**：第2章
- **核心能力**：理解算子执行流程，掌握aclnn单算子API调用和GE图模式
- **典型任务**：能够使用不同执行模式调用算子，理解Host和Device侧分工
- **考核方式**：流程理解验证和实践操作

### 昇腾算子开发（高级）
- **覆盖章节**：第3章
- **核心能力**：掌握SIMT算子开发实践，具备算子调试和性能分析能力
- **典型任务**：能够独立开发并调试SIMT算子，处理常见运行问题
- **考核方式**：算子编程实践和调试案例分析

## 章节目录

### 1. 基础掌握

| Notebook | 说明 |
|----------|------|
| [1.0 章节介绍](01_basic_mastery/01.00_chapter_intro.ipynb) | 章节概述、前置要求、学习目标 |
| [1.1 芯片架构简介](01_basic_mastery/01.01_chip_architecture.ipynb) | A2/A3/A5芯片架构特点 |
| [1.2 SIMD Membase编程基础](01_basic_mastery/01.02_simd_membase_basics.ipynb) | Membase编程范式与常见指令 |
| [1.3 SIMT编程基础](01_basic_mastery/01.03_simt_basics.ipynb) | SIMT线程架构与内存层级 |
| [1.4 章节实践](01_basic_mastery/01.04_chapter_practice.ipynb) | 章节综合实践题 |

### 2. 加速库基础

| Notebook | 说明 |
|----------|------|
| [2.0 章节介绍](02_acceleration_library_basics/02.00_chapter_intro.ipynb) | 章节概述、前置要求、学习目标 |
| [2.1 算子执行流程](02_acceleration_library_basics/02.01_operator_execution_flow.ipynb) | aclnn单算子API、GE图模式、Kernel直调 |
| [2.2 章节实践](02_acceleration_library_basics/02.02_chapter_practice.ipynb) | 章节综合实践题 |

### 3. 昇腾算子开发

| Notebook | 说明 |
|----------|------|
| [3.0 章节介绍](03_ascend_operator_development/03.00_chapter_intro.ipynb) | 章节概述、前置要求、学习目标 |
| [3.1 SIMT算子开发](03_ascend_operator_development/03.01_simt_operator_development.ipynb) | SIMT算子开发实战 |
| [3.2 算子调试](03_ascend_operator_development/03.02_operator_debugging.ipynb) | TTK工具使用与调试方法 |
| [3.3 章节实践](03_ascend_operator_development/03.03_chapter_practice.ipynb) | 章节综合实践题 |
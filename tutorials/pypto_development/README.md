# PyPTO 算子开发系列教程

本教程面向希望基于 PyPTO 学习昇腾 NPU 算子开发的开发者，围绕
CANN、芯片基础、PyPTO 编程范式与初级算子实践，提供一套由浅入深的
Notebook 教程。

课程内容覆盖从整体认知建立，到 Hello World 跑通、API 与计算图理解，
再到 elementwise、matmul、reduction、tiling / shape / slice /
transpose 等基础实践，帮助开发者建立使用 PyPTO 进行算子开发的系统化
学习路径。

教程按章节划分，每个章节包含以下内容：

- Notebooks：课程知识讲解、示例代码、练习或章节实践
- Answer：章节练习与实践的答案占位目录
- Images：章节配图占位目录
- Src：不便直接写入 Notebook 的源码占位目录

## 适用对象

- 希望系统学习 PyPTO 算子开发的初学者
- 已具备 CANN / 昇腾基础认知，希望进一步理解 PyPTO 编程模型的开发者
- 需要通过 Notebook 方式快速体验 PyPTO 开发流程的学习者

## 整体学习目标

完成本教程后，学习者应能够：

1. 说明 CANN、昇腾芯片与 PyPTO 之间的关系与分层职责
2. 理解 PyPTO 的 Host / Kernel 分工、MPMD 编程范式和计算图抽象
3. 跑通 PyPTO Hello World，并理解基本执行流程
4. 使用 PyPTO 完成基础 elementwise、matmul、reduction 算子表达
5. 理解 vec / cube tile shapes，以及 shape、slice、transpose 等基础操作
6. 具备继续学习更复杂 PyPTO 算子开发与优化的基础

## 环境要求

- 硬件：昇腾 NPU（建议 Atlas A2 / A3 系列）
- 软件：CANN 8.5.0 及以上版本
- Python：Python 3.10 及以上
- 工具：Jupyter Lab / Notebook

> 说明：
> 当前教程按 PyPTO Notebook 交互式学习方式组织，建议在支持 CANN 与
> torch_npu / pypto 的环境中体验。

## 章节规划

### 第一章：CANN、芯片以及 PyPTO 介绍

目录：`01_cann_chip_pypto_intro`

目标：

- 建立 CANN、昇腾芯片和 PyPTO 的整体认知
- 理解芯片硬件能力、软件栈与编程框架之间的关系

Notebook 列表：

| Notebook | 说明 |
| --- | --- |
| `01.01_chapter_intro.ipynb` | 章节介绍与学习目标 |
| `01.02_CANN.ipynb` | CANN 基础认知 |
| `01.03_chip.ipynb` | 昇腾芯片基础认知 |
| `01.04_PyPTO.ipynb` | PyPTO 框架基础介绍 |

### 第二章：PyPTO 算子开发基础知识

目录：`02_pypto_operator_development_basics`

目标：

- 理解一个最小 PyPTO 算子的定义、运行与验证流程
- 掌握 MPMD 编程范式、Host / Kernel 分工以及 API / 计算图基本概念

Notebook 列表：

| Notebook | 说明 |
| --- | --- |
| `02.01_chapter_intro.ipynb` | 章节介绍与学习目标 |
| `02.02_run_hello_world.ipynb` | PyPTO Hello World 跑通 |
| `02.03_programming_paradigm_mpmd.ipynb` | 编程范式与 MPMD |
| `02.04_api_and_compute_graph.ipynb` | API 与计算图基础 |

### 第三章：初级算子实践

目录：`03_beginner_operator_practice`

目标：

- 通过基础案例理解 PyPTO 中不同算子类型的表达方式
- 掌握常见基础操作在 Notebook 场景中的组织与验证方法

Notebook 列表：

| Notebook | 说明 |
| --- | --- |
| `03.01_chapter_intro.ipynb` | 章节介绍与学习目标 |
| `03.02_elementwise_vec_tile.ipynb` | elementwise 与 vec tile |
| `03.03_matmul_cube_tile.ipynb` | matmul 与 cube tile |
| `03.04_reduction_ops.ipynb` | reduction 算子 |
| `03.05_tiling_shape_slice_transpose.ipynb` | tiling / shape / slice / transpose |
| `03.06_chapter_practice.ipynb` | 章节实践 |

## 当前状态说明

当前版本为课程初版内容，已完成主要 Notebook 结构与章节内容草稿。
后续会继续补充：

- 章节实践答案
- 配图资源
- 需要独立存放的源码文件
- 在线体验与运行验证信息

## 反馈与建议

若在阅读或运行教程过程中遇到问题，欢迎通过仓库 Issue 反馈。

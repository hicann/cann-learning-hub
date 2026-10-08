# 第1章 昇腾开发基础

> 所属部分：第一部分 FA 前导课程 ｜ 建议学时：2 学时 ｜ 状态：✅ 已发布

## 章节概述

本章为全书打两块地基：**昇腾硬件架构**与 **Ascend C 编程模型**。先纵览昇腾 950 架构与关键芯片参数，建立"装备清单"式的硬件直觉；再拆解 Ascend C 算子的开发基础——算子数据流、内存结构、搬运指令、Cube/Vector 计算指令与同步指令。

学完本章，读者应能看懂一个 Ascend C Kernel 的数据流图，并说出图中每一段搬运与计算分别由哪类指令完成——这是后续 FA 算子开发与性能优化的共同语言。

## 章节内容

### 1. 昇腾架构基础

| 小节 | 内容 | Notebook | 状态 |
|--|--|--|--|
| 1.1 章节介绍 | 目标、前置知识、学习路线 | [01.01_chapter_intro.ipynb](01.01_chapter_intro.ipynb) | ✅ 已发布 |
| 1.2 昇腾 950 架构 | Chiplet UMA 整体架构、达芬奇架构演进（耦合→分离）、三代规格对比（910B/910D/950）、第三代 DaVinciCore（SIMD/SIMT 双模、CV 融合）、GM 访问通路（SMMU/MATA）、对 FlashAttention 的硬件优化 | [01.02_ascend_950_arch.ipynb](01.02_ascend_950_arch.ipynb) | ✅ 已发布 |

### 2. Ascend C 开发基础

| 小节 | 内容 | Notebook | 状态 |
|--|--|--|--|
| 2.1 算子数据流 | SPMD 多核模型（block_idx）、三条流（指令流/同步流/数据流）、搬运-计算-搬运范式、MatMul 全数据流、FA 计算与硬件单元的映射 | [01.03_operator_dataflow.ipynb](01.03_operator_dataflow.ipynb) | ✅ 已发布 |
| 2.2 内存结构 | GM/L2/L1/L0A/L0B/L0C/UB/BT/FP Buffer 存储层级、计算单元的私有领地、ND 与分形排布格式 | [01.04_memory_structure.ipynb](01.04_memory_structure.ipynb) | ✅ 已发布 |
| 2.3 搬运指令 | MTE2 / MTE1 / MTE3 / FIXPIPE、NDDMA、流水线并行与 Double Buffering | [01.05_data_move_instructions.ipynb](01.05_data_move_instructions.ipynb) | ✅ 已发布 |
| 2.4 Cube 计算指令 | 脉动阵列与 mmad、M/N/K、Cube 数据类型、Cube 编程范式 | [01.06_cube_instructions.ipynb](01.06_cube_instructions.ipynb) | ✅ 已发布 |
| 2.5 Vector 计算指令 | 向量指令三要素（Repeat/Mask/Stride）、API 分级（3 级 → 0 级）、常用向量指令与 FA 的对应、Vector 编程范式 | [01.07_vector_instructions.ipynb](01.07_vector_instructions.ipynb) | ✅ 已发布 |
| 2.6 同步指令 | PIPE 与事件（SetFlag/WaitFlag）、TQue 队列封装、底层时序、C API 同步 | [01.08_sync_instructions.ipynb](01.08_sync_instructions.ipynb) | ✅ 已发布 |
| — | 章节测试 | [01.09_chapter_test.ipynb](01.09_chapter_test.ipynb) | ✅ 已发布 |

## 前置知识

- C/C++ 编程基础
- 基本的计算机体系结构概念（缓存、内存层级）
- 不要求 NPU 环境：本章以原理讲解与示意图为主，可在任意环境阅读

## 学习建议

- **1.2 是"装备清单"**：先知道手里有什么武器，后面每一章都会回来查这张清单；
- **2.1 ~ 2.6 是"六块积木"**：任何 Ascend C Kernel 都由这六块积木拼装而成，FA 算子尤其如此；
- 每节末尾的课后练习用于自检，答案在 `answer/` 目录。

## 下一章

掌握昇腾硬件与 Ascend C 编程基础后，进入[第2章 FlashAttention 原理](../02_fa_principle/README.md)，理解 FA 算子的数学动机。

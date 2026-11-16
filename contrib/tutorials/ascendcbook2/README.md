[![Ascend C 异构并行程序设计（第二版）](https://raw.gitcode.com/HIT1920/AscendCBook/blobs/7209d62989a5e2c52f3829c1ac16a81dcc804f74/docs/images/readme/cover.png)](https://gitcode.com/HIT1920/AscendCBook.git)

# AscendCBook2：Ascend C 异构并行程序设计（第二版）

## 教材简介

《Ascend C 异构并行程序设计——昇腾算子编程指南（第二版）》面向昇腾算子开发者、AI 系统工程师与异构并行计算学习者，围绕算子开发全生命周期，系统讲解异构计算与软硬件原理、昇腾 AI 处理器、CANN 软件栈，以及 Ascend C 的 SIMD/SIMT 编程、编译运行、功能调试、性能优化和工程集成。

全书按照“算子编程原理、算子编程基础、算子编程案例、算子编程进阶”四个部分组织，共 20 章。教材通过 Softmax、RoPE、Batch MatMul、FlashAttention、Gather 等典型算子案例，贯通计算特征解析、基础实现、性能剖析、优化设计与效果验证，并进一步介绍混合编程、AI 框架集成、通信编程、大模型辅助开发、扩展库和 Python 开发生态。

本教材由哈尔滨工业大学团队编写，配套新版书稿、案例代码和课后习题，帮助学习者建立从硬件架构认知到高性能算子工程实践的完整知识体系。

## 适合人群

- 具备 C/C++ 基础，希望系统学习 Ascend C 算子开发的开发者；
- 从事昇腾 AI 处理器、CANN 软件栈或 AI 系统开发的工程师；
- 计算机、软件工程及相关专业中学习异构计算与并行编程的高校师生。

## 学习目标

- 理解异构计算、Host/Device 协作、算子分类与并行计算模型，建立昇腾 AI 处理器的软硬件协同认知；
- 掌握 CANN 软件栈、Ascend C SIMD/SIMT 编程模型、内存层级、同步机制及常用编程接口；
- 能够完成 Ascend C 算子的编译、加载、运行和功能调试，并掌握性能数据分析与优化方法；
- 能够围绕 Softmax、RoPE、Batch MatMul、FlashAttention、Gather 等典型算子进行并行算法设计、实现与性能验证；
- 了解 SIMD 与 SIMT 混合编程、AI 框架集成、通信编程、大模型辅助开发、扩展库和 Python 开发生态。

## 课程章节目录

### 第一部分：算子编程原理

| 章节 | 内容 |
| --- | --- |
| 第 1 章 | 异构计算与算子概述 |
| 第 2 章 | 昇腾 AI 处理器架构 |
| 第 3 章 | 昇腾异构计算架构 CANN |

### 第二部分：算子编程基础

| 章节 | 内容 |
| --- | --- |
| 第 4 章 | Ascend C 快速入门 |
| 第 5 章 | Ascend C SIMD 编程模型 |
| 第 6 章 | Ascend C SIMT 编程模型 |
| 第 7 章 | Ascend C 算子的编译与运行 |
| 第 8 章 | Ascend C 算子功能调试 |
| 第 9 章 | Ascend C 算子性能调试与优化 |

### 第三部分：算子编程案例

| 章节 | 内容 |
| --- | --- |
| 第 10 章 | SIMD 基于寄存器的矢量编程案例：Softmax 算子最佳实践 |
| 第 11 章 | SIMD Memory 矢量编程案例：RoPE 算子最佳实践 |
| 第 12 章 | SIMD 矩阵编程案例：Batch MatMul 算子最佳实践 |
| 第 13 章 | SIMD Cube-Vector（CV）融合编程案例：FlashAttention 算子最佳实践 |
| 第 14 章 | SIMT 编程案例：Gather 算子最佳实践 |

### 第四部分：算子编程进阶

| 章节 | 内容 |
| --- | --- |
| 第 15 章 | SIMT 与 SIMD 混合编程 |
| 第 16 章 | 算子 AI 框架集成与入图 |
| 第 17 章 | 通信编程 |
| 第 18 章 | 大模型辅助的算子开发 |
| 第 19 章 | Ascend C 扩展开发方式 |
| 第 20 章 | Ascend C 的典型 Python 开发生态 |

## 官方仓库

[AscendCBook 官方仓库](https://gitcode.com/HIT1920/AscendCBook.git)

# 第1章 卷积算子开发

## 章节概述
本章讲解 AscendC 卷积算子开发的核心流程，从卷积数学原理到前向 Conv2D 与反向 Conv3D Backprop Filter 的核函数实现，涵盖 NPU 片上存储层次、Tiling 参数计算、im2col（Load3D）、Mmad 与 Fixpipe 的协作关系。

## 章节内容

| Notebook | 内容 |
|--|--|
| 1.1 章节介绍 | 前置知识说明、学习目标与章节导航 |
| 1.2 背景介绍与卷积数学原理 | 卷积应用场景、CNN架构演进、梯度计算与链式法则、img2col变换、前向与反向卷积原理 |
| 1.3 Conv2D V2 核函数开发 | 算子分析（数学表达式、I/O规格、片上存储层次）、Tiling设计、Buffer管理、im2col+Mmad+Fixpipe流水线、Host端调用与编译运行 |
| 1.4 Conv3D Backprop Filter V2 核函数开发 | 权重梯度算子分析、Load2D(transpose)与Load3D(LEFT Fmatrix)、反向卷积核函数实现、Host端调用与编译运行 |
- **注意：**
- 本教程当前仅针对Ascend 950PR/Ascend 950DT系列产品进行验证，其他产品使用存在问题，欢迎开发者提出issue或PR进行共建
## 前置知识
- AscendC 基础（Ascend C 算子开发系列教程 第1-2章）
- 矩阵算子开发（Ascend C 算子开发系列教程 第4章）

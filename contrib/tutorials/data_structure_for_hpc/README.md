# Data Structure for High Performance Computing

课程名称：面向高性能计算的数据结构

本课程面向已经具备 C/C++ 基础和基本数据结构知识的学习者，围绕高性能计算场景中常见的数据组织、内存访问、并行计算和性能优化方法展开。

## 适用对象

- 了解 C/C++ 基础语法和二维数组存储方式的学习者。
- 希望理解矩阵、张量、分块、并行计算与硬件执行关系的开发者。
- 准备在 Ascend C 环境中学习高性能计算数据组织方法的学生。

## 整体学习目标

- 理解高性能计算中数据结构与内存访问模式的关系。
- 掌握矩阵、张量等数据结构在并行计算中的组织方式。
- 能够通过可执行实验分析数据分块、局部存储和多核划分对性能的影响。

## 已验证硬件与软件环境

- 已验证硬件：Ascend 910B，`SOC_VERSION=ascend910b1`。
- 已验证软件：CANN 8.3.RC1。
- 已验证运行环境：华为云 ModelArts Notebook / CANNLab 云开发环境。

## 目录结构

- `02_parallel_computing`：第 2 章，并行计算。
- `02_parallel_computing/02.02_extra_ascendc_static_tensor_matmul.ipynb`：课外实验，静态 Tensor 矩阵乘分块优化。

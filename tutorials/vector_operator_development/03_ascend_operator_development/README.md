# 第三章：Ascend 算子开发

本章深入讲解 Vector 算子的开发方法，以 FastGelu（y = x / (1 + exp(-1.702 * x))）为例，分别用 membase 和 regbase 两种模式实现完整的算子。同时介绍 SIMT 算子开发、aclnn 和 PTA 接口开发方式，以及算子调试和性能优化手段。

## 章节内容

| 章节 | 标题 | 状态 |
|------|------|------|
| 3.1 | [章节介绍](./03.01_chapter_intro.ipynb) | ✅ |
| 3.2 | [SIMD 算子开发](./03.02_simd_operator_development.ipynb) | ✅ |
| 3.3 | [SIMT 算子开发](./03.03_simt_operator_development.ipynb) | ✅ |
| 3.4 | [调用接口开发](./03.04_api_interface_development.ipynb) | ✅ |
| 3.5 | [算子调试](./03.05_operator_debugging.ipynb) | ✅ |
| 3.6 | [算子性能优化](./03.06_performance_optimization.ipynb) | ✅ |
| 3.7 | [章节实践](./03.07_chapter_practice.ipynb) | ✅ |

## 学习目标

完成本章后，你将能够：
- 使用 membase 和 regbase 两种模式独立开发 Vector 算子
- 掌握 SIMT 算子开发方法和实践技巧
- 了解 aclnn 两段式接口和 PTA 调用模式的开发方法
- 理解算子调试的常见方法和调试工具
- 应用 Double Buffer、UB 空间优化、Scalar 优化、数据对齐等常见性能优化手段

## 前置要求

- 完成第一、二章学习
- 已搭建开发环境


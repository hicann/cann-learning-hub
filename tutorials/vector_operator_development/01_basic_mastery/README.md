# 第一章：基础掌握

本章介绍昇腾 AI 处理器的 Vector 计算架构，从芯片架构到 SIMD 指令集，再到 regbase 编程模型的基础知识，为后续的算子开发打下坚实根基。

## 章节内容

| 编号 | 名称 | 说明 |
|------|------|------|
| 01.01 | [章节介绍](./01.01_chapter_intro.ipynb) | 章节概览与学习指引 |
| 01.02 | [芯片架构](./01.02_chip_architecture.ipynb) | A2/A3/A5 芯片架构简介 |
| 01.03 | [SIMD Membase 基础](./01.03_simd_membase_basics.ipynb) | SIMD Membase 编程基础与常见指令 |
| 01.04 | [regbase 编程基础](./01.04_simd_regbase.ipynb) | regbase 编程模型核心概念与 VF 指令 |
| 01.05 | [SIMT 基础](./01.05_simt_basics.ipynb) | SIMT 编程模型与线程架构 |
| 01.06 | [章节实践](./01.06_chapter_practice.ipynb) | FastGelu 算子实践练习 |

## 前置要求

- 具备 C++ 编程基础
- 了解深度学习框架基本概念
- 了解计算机体系结构基础知识
- 了解 Linux 系统基本操作

## 环境要求

- 昇腾 NPU 硬件（A5 系列）
- CANN 开发环境（建议 CANN 8.0 以上版本）
- Python 3.8+，Jupyter Lab / Notebook

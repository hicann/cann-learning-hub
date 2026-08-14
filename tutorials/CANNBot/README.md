# CANNBot 系列课程

CANNBot 系列课程围绕 CANNBot 算子开发展开，从入门到进阶，系统介绍如何使用 CANNBot 生成与优化 Ascend 算子。课程涵盖 Ascend C、PyPTO、TileLang-Ascend 等多种算子开发路径，并深入讲解 Vector 算子自动生成、单指令多线程算子、Harness 工程建设以及算子测试全流程，帮助开发者快速掌握 CANNBot 的核心能力。

> **学习建议**：本课程以 CANNBot 为工具介绍算子开发，重点在于"如何借助 CANNBot 高效完成算子开发与优化"，而非从零讲解算子开发原理。建议先对照下方《前置知识》补齐基础，再按 [课程内容](#课程内容) 中的前置依赖顺序学习。

## 前置知识

本课程面向已具备一定算子开发基础的开发者，建议具备以下前置知识：

- **Python 基础**：熟练使用 Python 编写与调试脚本
- **C/C++ 基础**：掌握指针、内存访问等基础概念，便于理解 Ascend C 算子代码
- **昇腾 NPU 基础**：了解昇腾 NPU 的 Cube / Vector / Scalar 三大计算单元及其分工
- **CANN 基础概念**：了解算子、Tiling、Kernel 等基本概念

若暂无上述基础，建议先学习以下课程补齐后再返回本课程：

- **CANN 与昇腾 NPU 基础**：参见 [Ascend C 算子开发系列教程（Kernel 直调）第 1 章](./../ascendc_operator_development_light/01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb)，涵盖人工智能与算子基础、CANN 架构与昇腾 NPU 原理
- **Ascend C 快速入门**：参见 [Ascend C 算子开发系列教程（Kernel 直调）](./../ascendc_operator_development_light/README.md)

## 快速开始：获取 CANNBot 仓库

CANNBot 的 Skills 与 Agents 统一托管在以下仓库（课件中的链接均指向该仓库）：

- 仓库地址：[cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)
- 安装说明：[CANNBot 安装指南](https://gitcode.com/cann/cannbot-skills/blob/master/docs/installation-guide.md)

## 课程内容

| 序号 | 主题 | 主要内容 | 前置 | 课件 |
| :---: | :--- | :--- | :--- | :--- |
| 1 | CANNBot 入门：从 0 到 1 生成你的第一个算子 | CANNBot 简介，从零开始生成第一个算子 | 无（具备[前置知识](#前置知识)即可） | [01_cannbot_start.pdf](./slides/01_cannbot_start.pdf) |
| 2 | CANNBot 开发进阶：Ascend C 算子开发实操 | 基于 Ascend C 的算子开发实操 | 第 1 课 | [02_ascend_c_operator.pdf](./slides/02_ascend_c_operator.pdf) |
| 3 | CANNBot 开发进阶：PyPTO 算子开发实操 | 基于 PyPTO 的算子开发实操 | 第 1 课（与第 2 课平行，任选其一） | [03_pypto_operator.pdf](./slides/03_pypto_operator.pdf) |
| 4 | CANNBot 开发进阶：TileLang-Ascend 算子开发实操 | 基于 TileLang-Ascend 的算子开发实操 | 第 1 课（与第 2 课平行，任选其一） | [04_tilelang_operator.pdf](./slides/04_tilelang_operator.pdf) |
| 5 | CANNBot 进阶开发：自动生成 Vector 算子之 RegBase | 自动生成 Vector 算子，RegBase 机制详解 | 第 2 课（需先理解 Ascend C 手动开发流程，才能理解自动生成原理） | [05_vector_regbase.pdf](./slides/05_vector_regbase.pdf) |
| 6 | CANNBot 进阶开发：Vector 算子之排序性能优化 | Vector 算子排序性能优化方法 | 第 5 课 | [06_vector_sort_opt.pdf](./slides/06_vector_sort_opt.pdf) |
| 7 | CANNBot 支持生成单指令多线程算子 | 单指令多线程算子的生成 | 第 2 课 | [07_multi_thread_operator.pdf](./slides/07_multi_thread_operator.pdf) |
| 8 | CANNBot 算子 Harness 工程建设 | 算子 Harness 工程的建设与实践 | 第 2～4 课（至少掌握一种开发路径） | [08_harness_engineering.pdf](./slides/08_harness_engineering.pdf) |
| 9 | CANNBot 算子测试全流程 | 算子测试的完整流程 | 第 8 课 | [09_operator_testing.pdf](./slides/09_operator_testing.pdf) |

## 相关课程

本课程覆盖 Ascend C、PyPTO、TileLang-Ascend 三种算子开发路径，但侧重点是"用 CANNBot 生成与优化算子"。若想在具体技术路线上深入系统学习，可结合以下课程：

| 学习方向 | 建议课程 | 与本课程的关系 |
| :--- | :--- | :--- |
| Ascend C 手动开发 | [Ascend C 算子开发系列教程（V2）](./../ascendc_operator_development_V2/README.md) | 本课程第 2、5、6、7 课涉及 Ascend C / Vector 算子，可在此系统学习 SIMD/SIMT 编程模型与手动开发流程 |
| Ascend C 快速入门 | [Ascend C 算子开发系列教程（Kernel 直调）](./../ascendc_operator_development_light/README.md) | 面向零基础的前置课程，覆盖 CANN / NPU 基础与 Ascend C 入门 |
| PyPTO 开发 | [PyPTO 算子开发系列教程](./../pypto_development/README.md) | 本课程第 3 课介绍 PyPTO 开发，可在此深入学习 PyPTO 编程范式与算子实践 |
| Vector 算子 | [Vector 算子开发课程](./../vector_operator_development/README.md) | 本课程第 5、6 课涉及 Vector 算子的 SIMD/SIMT 与 RegBase，可在此了解更详细的硬件架构与编程细节 |

> **学习路径建议**：零基础 → [Ascend C 算子开发系列教程（Kernel 直调）](./../ascendc_operator_development_light/README.md) → 本课程第 1～2 课 → 按需深入对应相关课程。

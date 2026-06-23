# 第2章 算子开发教程

## 章节概述

本章教程是昇腾 CANN（Compute Architecture for Neural Networks）算子开发系列教程的高级篇章，聚焦于大模型领域最前沿的 **MoE（Mixture of Experts，混合专家模型）** 架构下的通信算子开发。课程从 MoE 架构原理出发，逐步深入到 Dispatch/Combine 核心算子的实现细节，并通过量化 Dispatch 实战项目帮助开发者掌握高性能 MoE 通信算子的开发技能。学习 Win 区内存布局、Tiling 策略、流水并行等关键优化技术, 动手实现支持 MXFP8 动态量化的高性能 Dispatch 算子。
## 在线体验

| Notebook | Link | 状态 |
|--|--|--|
| 2.1 章节介绍 | [在线体验](./02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 2.2 MoE 架构概述 | [在线体验](./02.02_moe_architexture_overview.ipynb) | ✅ 已发布 |
| 2.3 并行策略 | [在线体验](./02.03_parallel_strategy.ipynb) | ✅ 已发布 |
| 2.4 Dispatch/Combine 算子 | [在线体验](./02.04_dispatch_combine_operator.ipynb) | ✅ 已发布 |
| 2.5 算子逻辑概述 | [在线体验](./02.05_operator_logic_overview.ipynb) | ✅ 已发布 |
| 2.6 核心流程拆解 | [在线体验](./02.06_dispatch_combine_core_flow.ipynb) | ✅ 已发布 |
| 2.7 Win 区内存布局 | [在线体验](./02.07_win_memory_layout.ipynb) | ✅ 已发布 |
| 2.8 Tiling 指南 | [在线体验](./02.08_tiling_guide.ipynb) | ✅ 已发布 |
| 2.9 Kernel 阶段指南 | [在线体验](./02.09_kernel_stage_guide.ipynb) | ✅ 已发布 |
| 2.10 量化 Dispatch 实践 | [在线体验](./02.10_quant_dispatch_practice.ipynb) | ✅ 已发布 |


## 运行环境与约束

- **硬件**：NPU多卡环境
- **芯片型号**：当前样例仅支持'dav-3510'。
- **软件环境**：需要已安装 Ascend CANN Tooltik。
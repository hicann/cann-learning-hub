# HCCL 集合通信系列课程

HCCL（Huawei Collective Communication Library）是昇腾集合通信库，为深度学习训练和推理场景提供高性能的多卡通信能力。本系列课程从 HCCL 基础概念入手，逐步深入到核心算子、算法实现以及性能优化工具，帮助开发者全面掌握 HCCL 的设计原理与应用实践。

课程涵盖 HCCL 简介与集合通信基础、基础算子和算法实现、北极星工具使用等核心内容，适合希望深入理解昇腾分布式通信机制的开发者学习。

## 课程内容

### 基础课程

| 序号 | 主题 | 主要内容 | 课件 |
| :---: | :--- | :--- | :--- |
| 1 | HCCL 简介与集合通信基础 | HCCL 架构概述、集合通信基础概念、昇腾通信机制介绍 | [01_hccl_intro_collective_comm.pdf](./slides/01_hccl_intro_collective_comm.pdf) |
| 2 | HCCL 基础算子和算法介绍 | HCCL 核心算子实现、常用通信算法（Ring、Tree、Recursive doubling 等）、性能分析 | [02_hccl_basic_operators_algorithms.pdf](./slides/02_hccl_basic_operators_algorithms.pdf) |
| 3 | HCCL 北极星工具介绍 | 北极星工具功能详解、通信性能分析、问题定位与优化实践 | [03_hccl_polaris_tool.pdf](./slides/03_hccl_polaris_tool.pdf) |

### 中级课程

| 序号 | 主题 | 主要内容 | 课件 |
| :---: | :--- | :--- | :--- |
| 4 | HCCL 软件架构和算子编程模型 | HCCL 软件整体架构、算子编程模型详解、HCCL算子常用编程API介绍 | [04_hccl_software_architecture.pdf](./slides/04_hccl_software_architecture.pdf) |
| 5 | HCCL 算子开发入门 - AICPU TS 引擎 | AICPU TS 引擎算子执行流程、算子开发实战演练 | [05_hccl_aicpu_ts_engine.pdf](./slides/05_hccl_aicpu_ts_engine.pdf) |
| 6 | HCCL 算子开发入门 - CCU 引擎 | CCU 简介、CCU 编程模型和 API介绍、算子开发实战演练 | [06_hccl_ccu_engine.pdf](./slides/06_hccl_ccu_engine.pdf) |

## 目录结构

```
hccl_development/
├── slides/                                # 课件目录
│   ├── 01_hccl_intro_collective_comm.pdf     # HCCL 简介与集合通信基础
│   ├── 02_hccl_basic_operators_algorithms.pdf # HCCL 基础算子和算法介绍
│   ├── 03_hccl_polaris_tool.pdf              # HCCL 北极星工具介绍
│   ├── 04_hccl_software_architecture.pdf     # HCCL 软件架构和算子编程模型
│   ├── 05_hccl_aicpu_ts_engine.pdf           # HCCL 算子开发入门 - AICPU TS 引擎
│   └── 06_hccl_ccu_engine.pdf                # HCCL 算子开发入门 - CCU 引擎
└── README.md                              
```


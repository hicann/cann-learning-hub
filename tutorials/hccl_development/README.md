# HCCL 集合通信系列课程

**HCCL（Huawei Collective Communication Library）是华为集合通信库**，是CANN的基础组件之一，依托昇腾芯片高效的通信引擎与总线/网络协议，为昇腾计算集群提供**高性能、高可靠、高易用集合通信解决方案**。本系列课程从HCCL基础概念入手，逐步深入到核心算子、算法实现以及模拟验证工具，帮助开发者全面掌握HCCL的设计原理与应用实践。

本课程分为初级课程和中级课程，其中1-3章节为初级课程，涵盖**HCCL简介与集合通信基础**、**HCCL基础算子和算法**、**北极星工具介绍** 等核心内容，适合刚刚接触分布式集群通信的开发者入门学习；4-7章节为中级课程，涵盖**HCCL软件架构和编程模型**、**基于AICPU引擎的HCCL算子开发**、**基于CCU引擎的HCCL算子开发**等内容，适合希望深入了解昇腾集合通信机制并尝试上手实践的开发者学习。

课程支持的硬件产品：Atlas 950 系列产品。
HCCL北极星工具的系统依赖请见：https://gitcode.com/cann/hcomm/tree/master/test/hccl_vm

## 课程内容

### 初级课程

| 序号 | 主题 | 主要内容 | 课件 |
| :---: | :--- | :--- | :--- |
| 1 | HCCL简介与集合通信基础 | HCCL架构概述、集合通信基础概念、昇腾通信机制介绍 | [01_hccl_intro_collective_comm.pdf](./slides/01_hccl_intro_collective_comm.pdf) |
| 2 | HCCL基础算子和算法介绍 | HCCL基础算子、通信算法 | [02_hccl_basic_operators_algorithms.pdf](./slides/02_hccl_basic_operators_algorithms.pdf) |
| 3 | HCCL北极星工具介绍 | 北极星工具功能详解、问题定位实践 | [03_hccl_polaris_tool.pdf](./slides/03_hccl_polaris_tool.pdf)<br>详细工具使用指导：https://gitcode.com/cann/hcomm/tree/master/test/hccl_vm |

### 中级课程

| 序号 | 主题 | 主要内容 | 课件 |
| :---: | :--- | :--- | :--- |
| 4 | HCCL 软件架构和算子编程模型 | HCCL整体软件架构、算子编程模型详解、HCCL算子常用编程API介绍 | [04_hccl_software_architecture.pdf](./slides/04_hccl_software_architecture.pdf) |
| 5 | HCCL 算子开发入门 - AICPU_TS 引擎 | AICPU_TS引擎算子执行流程、算子开发实战演练 | [05_hccl_aicpu_ts_engine.pdf](./slides/05_hccl_aicpu_ts_engine.pdf) |
| 6 | HCCL 算子开发入门 - CCU 引擎 | CCU简介、CCU编程模型和API介绍、算子开发实战演练 | [06_hccl_ccu_engine.pdf](./slides/06_hccl_ccu_engine.pdf) |

## 目录结构

```
hccl_development/
├── slides/                                    # 课件目录
│   ├── 01_hccl_intro_collective_comm.pdf      # HCCL简介与集合通信基础
│   ├── 02_hccl_basic_operators_algorithms.pdf # HCCL基础算子和算法介绍
│   ├── 03_hccl_polaris_tool.pdf               # HCCL北极星工具介绍
│   ├── 04_hccl_software_architecture.pdf      # HCCL软件架构和算子编程模型
│   ├── 05_hccl_aicpu_ts_engine.pdf            # HCCL算子开发入门-AICPU_TS引擎
│   └── 06_hccl_ccu_engine.pdf                 # HCCL算子开发入门-CCU引擎
└── README.md                              
```


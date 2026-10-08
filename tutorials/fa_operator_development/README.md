# FlashAttention 算子开发课程

## 课程简介

本课程面向高校学生与算子开发初学者，以「**基础 → 实战 → 优化**」三部分、十章的渐进式结构，讲解 FlashAttention（FA）算子从原理到昇腾 Ascend C 开发、再到性能优化高阶实战的完整链路。

课程以昇腾开源算子 [FusedInferAttentionScore](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score) 为工程蓝本。该算子适配增量与全量推理场景，支持全量计算（Prompt 场景）与增量计算（Decode 场景），覆盖 GQA/MQA、KV Cache、因果掩码、量化（MXFP8 等）等大模型推理的关键特性，运行于 Atlas A2 / A3 / Ascend 950PR 等系列产品。

## 课程大纲

### 第一部分：FA 前导课程

打好硬件、算法与代码结构三块地基。

| 章 | 标题 | 内容 | 目录 | 状态 |
|--|--|--|--|--|
| 第1章 | 昇腾开发基础 | 昇腾架构基础（950 架构与芯片参数）、Ascend C 开发基础（算子数据流、内存结构、搬运指令 MTE2/MTE1/MTE3/FIXPIPE、Cube/Vector 计算指令、同步指令） | [01_ascend_dev_basics](01_ascend_dev_basics/README.md) | ✅ 已发布 |
| 第2章 | FlashAttention 原理 | Transformer 简介、Attention 原理、标准 Attention 瓶颈、FlashAttention 原理（softmax 推导、FA V1 / V2）、FA 的演进（MLA、GQA、SFA 等） | [02_fa_principle](02_fa_principle/README.md) | ✅ 核心小节已发布，扩充中 |
| 第3章 | FA 算子代码结构 | 以基础极简 FA 代码为例：算子调用通路、Tiling、Kernel | [03_fa_code_architecture](03_fa_code_architecture/README.md) | ✅ 已发布 |

### 第二部分：FA 算子开发实战

先跑通官方算子，再从 0 到 1 亲手构建。

| 章 | 标题 | 内容 | 目录 | 状态 |
|--|--|--|--|--|
| 第4章 | FA 算子运行 | 环境搭建、FA 算子工程介绍、FA 算子接口介绍（FA 推理）、FA 算子运行（基于 CANNSim 仿真） | [04_fa_operator_running](04_fa_operator_running/README.md) | ✅ 已发布 |
| 第5章 | 非量化 FA 开发 | 基础极简 FA、BN1 简单分核、buffer 复用（开 pingpong）、preload 流水、mask、QS 和 KVS 带尾块、负载均衡 | [05_flash_attn_non_quantized](05_flash_attn_non_quantized/README.md) | ✅ 已发布 |
| 第6章 | 全量化 FA 开发 | 数据格式（FP8/HiF8）、量化原理、量化方式（各种量化方式 + 随路 MXFP8/HiF8）、FA 支持 MXFP8 量化方案、FA 支持 HiP8 量化方案 | [06_flash_attn_quantized](06_flash_attn_quantized/README.md) | ✅ 已发布 |

### 第三部分：FA 算子性能优化实战

从建模分析到逐引擎优化，再到高阶负载均衡。

| 章 | 标题 | 内容 | 目录 | 状态 |
|--|--|--|--|--|
| 第7章 | 性能优化基础 | 性能建模、Roofline 模型、访存性能、常用性能优化步骤（均衡性、CV 流水/切块、核内流水） | [07_performance_basics](07_performance_basics/README.md) | ✅ 已发布 |
| 第8章 | 性能优化进阶 | Cube 性能（K 切分、最优 K 切分确定）、Vector 性能（VF 优化、DN、VF 建模）、Scalar 优化 | [08_performance_advanced](08_performance_advanced/README.md) | ✅ 已发布 |
| 第9章 | 性能优化高阶 | 负载均衡：带 section + GS1 不合轴 + 切 S2、带 section + GS1 合轴 + 切 S2 | [09_performance_high_level](09_performance_high_level/README.md) | ✅ 已发布 |
| 第10章 | 性能优化指南 | 汇总全书性能优化方法，形成完整优化路径与排查指南 | [10_performance_guide](10_performance_guide/README.md) | ✅ 已发布 |

## 课程主线

1. **基础**（第1~3章）：懂硬件（昇腾架构与 Ascend C 编程模型）、懂算法（FA 数学原理）、懂架构（算子代码的三层骨架）；
2. **实战**（第4~6章）：先在 950 上跑通官方 FA 算子获得直观体验，再以渐进式路线（极简 → 分核 → 双缓冲 → 流水 → mask → 尾块 → 负载均衡）从 0 到 1 构建非量化 FA，进而攻克 MXFP8 / HiP8 全量化方案；
3. **优化**（第7~10章）：以性能建模与 Roofline 为理论起点，逐引擎优化 Cube / Vector / Scalar，再攻克核间负载均衡，最终以性能优化指南汇总全书方法，逼近硬件理论性能。

## 前置知识

- Python 编程基础与 numpy 的基本使用（数组、矩阵乘法 `@`）
- 基本线性代数：矩阵乘法、转置
- 第1章起需要 C/C++ 基础
- 不要求深度学习框架经验，第2章不要求 NPU 环境

## 环境要求

- **第2章**：任意 Python 3.8+ 环境，只需 numpy 与 Jupyter。开始学习前请先安装：

  ```bash
  python -m pip install numpy ipykernel --user
  python -m ipykernel install --user --name python3 --display-name "Python 3 (fa-course)"
  ```

  详细步骤与离线安装方法见[第2章 2.1 环境准备](02_fa_principle/01.01_chapter_intro.ipynb)。

- **第4章起**：涉及 FA 算子运行与 Ascend C 开发，需要昇腾 NPU 硬件（Ascend 950 或 Atlas 系列）、云服务器或 CANNSim 仿真环境，并按 [CANN 下载页面](https://www.hiascend.com/cann/download) 完成开发环境部署。

## 目录结构

```
fa_operator_development/
├── README.md                       # 本文件：课程大纲
├── 01_ascend_dev_basics/           # 第1章：昇腾开发基础（已发布）
├── 02_fa_principle/                # 第2章：FlashAttention 原理（已发布，扩充中）
├── 03_fa_code_architecture/        # 第3章：FA 算子代码结构（已发布）
├── 04_fa_operator_running/         # 第4章：FA 算子运行（已发布）
├── 05_flash_attn_non_quantized/    # 第5章：非量化 FA 开发（已发布）
├── 06_flash_attn_quantized/        # 第6章：全量化 FA 开发（已发布）
├── 07_performance_basics/          # 第7章：性能优化基础（已发布）
├── 08_performance_advanced/        # 第8章：性能优化进阶（已发布）
├── 09_performance_high_level/     # 第9章：性能优化高阶（已发布）
└── 10_performance_guide/          # 第10章：性能优化指南（已发布）
```

## 参考资料与开源代码

- FlashAttention 论文：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., 2022)；*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (Dao, 2023)
- 昇腾开源算子仓库 ops-transformer（本课程蓝本）：[gitcode.com/cann/ops-transformer](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)，代码路径 `attention/fused_infer_attention_score`
- 配套课程：[Ascend C 算子开发系列教程](../ascendc_operator_development/README.md)

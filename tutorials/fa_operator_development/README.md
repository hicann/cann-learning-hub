# FlashAttention 算子开发课程

## 课程简介

本课程以昇腾开源算子 [FusedInferAttentionScore](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score) 为蓝本，讲解 FlashAttention 算子的数学原理与核心实现思想。

FusedInferAttentionScore 是适配增量与全量推理场景的 FlashAttention 算子，支持全量计算（Prompt 场景）与增量计算（Decode 场景），覆盖 GQA/MQA、KV Cache、因果掩码、量化等大模型推理的关键特性，运行于 Atlas A2 / A3 / Ascend 950PR 等系列产品。

课程采用「原理 → 数值实验 → 算子开发」的渐进式结构：第1章通过纯 Python 数值实验在无 NPU 环境下讲清原理，第2章起进入昇腾 Ascend C 算子开发。

## 课程大纲

| 章节 | 内容 | 状态 |
|--|--|--|
| 第1章 FlashAttention 算子原理 | Attention 机制基础（QKV / 多头 / GQA / KV Cache）、标准实现的访存瓶颈（存储层次 / 算术强度 / Roofline）、FlashAttention 核心原理（Safe Softmax / Online Softmax 三步递推 / 分块计算 / 因果掩码块跳过）、Online Softmax 数值实验 | ✅ 已发布 |
| 第2章 非量化 Flash Attention 算子开发 | 非量化 FA 的概念与量化版本对比、算子规格（q/k/v/attn_out 的 dtype/shape/format）、QK^T → 在线 Softmax → PV 计算流程、Ascend C 实现思路（Cube/Vector 协同、双 Matmul、Tiling 切分） | 🚧 建设中 |

## 前置知识

- Python 基础与基本线性代数（矩阵乘法），无需 NPU 环境。

## 环境要求

- **第1章**：任意 Python 3.8+ 环境，只需 numpy 与 Jupyter。开始学习前请先安装：

  ```bash
  python -m pip install numpy ipykernel --user
  python -m ipykernel install --user --name python3 --display-name "Python 3 (fa-course)"
  ```

  详细步骤与离线安装方法见[第1章 1.1 环境准备](01_fa_principle/01.01_chapter_intro.ipynb)。

- **第2章起**：涉及 Ascend C 算子开发，需要昇腾 NPU 硬件（或云服务器/仿真环境），并按 [CANN 下载页面](https://www.hiascend.com/cann/download) 完成开发环境部署。

## 目录结构

```
fa_operator_development/
├── README.md                          # 本文件：课程大纲
├── 01_fa_principle/                   # 第1章：FlashAttention 算子原理
└── 02_flash_attn_non_quantized/       # 第2章：非量化 Flash Attention 算子开发
    ├── README.md                      # 章节导航
    ├── 01.01_chapter_intro.ipynb      # 章节介绍
    ├── 01.02_attention_basics.ipynb   # Attention 机制基础
    ├── 01.03_attention_bottleneck.ipynb  # 标准实现的访存瓶颈
    ├── 01.04_fa_principle.ipynb       # FlashAttention 核心原理
    ├── 01.05_online_softmax_experiment.ipynb  # Online Softmax 数值实验
    ├── images/                        # 示意图
    └── answer/                        # 测验答案
```

## 参考资料与开源代码

- FlashAttention 论文：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., 2022)；*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (Dao, 2023)
- 昇腾开源算子仓库 ops-transformer（本课程参考）：[gitcode.com/cann/ops-transformer](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)，代码路径 `attention/fused_infer_attention_score`
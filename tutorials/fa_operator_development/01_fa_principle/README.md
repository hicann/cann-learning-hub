# 第1章 FlashAttention 算子原理

## 章节概述

本章讲解 FlashAttention 的数学原理与设计动机：从 Self-Attention 的计算公式与图解出发，介绍多头注意力、GQA/MQA 与 KV Cache 等大模型推理关键概念；随后从加速器存储层次、算术强度（Roofline 模型）与访存量三个角度，定量分析标准 Attention 实现「三步走」的 HBM 访存瓶颈；最后系统推导 FlashAttention 的核心原理——Safe Softmax 的数值稳定性、Online Softmax 的三步递推、外层 Q 块 / 内层 KV 块的分块计算（Tiling）流程与因果掩码的整块跳过，并通过纯 Python 数值实验验证 Online Softmax 与标准 Softmax 的精确等价性。

本章全部数值实验仅依赖 numpy，无需 NPU 环境，零基础读者可完整学习。

## 章节内容

| Notebook | 内容 |
|--|--|
| 1.1 章节介绍 | 前置知识说明、学习目标与章节导航 |
| 1.2 Attention 机制基础 | Attention 的直观含义与图解、QKV 变换、打分-归一化-加权求和三步计算、多头注意力、GQA/MQA 与 KV Cache |
| 1.3 标准 Attention 的访存瓶颈 | 加速器存储层次、算术强度与 Roofline 模型、标准实现三步数据流分析、访存量定量计算与数值实验 |
| 1.4 FlashAttention 核心原理 | Safe Softmax、Online Softmax 三步递推推导、分块计算（Tiling）伪代码、因果掩码整块跳过、复杂度对比与 NPU 算子形态映射 |
| 1.5 Online Softmax 数值实验 | 朴素/安全/在线三种 Softmax 实现、正确性对拍、数值稳定性实验、分块大小与误差关系 |

## 前置知识

- Python 编程基础与 numpy 的基本使用（数组、矩阵乘法 `@`）。
- 基本线性代数：矩阵乘法、转置。
- 不要求了解深度学习框架，不要求 NPU 环境。

## 环境准备

本章数值实验只需 Python + numpy + Jupyter，无需 NPU。开始学习前请先安装依赖并注册内核：

```bash
python -m pip install numpy ipykernel --user
python -m ipykernel install --user --name python3 --display-name "Python 3 (fa-course)"
```

在 IDE 中打开 notebook 并选择该内核，运行 `import numpy as np; print(np.__version__)` 输出版本号即说明就绪。详见 [1.1 章节介绍](01.01_chapter_intro.ipynb) 中的「环境准备」一节。

## 学习建议

- **1.2 与 1.3 是动机链条**：先理解「Attention 在算什么」，再理解「它为什么慢」——两条线索汇合处就是 FlashAttention 的设计出发点。
- **1.4 是本章的核心**：NPU Kernel 中的每一段代码，都是 1.4 节数学公式的工程化落地。建议对照示意图反复推导 Online Softmax 的三步递推，直到可以白板复现。
- **1.5 动手实验**：亲手运行并修改分块大小，观察数值误差，建立对「精确算法」的直观信心。

## 下一章

掌握 FlashAttention 的数学原理后，进入[第2章 非量化 Flash Attention 算子开发](../02_flash_attn_non_quantized/README.md)，开始昇腾 Ascend C 算子实现之旅。

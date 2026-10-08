# 第2章 FlashAttention 原理

> 所属部分：第一部分 FA 前导课程 ｜ 状态：✅ 已完成

## 章节概述

本章讲解 FlashAttention 的数学原理与设计动机：先从 Transformer 整体结构出发，看清 Attention 是唯一随序列长度平方增长的部件、值得专门优化；随后从 Self-Attention 的计算公式与图解出发，介绍多头注意力、GQA/MQA 与 KV Cache 等大模型推理关键概念；接着从加速器存储层次、算术强度（Roofline 模型）与访存量三个角度，定量分析标准 Attention 实现「三步走」的 HBM 访存瓶颈；然后系统推导 FlashAttention 的核心原理——Safe Softmax 的数值稳定性、Online Softmax 的三步递推、外层 Q 块 / 内层 KV 块的分块计算（Tiling）流程与因果掩码的整块跳过，并通过纯 Python 数值实验验证 Online Softmax 与标准 Softmax 的精确等价性；再按历史顺序完整还原 FlashAttention V1 的原始算法与它的两大遗憾、V2 的三刀手术（循环交换、并行扩维、分工重构）；最后纵览 FA 的演进族谱——GQA、MLA、SFA 如何分别压下 KV Cache 与计算量，以及它们在昇腾 `FusedInferAttentionScore` 算子接口上的映射。

本章全部数值实验仅依赖 numpy，无需 NPU 环境，零基础读者可完整学习。

## 章节内容

| Notebook | 内容 |
|--|--|
| 2.1 章节介绍 | 前置知识说明、学习目标与章节导航 |
| 2.2 Transformer 简介 | RNN 痛点与自注意力、原版 Encoder-Decoder 与现代 Decoder-Only 架构、参数量与计算量账单、Prefill 与 Decode 两种推理节奏、numpy 最小 Decoder 实例 |
| 2.3 Attention 机制基础 | Attention 的直观含义与图解、QKV 变换、打分-归一化-加权求和三步计算、多头注意力、GQA/MQA 与 KV Cache（含逐步推理推导与 PagedAttention 分页管理） |
| 2.4 标准 Attention 的访存瓶颈 | 加速器存储层次、算术强度与 Roofline 模型、标准实现三步数据流分析、访存量定量计算与数值实验 |
| 2.5 FlashAttention 核心原理 | Safe Softmax、Online Softmax 三步递推推导、分块计算（Tiling）伪代码、因果掩码整块跳过、训练场景的反向重计算与稀疏/低秩近似路线对比、复杂度对比与 NPU 算子形态映射（四拍循环 + 存储通路） |
| 2.6 Online Softmax 数值实验 | 朴素/安全/在线三种 Softmax 实现、正确性对拍、数值稳定性实验、分块大小与误差关系 |
| 2.7 FlashAttention V1 | FA1 原始算法（外层 KV 块、内层 Q 块、每步归一化）、IO 直觉与搬运账单（Q/O/l/m 随 Tc 轮往返）、Cube/Vector 逐步执行映射、两大遗憾（大量 non-matmul 运算、并行度仅 B×H）、FA1/V2 风格 numpy 对拍与逐元素运算统计 |
| 2.8 FlashAttention V2 | 三刀手术：循环交换与归一化后移（片上初始化、logsumexp 合一写出）、B×H×Q 块并行扩维、Warp 按 Q 行切分零通信；FA3 的 TMA/WGMMA/FP8；成绩单（25%~40% → 50%~73%）与昇腾映射 |
| 2.9 FA 的演进：MLA、GQA、SFA | KV Cache 经济账、MHA/MQA/GQA 拓扑与显存节省、MLA 低秩压缩与 RoPE 解耦与权重吸收、SFA 双阶段稀疏架构与昇腾工程挑战 |

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

在 IDE 中打开 notebook 并选择该内核，运行 `import numpy as np; print(np.__version__)` 输出版本号即说明就绪。详见 [2.1 章节介绍](02.01_chapter_intro.ipynb) 中的「环境准备」一节。

## 学习建议

- **2.2 ~ 2.4 是动机链条**：先看清「Attention 在 Transformer 里的位置」，再理解「它在算什么」，最后理解「它为什么慢」——三条线索汇合处就是 FlashAttention 的设计出发点。
- **2.5 是本章的核心**：NPU Kernel 中的每一段代码，都是 2.5 节数学公式的工程化落地。建议对照示意图反复推导 Online Softmax 的三步递推，直到可以白板复现。
- **2.6 动手实验**：亲手运行并修改分块大小，观察数值误差，建立对「精确算法」的直观信心。
- **2.7 与 2.8 是一条手术线**：先看清 FA1 的两个遗憾，再逐刀核对 V2 的三处修改——每一刀都直接对应后续章节的优化主题（分核策略、Cube/Vector 流水、VF 优化）。
- **2.9 纵览即可**：记住族谱表与「GQA/MLA 压缓存、SFA 压计算」的主线，具体算子接口在第 4 章展开。

## 下一章

掌握 FlashAttention 的数学原理后，进入[第3章 FA 算子代码结构](../03_fa_code_architecture/README.md)，认识 FA 算子的工程骨架。

# 第5章 非量化 FA 开发

> 所属部分：第二部分 FA 算子开发实战 ｜ 状态：✅ 已发布

## 章节概述

本章聚焦**非量化** Flash Attention（闪存注意力）算子，即输入 Q、K、V 均为 FP16 或 BF16 的高精度浮点类型。本章首先介绍非量化 Flash Attention 的概念、算子规格与计算流程，帮助开发者建立对非量化注意力算子的整体认识，为后续具体实现章节打好基础。

本章是课程的核心实战章节：以渐进式的方式从最简实现一步步逼近工业级 Kernel——从基础极简 FA 出发，逐步引入分核、buffer 复用、流水、mask、尾块与负载均衡，**每一步都只引入一个新概念**，每一步都对应一次可独立调测、可对拍验证的代码迭代。

## 开发路线

| 小节 | 引入的新概念 | 状态 |
|--|--|--|
| 5.1 基础极简 FA | 四拍循环 + 单缓冲串行单核 | ✅ 已发布 |
| 5.2 BN1 简单分核 | 核间切块第一原则：无数据依赖 | ✅ 已发布 |
| 5.3 buffer 复用（pingpong） | Q 常驻 + KV 乒乓，搬运藏进计算 | ✅ 已发布 |
| 5.4 preload 流水 | C2 滞后发射，两 Cube 掩一长 V1 | ✅ 已发布 |
| 5.5 mask | 块级跳过 + 行级掩码（sparsemode3） | ✅ 已发布 |
| 5.6 QS 和 KVS 带尾块 | padding 的「必须算的浪费」vs 尾块的「不算的变长」 | ✅ 已发布 |
| 5.7 负载均衡 | BN1S1 细粒度分核，任务数/任务量双均衡 | ✅ 已发布 |

## 在线体验

| Notebook | 内容 | Link |
|--|--|--|
| 5.1 章节导学 | 七阶段路线图与学习方法 | [05.01_chapter_intro.ipynb](05.01_chapter_intro.ipynb) |
| 5.2 非量化 FA 介绍 | 算子规格、计算流程与规格约束 | [05.02_flash_attention_introduction.ipynb](05.02_flash_attention_introduction.ipynb) |
| 5.3 基础极简 FA | 阶段一：单核串行最小可运行 FA | [05.03_basic_fa.ipynb](05.03_basic_fa.ipynb) |
| 5.4 BN1 简单分核 | 阶段二：按 B·N 无依赖切分 | [05.04_bn1_split.ipynb](05.04_bn1_split.ipynb) |
| 5.5 buffer 复用 | 阶段三：Q 常驻 + KV pingpong | [05.05_pingpong.ipynb](05.05_pingpong.ipynb) |
| 5.6 preload 流水 | 阶段四：消除 Cube 断流 | [05.06_preload.ipynb](05.06_preload.ipynb) |
| 5.7 mask | 阶段五：sparsemode3 块级跳过 | [05.07_mask.ipynb](05.07_mask.ipynb) |
| 5.8 尾块 | 阶段六：QS/KVS 带尾块支持 | [05.08_tail_block.ipynb](05.08_tail_block.ipynb) |
| 5.9 负载均衡 | 阶段七：BN1S1 细粒度分核 | [05.09_load_balance.ipynb](05.09_load_balance.ipynb) |
| 5.10 章节测验 | 综合测验 | [05.10_chapter_test.ipynb](05.10_chapter_test.ipynb) |

## 配图

- [preload_pipeline.svg](images/preload_pipeline.svg)：串行 / NBuffer 乒乓 / Preload 三种编排的时序对比

## 前置知识

- [第2章 FlashAttention 原理](../02_fa_principle/README.md)
- [第3章 FA 算子代码结构](../03_fa_code_architecture/README.md)
- [第4章 FA 算子运行](../04_fa_operator_running/README.md)

## 下一章

完成非量化 FA 开发后，进入[第6章 全量化 FA 开发](../06_flash_attn_quantized/README.md)。

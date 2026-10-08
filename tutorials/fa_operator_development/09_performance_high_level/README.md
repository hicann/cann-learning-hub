# 第9章 性能优化高阶

> 所属部分：第三部分 FA 算子性能优化实战 ｜ 状态：✅ 已发布

## 章节概述

负载均衡是 FA 算子性能优化的最后一公里：当各引擎的效率都已榨干，核间任务分配的不均衡往往成为新的瓶颈。本章从「为什么名义 shape 均匀 ≠ 负载均匀」出发，讲解带 section 场景下的两类负载均衡策略——**GS1 不合轴 + 切 S2**（嵌套循环、行粒度贪心装箱）与 **GS1 合轴 + 切 S2**（展平一维轴、块粒度装箱），并补上切 S2 后 softmax 数值正确性的最后一块拼图——merge 归并。

学完本章，读者应能按有效基本块数（而非名义 shape）完成带 section 的分核设计，掌握两类 GS1 策略的选型判据，理解 flash decoding 的 (m, l, acc) 归并数学。

## 章节内容

| 小节 | 内容 | 状态 |
|--|--|--|
| [9.1 章节介绍](09.01_chapter_intro.ipynb) | 负载均衡为什么是最后一公里、section/GS1/切 S2 概念 | ✅ |
| [9.2 均衡问题的三层来源](09.02_imbalance_sources.ipynb) | shape 锯齿、mask 无效块、section 变长；有效块数计量 | ✅ |
| [9.3 策略一：GS1 不合轴 + 切 S2](09.03_gs1_unmerged.ipynb) | 贪心分核流程、封核判据、六数组、flash decoding | ✅ |
| [9.4 策略二：GS1 合轴 + 切 S2](09.04_gs1_merged.ipynb) | 合轴的收益与代价、选型判据、三个工程坑 | ✅ |
| [9.5 切 S2 的 softmax 归并](09.05_s2_merge.ipynb) | (m, l, acc) 统计量、归并公式、merge kernel、成本账 | ✅ |
| [9.6 章节测验](09.06_chapter_test.ipynb) | 综合测验（10 题） | ✅ |

## 配图

- [images/gs1_axis.svg](images/gs1_axis.svg)：GS1 合轴 vs 不合轴的封核边界对比
- [images/s2_split_merge.svg](images/s2_split_merge.svg)：切 S2 的分核与 merge 归并流程

## 前置知识

- [第7章 性能优化基础](../07_performance_basics/README.md)（尤其 7.4 均衡性：CoreWeight、贪心分核、有效块统计）
- [第5章 非量化 FA 开发](../05_flash_attn_non_quantized/README.md)：S2 外层循环与 softmax update
- [第2章 FlashAttention 原理](../02_fa_principle/README.md)：在线 softmax

## 下一章

攻克负载均衡后，进入[第10章 性能优化指南](../10_performance_guide/README.md)，纵览全书的性能优化方法。

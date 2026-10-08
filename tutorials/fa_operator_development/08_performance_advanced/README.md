# 第8章 性能优化进阶

> 所属部分：第三部分 FA 算子性能优化实战 ｜ 状态：✅ 已发布

## 章节概述

[第7章](../07_performance_basics/README.md)给出了「均衡性 → CV 流水与切块 → 核内流水」的通用优化路径，本章按执行引擎逐个击破：**Cube 性能**聚焦 K 切分——不同切分方式对带宽的影响，以及拿到一个 shape 后如何确定最优 K 切分；**Vector 性能**覆盖 VF 的指令级优化、DN 直进通路与 VF 静态建模；**Scalar 性能**讲解标量上提、增量化、分支消除与同步精简。

学完本章，读者应能对给定 shape 求解最优 K 切分方案，对 VF 代码做指令级优化并静态估算耗时，掌握 DN vs ND 的选型判据，以及一套 kernel 侧 Scalar 体检清单。

## 章节内容

| 小节 | 内容 | 状态 |
|--|--|--|
| [8.1 章节介绍](08.01_chapter_intro.ipynb) | 三引擎瓶颈画像、本章路线图 | ✅ |
| [8.2 Cube 性能（上）：K 切分原理](08.02_cube_k_split.ipynb) | K 累加轴、L0 容量约束、切分对带宽的影响、部分和 | ✅ |
| [8.3 Cube 性能（下）：最优 K 切分](08.03_cube_best_k.ipynb) | 四步选优流程、容量方程、切分方案建模与选优、尾块处理 | ✅ |
| [8.4 Vector 性能（上）：VF 优化](08.04_vector_vf_opt.ipynb) | vLoop 切分、指令精简五式、MaskReg 尾块、指令编排 | ✅ |
| [8.5 Vector 性能（下）：DN 与 VF 建模](08.05_vector_dn_modeling.ipynb) | DN 通路、DN vs ND 选型、VF 耗时静态建模 | ✅ |
| [8.6 Scalar 优化](08.06_scalar_opt.ipynb) | 标量上提、循环增量化、分支消除、同步精简 | ✅ |
| [8.7 章节测验](08.07_chapter_test.ipynb) | 综合测验（10 题） | ✅ |

## 配图

- [images/k_split.svg](images/k_split.svg)：K 切分与部分和、L0 占用与开销账单
- [images/dn_vs_nd.svg](images/dn_vs_nd.svg)：DN 与 ND 两条 GM→L0 数据通路对比

## 前置知识

- [第7章 性能优化基础](../07_performance_basics/README.md)：Roofline 建模、CV 流水、RegBase 编程骨架
- [第1章 昇腾开发基础](../01_ascend_dev_basics/README.md)：Cube/Vector/Scalar 指令体系、MTE 通道
- [第5章 非量化 FA 开发](../05_flash_attn_non_quantized/README.md)：FA kernel 的四拍循环结构

## 下一章

单引擎优化到位后，进入[第9章 性能优化高阶](../09_performance_high_level/README.md)，攻克制约性能的最后一块硬骨头——负载均衡。

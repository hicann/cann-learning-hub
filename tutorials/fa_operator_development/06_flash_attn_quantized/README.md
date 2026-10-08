# 第6章 全量化 FA 开发

> 所属部分：第二部分 FA 算子开发实战 ｜ 状态：✅ 已发布

## 章节概述

量化是 FA 算子走向极致性能的关键一步。本章从数据格式（FP8 家族 / E8M0 / HiF8）入手，讲解量化原理（scale / 零点 / 对称非对称 / 静态动态），再介绍各种量化方式（T/C/K/G/B 粒度、全量化组合、mx 量化与随路量化），最后结合硬件能力落地两种全量化 FA 方案：FA 支持 MXFP8 量化方案与 FA 支持 HiP8 量化方案。

本章的核心视角：**第 5 章的 Kernel 骨架（四拍循环、分核、乒乓、preload、mask、尾块、负载均衡）全部保留，量化改变的只有「搬入、计算前、计算后」三处数据通路**。

学完本章，读者应能理解 KV Cache 量化带来的带宽/显存收益与精度代价，并掌握在昇腾上实现全量化 FA 的工程方法。

## 章节内容

| 小节 | 内容 | 状态 |
|--|--|--|
| 6.1 章节导学 | 为什么量化、收益账单、与第 5 章的关系 | ✅ 已发布 |
| 6.2 数据格式 | 浮点本质、IEEE 754、FP32/FP16/BF16/FP8 对比、E8M0、HiF8、FP32 累加黄金法则 | ✅ 已发布 |
| 6.3 量化原理 | 量化公式、scale/零点、对称/非对称、静态/动态、量化/伪量化/反量化/重量化 | ✅ 已发布 |
| 6.4 量化方式 | T/C/K/G/B 粒度、全量化组合模式、mx 量化（G-G 特例）、随路量化 | ✅ 已发布 |
| 6.5 FA 支持 MXFP8 量化方案 | 块浮点布局、Kernel 三处 diff、scale 生效位置、对拍方法 | ✅ 已发布 |
| 6.6 FA 支持 HiP8 量化方案 | 单数据格式、锥形精度、与 MXFP8 的 diff 与选型 | ✅ 已发布 |

## 在线体验

| Notebook | 内容 | Link |
|--|--|--|
| 6.1 章节导学 | 量化动机与本章路线 | [06.01_chapter_intro.ipynb](06.01_chapter_intro.ipynb) |
| 6.2 数据格式 | FP8 家族 / E8M0 / HiF8 | [06.02_data_formats.ipynb](06.02_data_formats.ipynb) |
| 6.3 量化原理 | 量化数学与误差来源 | [06.03_quant_principle.ipynb](06.03_quant_principle.ipynb) |
| 6.4 量化方式 | 粒度、组合与随路量化 | [06.04_quant_modes.ipynb](06.04_quant_modes.ipynb) |
| 6.5 MXFP8 量化方案 | 块浮点 FA 的工程实现 | [06.05_fa_mxfp8.ipynb](06.05_fa_mxfp8.ipynb) |
| 6.6 HiP8 量化方案 | 单数据格式 FA 的工程实现 | [06.06_fa_hif8.ipynb](06.06_fa_hif8.ipynb) |
| 6.7 章节测验 | 综合测验 | [06.07_chapter_test.ipynb](06.07_chapter_test.ipynb) |

> 6.2 ~ 6.4 节含可运行的 numpy 实验（E4M3 手工解码、对称/非对称量化、粒度误差对比），任意 Python 3.8+ 环境即可运行。

## 配图

- [fp8_formats.svg](images/fp8_formats.svg)：E4M3 / E5M2 / E8M0 / HiF8 的 8-bit 位布局对比
- [mxfp8_layout.svg](images/mxfp8_layout.svg)：MXFP8 存储布局与 Kernel 数据通路

## 前置知识

- [第5章 非量化 FA 开发](../05_flash_attn_non_quantized/README.md)（先掌握非量化实现）
- [第1章 昇腾开发基础](../01_ascend_dev_basics/README.md)中的数据搬运与格式转换
- [第3章 FA 算子代码结构](../03_fa_code_architecture/README.md)中的 Tiling / Kernel 两层结构

## 下一章

完成开发实战后，进入[第7章 性能优化基础](../07_performance_basics/README.md)，开始性能优化之旅。

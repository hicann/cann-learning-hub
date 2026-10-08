# 第3章 FA 算子代码结构

> 所属部分：第一部分 FA 前导课程 ｜ 状态：✅ 已完成

## 章节概述

本章以基础极简 FA 代码为例，自顶向下拆解 FA 算子的工程骨架，给出一份"读源码的地图"。FA 算子代码量大、模块繁多，但从调用者的视角看只有三层：**算子调用通路**（从框架/API 到算子执行）、**Tiling**（切分参数的计算与下发）、**Kernel**（核函数的分层实现）。

学完本章，读者再打开 [FusedInferAttentionScore](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score) 的源码目录时，应能迅速定位每个文件所处的层次与职责，为第二部分的开发实战建立整体认知。

## 章节内容

| 小节 | 内容 | 状态 |
|--|--|--|
| [3.1 章节介绍](03.01_chapter_intro.ipynb) | 学习目标、路线图 | ✅ |
| [3.2 算子调用通路](03.02_call_path.ipynb) | aclnn 两段式接口、FA 参数表（numHeads/numKeyValueHeads/scaleValue/preTokens/nextTokens/inputLayout）、单算子直调动线、框架集成视角 | ✅ |
| [3.3 数据排布格式](03.03_data_layout.ipynb) | BSND/BNSD/BSH/TND 四种排布、padding 与尾块、actualSeqLengths 两种语义、GQA 排布（BSNGD→BN2SGD）、基本块切分 | ✅ |
| [3.4 Tiling 设计原则](03.04_tiling_design.ipynb) | 五大设计原则（Scalar 上提、TilingKey 治 ICache Miss、负载均衡从简等）、多模板优先级调度、Cube/Vector 切分权衡 | ✅ |
| [3.5 Tiling 源码走读](03.05_tiling_code_walkthrough.ipynb) | 单模板八步流程逐段读码：GQA 维度推导、基本块选择（64→128）、nRatio 配比、workspace 倍数、CalcTschBlockDim | ✅ |
| [3.6 Kernel 整体框架](03.06_kernel_structure.ipynb) | Kernel 入参四件套、SPMD、核间切块（B/N2/S1 轴，KV 完整 S 轴加载）、负载均衡（splitFactorSize） | ✅ |
| [3.7 核内切块与流水](03.07_kernel_pipeline.ipynb) | 四拍循环（mm1→vector1→mm2→vector2）、MM1/MM2 搬运链路（DataCopy/LoadData/LoadDataWithTranspose/FixPipe）、VF 指令与 64×128 基本块、乒乓内存与 Preload | ✅ |
| [3.8 章节测验](03.08_chapter_test.ipynb) | 综合练习（[答案](answer/03.08_answer.txt)） | ✅ |

各小节测验答案见 [answer/](answer/) 目录。

## 前置知识

- [第1章 昇腾开发基础](../01_ascend_dev_basics/README.md)
- [第2章 FlashAttention 原理](../02_fa_principle/README.md)

## 学习建议

- 本章是**代码阅读课**，无需 NPU 环境；建议对照 ops-transformer 仓库源码同步阅读。
- 读源码时先问自己「这行代码属于调用通路、Tiling 还是 Kernel？」——定位层次比读懂细节更重要。
- 代码不必逐行抠：目标是「读懂骨架 + 记住关键设计动机」，第 5 章实战时再回头精读。

## 下一章

理解代码架构后，进入[第4章 FA 算子运行](../04_fa_operator_running/README.md)，亲手把官方 FA 算子跑起来。

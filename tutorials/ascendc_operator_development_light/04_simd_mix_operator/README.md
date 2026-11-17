# 第4章：Ascend C SIMD 融合算子编程

## 章节概述

学习 Cube 与 Vector 协同执行的融合算子。由融合概念进入 CV 融合算子的实现，最后通过 MatmulSinh 实践巩固数据流、同步、Tiling 与核函数直调。

## 前置要求

- 完成第二、三章，掌握矢量算子、矩阵算子和 Tiling。
- 能够阅读并编译 Ascend C Kernel 与 Host 调用代码。

## 学习目标

- 理解算子融合的目的及 Cube/Vector 协作方式。
- 组织融合算子的数据流、同步与 Host 侧参数。
- 完成 CV 融合算子的实现、直调与精度验证。

## Notebook 与在线体验

| Notebook | 在线体验 |
| --- | --- |
| [04.01 章节介绍](04.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_simd_mix_operator/04.01_chapter_intro.ipynb) |
| [04.02 融合算子概念介绍](04.02_fused_operator_concept_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_simd_mix_operator/04.02_fused_operator_concept_intro.ipynb) |
| [04.03 CV融合算子开发](04.03_cv_fused_operator_development.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_simd_mix_operator/04.03_cv_fused_operator_development.ipynb) |
| [04.04 CV融合算子实践](04.04_cv_fused_operator_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_simd_mix_operator/04.04_cv_fused_operator_practice.ipynb) |

## 文件与运行说明

Notebook 按文件名前缀顺序学习，`.01` 为章节介绍。请在本章目录启动或运行 Notebook。本章`images/` 保存课程图片，`answer/` 保存参考答案。运行生成的 `Sources/` 或 `output/` 与仓库配套资源分开保存。练习中的待补充代码需完成后再执行验证。

[返回教程总目录](../README.md)

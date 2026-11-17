# 第6章：Ascend C SIMD 最佳实践（典型算子与极致性能）

## 章节概述

在掌握算子开发与性能分析后，学习多核并行、流水并行和访存优化等典型优化方法，并结合最佳实践示例理解适用条件。

## 前置要求

- 完成第二章和第五章，理解数据切分、同步、Double Buffer 和性能分析。
- 涉及矩阵及融合算子的优化内容，建议先学习第三、四章。

## 学习目标

- 理解核间负载均衡和多核并行的优化方向。
- 了解 Double Buffer、UnitFlag 等流水优化方法。
- 认识 UB Bank 冲突、L2 Cache 控制等访存问题，并结合测量选择优化策略。

## Notebook 与在线体验

| Notebook | 在线体验 |
| --- | --- |
| [06.01 章节介绍](06.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/06_simd_best_practices/06.01_chapter_intro.ipynb) |
| [06.02 算子性能优化最佳实践](06.02_best_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/06_simd_best_practices/06.02_best_practice.ipynb) |

## 文件与运行说明

Notebook 按文件名前缀顺序学习，`.01` 为章节介绍。请在本章目录启动或运行 Notebook。运行生成的 `Sources/` 或 `output/` 与仓库配套资源分开保存。练习中的待补充代码需完成后再执行验证。

[返回教程总目录](../README.md)

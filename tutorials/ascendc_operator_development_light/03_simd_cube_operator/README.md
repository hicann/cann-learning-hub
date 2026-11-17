# 第3章：Ascend C SIMD Cube（矩阵）算子编程

## 章节概述

介绍矩阵乘法的数据形状、存储格式和分块方法，并通过 Matmul 高阶 API 完成矩阵算子的开发、Host 侧 Tiling、编译运行与综合实践。

## 前置要求

- 完成第二章，掌握核函数直调、数据搬运及 Tiling 基础。
- 了解矩阵乘法与 C/C++ 工程构建，具备 CANN 和 NPU 运行环境。

## 学习目标

- 理解矩阵乘法的数据布局、分块和矩阵编程范式。
- 使用 Matmul 高阶 API 实现矩阵乘算子并传递 Tiling 参数。
- 完成 Matmul 综合实践与结果验证。

## Notebook 与在线体验

| Notebook | 在线体验 |
| --- | --- |
| [03.01 章节介绍](03.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simd_cube_operator/03.01_chapter_intro.ipynb) |
| [03.02 矩阵乘法介绍](03.02_introduction_to_matrix_multiplication.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simd_cube_operator/03.02_introduction_to_matrix_multiplication.ipynb) |
| [03.03 高阶API矩阵乘算子开发](03.03_matrix_multiplication_operator_development_with_high_level_api.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simd_cube_operator/03.03_matrix_multiplication_operator_development_with_high_level_api.ipynb) |
| [03.04 Matmul综合实践](03.04_matmul_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simd_cube_operator/03.04_matmul_practice.ipynb) |

## 文件与运行说明

Notebook 按文件名前缀顺序学习，`.01` 为章节介绍。请在本章目录启动或运行 Notebook。本章`images/` 保存课程图片，`answer/` 保存参考答案。运行生成的 `Sources/` 或 `output/` 与仓库配套资源分开保存。练习中的待补充代码需完成后再执行验证。

[返回教程总目录](../README.md)

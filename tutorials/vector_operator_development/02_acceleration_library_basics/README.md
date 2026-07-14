# 第二章：加速库基础认知

本章首先介绍算子执行流程，帮助理解 aclnn 单算子 API、GE 图模式等不同执行模式的工作原理；然后介绍 CANN 算子库的标准目录结构和开发环境部署方法。以 FastGelu 算子为例，带领你理解算子工程的完整目录布局、各层文件的作用和关联，以及开发环境的安装与配置流程。

## 章节内容

| 章节 | 标题 | 状态 |
|------|------|------|
| 2.1 | [章节介绍](./02.01_chapter_intro.ipynb) | ✅ |
| 2.2 | [算子执行流程](./02.02_operator_execution_flow.ipynb) | ✅ |
| 2.3 | [算子目录结构](./02.03_operator_directory_structure.ipynb) | ✅ |
| 2.4 | [开发环境部署](./02.04_development_environment_setup.ipynb) | ✅ |
| 2.5 | [章节实践](./02.05_chapter_practice.ipynb) | ✅ |

## 学习目标

完成本章后，你将：

- 理解算子的多种执行模式及其应用场景
- 掌握 aclnn 单算子 API 执行流程的详细步骤
- 理解 CANN 算子库的标准目录结构
- 掌握 Tiling 的概念、参数组成和实现流程
- 了解 CMakeLists.txt 的编译配置和 build.sh 常用命令
- 能够安装 CANN Toolkit 和 Ops 包并配置开发环境

## 前置要求

- 完成第一章（基础掌握）的学习

## 下一章

[第三章：Ascend 算子开发](../03_ascend_operator_development/README.md)

# 第1章：算子开发基本概念（什么是算子，什么是CANN，什么是 AscendC）

## 章节概述

从人工智能与算子基础出发，认识 CANN 软件栈、昇腾 NPU 与 Ascend C，并通过 Hello World 完成首次核函数编写和调用。最后通过章节练习巩固基础概念。

## 前置要求

- 具备 C/C++、Python 和 Linux 命令行基础。
- 运行 Hello World 前，需配置 CANN 开发工具链及匹配的昇腾 NPU 环境。

## 学习目标

- 理解算子、张量、CANN 和 AI Core 的基本概念。
- 认识 Ascend C 核函数及 Host/Device 协作方式。
- 编译运行 Hello World，完成基础知识自测。

## Notebook 与在线体验

| Notebook | 在线体验 |
| --- | --- |
| [01.01 章节介绍](01.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.01_chapter_intro.ipynb) |
| [01.02 人工智能与算子基础](01.02_ai_and_operator_basics.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.02_ai_and_operator_basics.ipynb) |
| [01.03 CANN架构与昇腾NPU原理](01.03_cann_arch_ascend_npu_principle.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb) |
| [01.04 Ascend C算子开发基本概念](01.04_ascend_c_op_dev_basic_concepts.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.04_ascend_c_op_dev_basic_concepts.ipynb) |
| [01.05 Ascend C Hello World](01.05_HelloWorld.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.05_HelloWorld.ipynb) |
| [01.06 章节练习](01.06_chapter_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.06_chapter_practice.ipynb) |

## 文件与运行说明

Notebook 按文件名前缀顺序学习，`.01` 为章节介绍。请在本章目录启动或运行 Notebook。本章`images/` 保存课程图片，`answer/` 保存参考答案。运行生成的 `Sources/` 或 `output/` 与仓库配套资源分开保存。练习中的待补充代码需完成后再执行验证。

[返回教程总目录](../README.md)

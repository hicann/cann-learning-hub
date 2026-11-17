# 第8章：Ascend C 算子入图（PyTorch、AclGraph）

## 章节概述

在 PyTorch 单算子调用基础上，学习 AclGraph 入图编译与运行，理解捕获、实例化、重放如何减少 Host 侧重复下发开销，并验证自定义算子的图执行。

## 前置要求

- 完成第七章，掌握 pybind11 与 torch.library 接入方式。
- 配置匹配的 PyTorch、torch_npu、CANN 和 NPU 环境，具体接口支持要求以本章 Notebook 为准。

## 学习目标

- 理解 AclGraph 与 NPUGraph 的关系及捕获、实例化、重放流程。
- 了解显式图执行与 torch.compile 后端的使用方式。
- 验证自定义算子入图后的结果，并观察执行耗时。

## Notebook 与在线体验

| Notebook | 在线体验 |
| --- | --- |
| [08.01 章节介绍](08.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/08_operator_graph_integration/08.01_chapter_intro.ipynb) |
| [08.02 AclGraph入图编译与运行](08.02_aclgraph_compile_launch.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/08_operator_graph_integration/08.02_aclgraph_compile_launch.ipynb) |

## 文件与运行说明

Notebook 按文件名前缀顺序学习，`.01` 为章节介绍。请在本章目录启动或运行 Notebook。本章`src/` 保存配套源码，`answer/` 保存参考答案。运行生成的 `Sources/` 或 `output/` 与仓库配套资源分开保存。练习中的待补充代码需完成后再执行验证。

[返回教程总目录](../README.md)

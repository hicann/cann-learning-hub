# 第7章：Ascend C PyTorch 单算子调用

## 章节概述

将 Ascend C 自定义算子接入 PyTorch。先用 Add 学习 pybind11 与 torch.library 两种 Kernel 直调方式，再通过 Softmax 和 Matmul 综合实践掌握绑定、编译与正确性验证。

## 前置要求

- 完成第二章，理解 Kernel、Host 调用与编译流程；Matmul 实践还需第三章知识。
- 安装匹配的 PyTorch、torch_npu、pybind11、Python 开发头文件和 CANN 工具链。

## 学习目标

- 通过 pybind11 暴露自定义 Kernel 调用接口。
- 通过 torch.library 注册并调用自定义算子。
- 完成 Softmax、Matmul 的框架接入与精度验证。

## Notebook 与在线体验

| Notebook | 在线体验 |
| --- | --- |
| [07.01 章节介绍](07.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/07_pytorch_single_operator_call/07.01_chapter_intro.ipynb) |
| [07.02 PyTorch框架下Kernel直调](07.02_kernel_pytorch_call.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/07_pytorch_single_operator_call/07.02_kernel_pytorch_call.ipynb) |
| [07.03 PyTorch调用算子综合实践](07.03_pytorch_operator_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/07_pytorch_single_operator_call/07.03_pytorch_operator_practice.ipynb) |

## 文件与运行说明

Notebook 按文件名前缀顺序学习，`.01` 为章节介绍。请在本章目录启动或运行 Notebook。本章`src/` 保存配套源码，`answer/` 保存参考答案。运行生成的 `Sources/` 或 `output/` 与仓库配套资源分开保存。练习中的待补充代码需完成后再执行验证。

[返回教程总目录](../README.md)

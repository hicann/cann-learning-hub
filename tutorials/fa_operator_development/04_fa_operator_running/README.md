# 第4章 FA 算子运行

> 所属部分：第二部分 FA 算子开发实战 ｜ 状态：✅ 已完成

## 章节概述

先跑起来再说。本章带读者完成从环境搭建到 FA 算子成功运行的全过程：搭建昇腾开发环境，认识 FA 算子工程的组织方式，逐参解读 FA 算子接口（FA 推理），最终基于 CANNSim 仿真完成 FA 算子的调用并与 numpy golden 对拍验证。

本章是"体验"环节：亲眼见证 FlashAttention 相对标准 Attention 实现的性能优势，带着"它是怎么写出来的"这个悬念进入后续开发章节。

## 章节内容

| 小节 | 内容 | 状态 |
|--|--|--|
| [4.1 章节介绍](04.01_chapter_intro.ipynb) | 学习目标、路线图 | ✅ |
| [4.2 环境搭建](04.02_env_setup.ipynb) | 三条路径：本地 950 服务器 / CANNLab 在线环境 / CANNSim 仿真；驱动 → CANN → torch_npu 部署顺序与检查清单 | ✅ |
| [4.3 FA 算子工程介绍](04.03_project_structure.ipynb) | ops-transformer 工程目录结构、op_host/op_kernel 与三层结构的映射、AIC/AIV 双 `.o` 构建产物 | ✅ |
| [4.4 FA 算子接口介绍](04.04_api_reference.ipynb) | FA 推理接口逐参解读：数据组 / 掩码组（pseShift、dropMask、paddingMask、attenMask、preTokens/nextTokens）/ 形状语义组（numHeads、numKeyValueHeads、actualSeqLengths、inputLayout）/ scaleValue | ✅ |
| [4.5 FA 算子运行与调测](04.05_run_and_verify.ipynb) | CANNSim 仿真运行、DebugTools 工具、四步调测流程（tiling 编译→运行、kernel 编译→运行）、golden（bin）与 json（属性）、DumpTensor 逐拍定位、mskpp「四个分开」方法论 | ✅ |
| [4.6 章节测验](04.06_chapter_test.ipynb) | 综合练习（[答案](answer/04.06_answer.txt)） | ✅ |

各小节测验答案见 [answer/](answer/) 目录。

## 前置知识

- [第2章 FlashAttention 原理](../02_fa_principle/README.md)（理解接口参数背后的数学）
- [第3章 FA 算子代码结构](../03_fa_code_architecture/README.md)

## 环境要求

- 昇腾 NPU 硬件（Ascend 950 或 Atlas 系列）、CANNLab 在线环境，或 CANNSim 仿真环境
- 按 [CANN 下载页面](https://www.hiascend.com/cann/download) 完成开发环境部署

## 下一章

跑通官方算子后，进入[第5章 非量化 FA 开发](../05_flash_attn_non_quantized/README.md)，亲手从 0 到 1 构建一个 FA 算子。

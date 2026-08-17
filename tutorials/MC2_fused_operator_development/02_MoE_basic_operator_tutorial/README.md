# 第2章 算子开发教程

## 章节概述

本章教程是昇腾 CANN（Compute Architecture for Neural Networks）算子开发系列教程的高级篇章，聚焦于大模型领域最前沿的 **MoE（Mixture of Experts，混合专家模型）** 架构下的通信算子开发。课程从 MoE 架构原理出发，逐步深入到 Dispatch/Combine 核心算子的实现细节，并通过量化 Dispatch 实战项目帮助开发者掌握高性能 MoE 通信算子的开发技能。学习 Win 区内存布局、Tiling 策略、流水并行等关键优化技术, 动手实现支持 MXFP8 动态量化的高性能 Dispatch 算子。
## 在线体验

| Notebook | Link | 状态 |
|--|--|--|
| 2.1 章节介绍 | - | ✅ 已发布 |
| 2.2 MoE 架构概述 | - | ✅ 已发布 |
| 2.3 并行策略 | - | ✅ 已发布 |
| 2.4 Dispatch/Combine 算子 | - | ✅ 已发布 |
| 2.5 算子逻辑概述 | - | ✅ 已发布 |
| 2.6 核心流程拆解 | - | ✅ 已发布 |
| 2.7 Win 区内存布局 | - | ✅ 已发布 |
| 2.8 Tiling 指南 | - | ✅ 已发布 |
| 2.9 Kernel 阶段指南 | - | ✅ 已发布 |
| 2.10 章节测试 | - | ✅ 已发布 |

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A5 训练/推理系列产品|
| CANN 版本 | 9.0.0 及以上 |
| Python | 3.11 |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 950尝鲜体验 | cann_9.0.0_py3.11-A5-arm | Python 3.11.4 |参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)创建CANNLab环境运行notebook |

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)，并选择对应CANN版本的文档。


## 运行环境与约束

- **硬件**：NPU多卡环境
- **芯片型号**：当前样例仅支持'dav-3510'。
- **软件环境**：需要已安装 Ascend CANN Tooltik。
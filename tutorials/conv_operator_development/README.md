# Ascend C 卷积算子开发教程

本教程将带你学习基于 Ascend C 的卷积算子开发方法，围绕卷积算子的核心概念、设计思路与开发流程展开详细讲解，涵盖卷积数学原理、img2col 变换、前向 Conv2D 与反向 Conv3D Backprop Filter 核函数开发等典型场景，助力开发者掌握昇腾 NPU 上卷积算子的开发要点与实操技巧。

教程按章节划分，每个章节均包含以下内容：
- **Notebooks**：包含课程知识点与练习题，适用于自主学习或讲师引导式教学。
- **SRC**：包含课程中所有的源码，供开发者自行下载及修改。
- **Answer**：包含课后练习与实践的参考答案。

> 本教程当前仅针对 **Ascend 950 系列**产品进行验证，其它产品使用可能存在问题，欢迎开发者提出 Issue 或 PR 进行共建。

## 适用对象

- 具备 C/C++ 与 Python 基础，了解深度学习基本概念。
- 希望学习昇腾 NPU 算子开发，特别是卷积相关算子的开发者。

## 学习目标

完成本教程后，你将能够：

1. 理解卷积运算的数学原理与反向传播的梯度推导。
2. 掌握 img2col 变换与 Load3D 指令的协作关系。
3. 掌握前向 Conv2D V2 核函数的完整开发流程（Tiling 设计、Buffer 管理、im2col + Mmad + Fixpipe 流水线）。
4. 掌握反向 Conv3D Backprop Filter 核函数的开发方法与前向卷积的架构差异。

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Ascend 950 系列 |
| CANN 版本 | 9.0.0 及以上 |
| Python | 3.11 及以上 |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 950 尝鲜体验环境 | cann_9.0.0-beta.2-py3.12-a5 | Python 3.12 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建 Ascend 950 环境并体验本教程。进入开发者空间后，建议将 CANN 套件升级至最新稳定版本。|

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)，选择对应 CANN 版本文档。

## 章节目录

### 第一章：卷积算子基础开发

| Notebook | Link | 状态 |
| --- | --- | --- |
| 1.1 章节介绍 | - | ✅ 已发布 |
| 1.2 背景介绍与卷积数学原理 | - | ✅ 已发布 |
| 1.3 Conv2D V2 核函数开发 | - | ✅ 已发布 |
| 1.4 Conv3D Backprop Filter V2 核函数开发与章节实践 | - | ✅ 已发布 |

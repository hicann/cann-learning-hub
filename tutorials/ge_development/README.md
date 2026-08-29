# GE 图引擎开发系列教程

本教程面向 GE 图引擎初学者，包括模型部署工程师和推理应用开发者。内容围绕 GE 基础概念、AscendIR 构图、ATC 离线编译、ACL 推理部署与 GeSession 在线执行展开，帮助开发者掌握 GE 的基本定位和端到端推理流程。

教程按章节划分，每个章节均包含以下内容：
- Notebooks：包含课程知识点与练习题，适用于自主学习或讲师引导式教学，可在 gitcode 提供的轻量级 notebook 上运行。也可自行搭建 jupyter lab，在本地环境执行使用。
- answer：章节练习参考答案。

>- **注意：**
>- 本教程面向 GE 用户侧能力学习，重点关注模型编译、执行、部署与扩展开发实践，不深入展开 GE 内部实现细节。
>- 课程内容以 CANN/昇腾社区公开文档、GE 图开发指南、ATC 使用指南、ACL 推理开发文档及相关示例为参考。
>- 本教程当前仅针对 Atlas A2 系列产品进行验证，其它产品使用存在问题，欢迎开发者提出 issue 或 PR 进行共建。环境要求：CANN 9.0.0 及以上，Linux 已部署 CANN 开发环境（参考 [CANN 下载页面](https://www.hiascend.com/cann/download)）；「基础概念入门」无需 NPU，其余动手实践需 NPU 或昇腾云环境。

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上 |
| Python | 3.11 |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| cann-learning-hub 在线体验 notebook | cann_9.0.0_py3.11-A2-arm | Python 3.11.15 | 各 Notebook 表格中的"在线体验"链接可直接打开运行 |
| CANNLab 云开发环境 | cann_9.0.0_py3.11-A2-arm | Python 3.11.4 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建 CANNLab 环境运行 notebook |

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)。
> Notebook 用于阅读教程和章节练习；涉及 ATC 编译、ACL/GeSession 执行的动手实践需在配备 Ascend NPU 的服务器或昇腾云环境上运行（「基础概念入门」章节除外，无需 NPU）。



## GE 图引擎开发系列（快速入门）

### 第一章：基础概念入门

| Notebook | Link | 状态 |
| --- | --- | --- |
| 1.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ge_development&scanFilePath=tutorials/ge_development/01_basic_concepts/01.01_chapter_intro.ipynb) | ✅ 已发布 |
| 1.2 GE 定位与核心概念 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ge_development&scanFilePath=tutorials/ge_development/01_basic_concepts/01.02_ge_overview.ipynb) | ✅ 已发布 |
| 1.3 AscendIR 基础概念 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ge_development&scanFilePath=tutorials/ge_development/01_basic_concepts/01.03_ascend_ir.ipynb) | ✅ 已发布 |
| 1.4 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ge_development&scanFilePath=tutorials/ge_development/01_basic_concepts/01.04_chapter_practice.ipynb) | ✅ 已发布 |

### 第二章：推理流程介绍

| Notebook | Link | 状态 |
| --- | --- | --- |
| 2.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ge_development&scanFilePath=tutorials/ge_development/02_inference_workflow/02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 2.2 离线推理流程：ATC 编译与 ACL 推理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ge_development&scanFilePath=tutorials/ge_development/02_inference_workflow/02.02_offline_inference.ipynb) | ✅ 已发布 |
| 2.3 在线执行流程：GeSession 构图与执行 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ge_development&scanFilePath=tutorials/ge_development/02_inference_workflow/02.03_online_execution.ipynb) | ✅ 已发布 |
| 2.4 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ge_development&scanFilePath=tutorials/ge_development/02_inference_workflow/02.04_chapter_practice.ipynb) | ✅ 已发布 |

## GE 图引擎开发系列（进阶开发）

### 第三章：图编译

| Notebook | Link | 状态 |
| --- | --- | --- |
| 3.1 章节介绍 | 在线体验建设中 | 🚧 建设中 |
| 3.2 图的构建与输入：AscendIR 构图与 Parser 解析 | 在线体验建设中 | 🚧 建设中 |
| 3.3 编译配置：融合、精度、buffer、stream 等选项 | 在线体验建设中 | 🚧 建设中 |
| 3.4 编译产物：OM 结构、外置权重、SO in OM、模型缓存 | 在线体验建设中 | 🚧 建设中 |
| 3.5 图编译扩展能力：自定义算子入图 | 在线体验建设中 | 🚧 建设中 |
| 3.6 图编译扩展能力：自定义融合 Pass | 在线体验建设中 | 🚧 建设中 |
| 3.7 章节练习 | 在线体验建设中 | 🚧 建设中 |

### 第四章：模型执行与优化

| Notebook | Link | 状态 |
| --- | --- | --- |
| 4.1 章节介绍 | 在线体验建设中 | 🚧 建设中 |
| 4.2 静态 Shape 执行流程：整图下沉 | 在线体验建设中 | 🚧 建设中 |
| 4.3 动态 Shape 执行流程：Host 调度 | 在线体验建设中 | 🚧 建设中 |
| 4.4 静态 Shape 执行优化技术 | 在线体验建设中 | 🚧 建设中 |
| 4.5 动态 Shape 执行优化技术 | 在线体验建设中 | 🚧 建设中 |
| 4.6 章节练习 | 在线体验建设中 | 🚧 建设中 |

## GE 图引擎开发系列（高阶应用）

### 第五章：实践与问题定位

| Notebook | Link | 状态 |
| --- | --- | --- |
| 5.1 章节介绍 | 在线体验建设中 | 🚧 建设中 |
| 5.2 GE 对接 PyTorch：TorchAir 图模式 | 在线体验建设中 | 🚧 建设中 |
| 5.3 GE 对接 TensorFlow | 在线体验建设中 | 🚧 建设中 |
| 5.4 常见问题定位方法 | 在线体验建设中 | 🚧 建设中 |
| 5.5 章节练习 | 在线体验建设中 | 🚧 建设中 |

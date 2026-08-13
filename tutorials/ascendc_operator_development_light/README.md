# Ascend C 算子开发系列教程（Kernel 直调）

本教程将带你学习面向昇腾 NPU 的 Ascend C 高性能算子开发，包含算子核函数、算子编译、Tiling 计算、矩阵算子开发、CV 融合算子开发、算子调试调优等核心内容。

教程按章节划分，每个章节均包含以下内容：
- Notebooks：包含课程知识点与练习题，适用于自主学习或讲师引导式教学，可在 GitCode 提供的轻量级 Notebook 上运行，也可自行搭建 JupyterLab 在本地环境中使用。
- src：包含课程中的源码，供开发者下载及修改。

## 适用人群

- 希望了解并掌握 Ascend C 算子编程方法的开发者
- AI 算法工程师、C/C++ 开发者、算子开发工程师

## 前置知识要求

在学习本教程之前，建议先完成以下前置学习：

- 阅读 [cann-learning-hub 快速入门](https://gitcode.com/cann/cann-learning-hub/quick_start) 中的 `cann_basics` 章节，了解 AI 基础概念、NPU 硬件架构与 CANN 软件栈
- 掌握 C/C++ 编程基础
- 了解并行编程基本概念

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上 |
| Python | 3.11 |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| cann-learning-hub 在线体验 notebook | cann_9.0.0_py3.11-A2-arm | Python 3.11.15 | 各 Notebook 表格中的"在线体验"链接可直接打开运行 |
| CANNLab 云开发环境 | cann_9.0.0_py3.11-A2-arm | Python 3.11.4 |参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)创建CANNLab环境运行notebook |

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)，并选择对应 CANN 版本的文档。


### 第一章：Ascend C算子开发入门基础

| Notebook | Link | 状态 |
|--|--|--|
| 1.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.01_chapter_intro.ipynb) | ✅ 已发布 |
| 1.2 人工智能与算子基础 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.02_ai_and_operator_basics.ipynb) | ✅ 已发布 |
| 1.3 CANN架构与昇腾NPU原理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb) | ✅ 已发布 |
| 1.4 Ascend C算子开发的基本概念 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.04_ascend_c_op_dev_basic_concepts.ipynb) | ✅ 已发布 |
| 1.5 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.05_chapter_practice.ipynb) | ✅ 已发布 |

### 第二章：Ascend C算子开发基础知识

| Notebook | Link | 状态 |
|--|--|--|
| 2.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_AscendC_basic/02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 2.2 Ascend C的"Hello world" | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_AscendC_basic/02.02_HelloWorld.ipynb) | ✅ 已发布 |
| 2.3 AscendC编程范式介绍与API介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_AscendC_basic/02.03_ascendc_programming_paradigm_and_api_introduction.ipynb) | ✅ 已发布 |
| 2.4 基于Add算子的核函数介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_AscendC_basic/02.04_introduction_to_kernel_functions_based_on_add_operator.ipynb) | ✅ 已发布 |
| 2.5 泛化Tiling设计 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_AscendC_basic/02.05_generalized_tiling_design.ipynb) | ✅ 已发布 |
| 2.6 PyTorch框架下Kernel直调 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_AscendC_basic/02.06_kernel_pytorch_call.ipynb) | ✅ 已发布 |
| 2.7 核函数实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_AscendC_basic/02.07_kernel_function_practice.ipynb) | ✅ 已发布 |
| 2.8 Vector算子开发实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_AscendC_basic/02.08_vector_operator_practice.ipynb) | ✅ 已发布 |

### 第三章：简单算子实践

| Notebook | Link | 状态 |
|--|--|--|
| 3.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simple_operator_practice/03.01_chapter_intro.ipynb) | ✅ 已发布 |
| 3.2 矩阵乘法介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simple_operator_practice/03.02_introduction_to_matrix_multiplication.ipynb) | ✅ 已发布 |
| 3.3 基于高阶API的矩阵乘算子开发 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simple_operator_practice/03.03_matrix_multiplication_operator_development_with_high_level_api.ipynb) | ✅ 已发布 |
| 3.4 融合算子概念介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simple_operator_practice/03.04_fused_operator_concept_intro.ipynb) | ✅ 已发布 |
| 3.5 CV融合算子开发 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simple_operator_practice/03.05_cv_fused_operator_development.ipynb) | ✅ 已发布 |
| 3.6 Matmul算子实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simple_operator_practice/03.06_matmul_practice.ipynb) | ✅ 已发布 |
| 3.7 CV融合算子实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simple_operator_practice/03.07_cv_fused_operator_practice.ipynb) | ✅ 已发布 |

### 第四章：Ascend C算子调试调优

| Notebook | Link | 状态 |
|--|--|--|
| 4.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_debug/04.01_chapter_intro.ipynb) | ✅ 已发布 |
| 4.2 NPU域上板调试 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_debug/04.02_NPU_On-Board_Debugging.ipynb) | ✅ 已发布 |
| 4.3 算子性能优化工具 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_debug/04.03_profiling_tool_usage.ipynb) | ✅ 已发布 |
| 4.4 算子仿真调优 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_debug/04.04_simulation_analysis.ipynb) | ✅ 已发布 |

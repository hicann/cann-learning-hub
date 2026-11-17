# Ascend C 算子开发系列教程（Kernel 直调）

本教程将带你学习面向昇腾 NPU 的 Ascend C 高性能算子开发，包含算子核函数、算子编译、Tiling 计算、矩阵算子开发、CV 融合算子开发、算子调试调优、Aclgraph 入图编译与运行等核心内容。

教程按章节划分，每个章节均包含以下内容：
- Notebooks：包含课程知识点与练习题，适用于自主学习或讲师引导式教学，可在 GitCode 提供的轻量级 Notebook 上运行，也可自行搭建 JupyterLab 在本地环境中使用。
- src：包含课程中的源码，供开发者下载及修改。

## 适用人群

- 希望了解并掌握 Ascend C 算子编程方法的开发者
- AI 算法工程师、C/C++ 开发者、算子开发工程师

## 前置知识要求

在学习本教程之前，建议先完成以下前置学习：

- 阅读 [cann-learning-hub 快速入门](https://gitcode.com/cann/cann-learning-hub/tree/master/quick_start) 中的 `cann_basics` 章节，了解 AI 基础概念、NPU 硬件架构与 CANN 软件栈
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

## 课程导航

课程按以下八章组织。每章先阅读章节介绍，再按编号学习。请将 Notebook 所在章节目录作为工作目录，避免相对资源路径解析错误。

### 第1章：算子开发基本概念（什么是算子，什么是CANN，什么是 AscendC）

| Notebook | 在线体验 | 状态 |
| --- | --- | --- |
| [01.01 章节介绍](01_basic_overview/01.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.01_chapter_intro.ipynb) | ✅ 已发布 |
| [01.02 人工智能与算子基础](01_basic_overview/01.02_ai_and_operator_basics.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.02_ai_and_operator_basics.ipynb) | ✅ 已发布 |
| [01.03 CANN架构与昇腾NPU原理](01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb) | ✅ 已发布 |
| [01.04 Ascend C算子开发基本概念](01_basic_overview/01.04_ascend_c_op_dev_basic_concepts.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.04_ascend_c_op_dev_basic_concepts.ipynb) | ✅ 已发布 |
| [01.05 Ascend C Hello World](01_basic_overview/01.05_HelloWorld.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.05_HelloWorld.ipynb) | ✅ 已发布 |
| [01.06 章节练习](01_basic_overview/01.06_chapter_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/01_basic_overview/01.06_chapter_practice.ipynb) | ✅ 已发布 |

### 第2章：Ascend C SIMD 编程基础与矢量算子开发（A2/A3）

| Notebook | 在线体验 | 状态 |
| --- | --- | --- |
| [02.01 章节介绍](02_simd_basics_and_vector_operator/02.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.01_chapter_intro.ipynb) | ✅ 已发布 |
| [02.02 SIMD编程概述与Ascend C编程接口概述](02_simd_basics_and_vector_operator/02.02_simd_programming_and_ascendc_api_overview.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.02_simd_programming_and_ascendc_api_overview.ipynb) | ✅ 已发布 |
| [02.03 TPipe/TQue编程框架与矢量算子开发](02_simd_basics_and_vector_operator/02.03_vector_operator_development_with_tpipe_tque.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.03_vector_operator_development_with_tpipe_tque.ipynb) | ✅ 已发布 |
| [02.04 自主分配内存与同步管理](02_simd_basics_and_vector_operator/02.04_manual_memory_allocation_and_sync_management.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.04_manual_memory_allocation_and_sync_management.ipynb) | ✅ 已发布 |
| [02.05 核函数实践](02_simd_basics_and_vector_operator/02.05_kernel_function_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.05_kernel_function_practice.ipynb) | ✅ 已发布 |
| [02.06 泛化Tiling设计](02_simd_basics_and_vector_operator/02.06_generalized_tiling_design.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.06_generalized_tiling_design.ipynb) | ✅ 已发布 |
| [02.07 Vector算子开发实践](02_simd_basics_and_vector_operator/02.07_vector_operator_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/02_simd_basics_and_vector_operator/02.07_vector_operator_practice.ipynb) | ✅ 已发布 |

### 第3章：Ascend C SIMD Cube（矩阵）算子编程

| Notebook | 在线体验 | 状态 |
| --- | --- | --- |
| [03.01 章节介绍](03_simd_cube_operator/03.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simd_cube_operator/03.01_chapter_intro.ipynb) | ✅ 已发布 |
| [03.02 矩阵乘法介绍](03_simd_cube_operator/03.02_introduction_to_matrix_multiplication.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simd_cube_operator/03.02_introduction_to_matrix_multiplication.ipynb) | ✅ 已发布 |
| [03.03 高阶API矩阵乘算子开发](03_simd_cube_operator/03.03_matrix_multiplication_operator_development_with_high_level_api.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simd_cube_operator/03.03_matrix_multiplication_operator_development_with_high_level_api.ipynb) | ✅ 已发布 |
| [03.04 Matmul综合实践](03_simd_cube_operator/03.04_matmul_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/03_simd_cube_operator/03.04_matmul_practice.ipynb) | ✅ 已发布 |

### 第4章：Ascend C SIMD 融合算子编程

| Notebook | 在线体验 | 状态 |
| --- | --- | --- |
| [04.01 章节介绍](04_simd_mix_operator/04.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_simd_mix_operator/04.01_chapter_intro.ipynb) | ✅ 已发布 |
| [04.02 融合算子概念介绍](04_simd_mix_operator/04.02_fused_operator_concept_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_simd_mix_operator/04.02_fused_operator_concept_intro.ipynb) | ✅ 已发布 |
| [04.03 CV融合算子开发](04_simd_mix_operator/04.03_cv_fused_operator_development.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_simd_mix_operator/04.03_cv_fused_operator_development.ipynb) | ✅ 已发布 |
| [04.04 CV融合算子实践](04_simd_mix_operator/04.04_cv_fused_operator_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/04_simd_mix_operator/04.04_cv_fused_operator_practice.ipynb) | ✅ 已发布 |

### 第5章：Ascend C 调试与性能分析

| Notebook | 在线体验 | 状态 |
| --- | --- | --- |
| [05.01 章节介绍](05_debug_and_profile/05.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_debug_and_profile/05.01_chapter_intro.ipynb) | ✅ 已发布 |
| [05.02 NPU上板调试](05_debug_and_profile/05.02_NPU_On-Board_Debugging.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_debug_and_profile/05.02_NPU_On-Board_Debugging.ipynb) | ✅ 已发布 |
| [05.03 算子上板调优](05_debug_and_profile/05.03_profiling_tool_usage.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_debug_and_profile/05.03_profiling_tool_usage.ipynb) | ✅ 已发布 |
| [05.04 算子仿真调优](05_debug_and_profile/05.04_simulation_analysis.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_debug_and_profile/05.04_simulation_analysis.ipynb) | ✅ 已发布 |

### 第6章：Ascend C SIMD 最佳实践（典型算子与极致性能）

| Notebook | 在线体验 | 状态 |
| --- | --- | --- |
| [06.01 章节介绍](06_simd_best_practices/06.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/06_simd_best_practices/06.01_chapter_intro.ipynb) | ✅ 已发布 |
| [06.02 算子性能优化最佳实践](06_simd_best_practices/06.02_best_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/06_simd_best_practices/06.02_best_practice.ipynb) | ✅ 已发布 |

### 第7章：Ascend C PyTorch 单算子调用

| Notebook | 在线体验 | 状态 |
| --- | --- | --- |
| [07.01 章节介绍](07_pytorch_single_operator_call/07.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/07_pytorch_single_operator_call/07.01_chapter_intro.ipynb) | ✅ 已发布 |
| [07.02 PyTorch框架下Kernel直调](07_pytorch_single_operator_call/07.02_kernel_pytorch_call.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/07_pytorch_single_operator_call/07.02_kernel_pytorch_call.ipynb) | ✅ 已发布 |
| [07.03 PyTorch调用算子综合实践](07_pytorch_single_operator_call/07.03_pytorch_operator_practice.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/07_pytorch_single_operator_call/07.03_pytorch_operator_practice.ipynb) | ✅ 已发布 |

### 第8章：Ascend C 算子入图（PyTorch、AclGraph）

| Notebook | 在线体验 | 状态 |
| --- | --- | --- |
| [08.01 章节介绍](08_operator_graph_integration/08.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/08_operator_graph_integration/08.01_chapter_intro.ipynb) | ✅ 已发布 |
| [08.02 AclGraph入图编译与运行](08_operator_graph_integration/08.02_aclgraph_compile_launch.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/08_operator_graph_integration/08.02_aclgraph_compile_launch.ipynb) | ✅ 已发布 |

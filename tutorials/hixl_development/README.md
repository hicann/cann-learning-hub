# HIXL 开发系列教程

本教程面向昇腾单边通信库 HIXL（Huawei Xfer Library）初学者，内容包含单边通信基础、HIXL 核心 API、HIXL 多种传输模式、问题定位与性能分析等内容。

教程按章节划分，每个章节均包含以下内容：

- Notebooks：包含课程知识点与练习题，适用于自主学习或讲师引导式教学，可在 gitcode 提供的轻量级 notebook 上运行。也可自行搭建 jupyter lab，在本地环境执行使用。
- SRC：包含课程中引用的源码、脚本、样例数据或练习骨架，供开发者自行下载及修改。
- answer：包含章节练习参考答案。

> - **注意：**
> - 本教程当前仅针对 Atlas A2/A3 系列产品进行验证，其它产品使用存在问题，欢迎开发者提出 issue 或 PR 进行共建。

## Notebooks

### HIXL 基础介绍

| Notebook | Link | 状态 |
| --- | --- | --- |
| 1.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/01_basic_overview/01.01_chapter_intro.ipynb&npuCnt=2) | ✅ 已发布 |
| 1.2 单边通信基础概念 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/01_basic_overview/01.02_one_sided_comm_basic_concepts.ipynb&npuCnt=2) | ✅ 已发布 |
| 1.3 HIXL 简介 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/01_basic_overview/01.03_hixl_introduction.ipynb&npuCnt=2) | ✅ 已发布 |
| 1.4 HIXL 整体开发流程 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/01_basic_overview/01.04_hixl_development_flow.ipynb&npuCnt=2)  | ✅ 已发布 |
| 1.5 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/01_basic_overview/01.05_chapter_practice.ipynb&npuCnt=2) | ✅ 已发布 |

### HIXL 核心 API 使用方法

| Notebook | Link | 状态 |
| --- | --- | --- |
| 2.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/02_core_api_development/02.01_chapter_intro.ipynb&npuCnt=2) | ✅ 已发布 |
| 2.2 HIXL 资源管理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/02_core_api_development/02.02_hixl_resource_management.ipynb&npuCnt=2) | ✅ 已发布 |
| 2.3 HIXL 内存管理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/02_core_api_development/02.03_hixl_memory_management.ipynb&npuCnt=2) | ✅ 已发布 |
| 2.4 HIXL 链路管理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/02_core_api_development/02.04_hixl_link_management.ipynb&npuCnt=2) | ✅ 已发布 |
| 2.5 HIXL 数据传输 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/02_core_api_development/02.05_hixl_data_transfer.ipynb&npuCnt=2) | ✅ 已发布 |
| 2.6 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/02_core_api_development/02.06_chapter_practice.ipynb&npuCnt=2) | ✅ 已发布 |

### HIXL 传输模式介绍

| Notebook | Link | 状态 |
| --- | --- | --- |
| 3.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/03_multi_transfer_mode/03.01_chapter_intro.ipynb&npuCnt=2) | ✅ 已发布 |
| 3.2 直传模式 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/03_multi_transfer_mode/03.02_direct_transfer_mode.ipynb&npuCnt=2) | ✅ 已发布 |
| 3.3 中转传输模式 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/03_multi_transfer_mode/03.03_buffer_transfer_mode.ipynb&npuCnt=2) | ✅ 已发布 |
| 3.4 FabricMem 模式 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/03_multi_transfer_mode/03.04_fabricmem_mode.ipynb&npuCnt=2) | ✅ 已发布 |
| 3.5 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/03_multi_transfer_mode/03.05_chapter_practice.ipynb&npuCnt=2) | ✅ 已发布 |

### HIXL 问题定位和性能优化

| Notebook | Link | 状态 |
| --- | --- | --- |
| 4.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/04_troubleshooting_and_perf_optimization/04.01_chapter_intro.ipynb&npuCnt=2) | ✅ 已发布 |
| 4.2 HIXL 典型问题定位方法 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/04_troubleshooting_and_perf_optimization/04.02_typical_issue_troubleshoot.ipynb&npuCnt=2) | ✅ 已发布 |
| 4.3 HIXL 性能分析方法 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/04_troubleshooting_and_perf_optimization/04.03_performance_analysis.ipynb&npuCnt=2) | ✅ 已发布 |
| 4.4 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/hixl_development&scanFilePath=tutorials/hixl_development/04_troubleshooting_and_perf_optimization/04.04_chapter_practice.ipynb&npuCnt=2) | ✅ 已发布 |

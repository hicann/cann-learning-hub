# CANNBot 系列课程

CANNBot 系列课程围绕 CANNBot 算子开发展开，从入门到进阶，系统介绍如何使用 CANNBot 生成与优化 Ascend 算子。课程涵盖 Ascend C、PyPTO、TileLang-Ascend 等多种算子开发路径，并深入讲解 Vector 算子自动生成、单指令多线程算子、Harness 工程建设以及算子测试全流程，帮助开发者快速掌握 CANNBot 的核心能力。

> **学习建议**：本课程以 CANNBot 为工具介绍算子开发，重点在于"如何借助 CANNBot 高效完成算子开发与优化"，而非从零讲解算子开发原理。建议先对照下方《前置知识》补齐基础，再按 [课程内容](#课程内容) 中的前置依赖顺序学习。
>
> **零基础读者**（未接触过 AI Agent / CANNBot）：建议先学习 [零基础入门 Notebook 篇](./notebooks/README.md)，建立 Agent 与 CANNBot 的基础概念并完成首次实操后，再进入本课程。

## 前置知识

本课程面向已具备一定算子开发基础的开发者，建议具备以下前置知识：

- **Python 基础**：熟练使用 Python 编写与调试脚本
- **C/C++ 基础**：掌握指针、内存访问等基础概念，便于理解 Ascend C 算子代码
- **昇腾 NPU 基础**：了解昇腾 NPU 的 Cube / Vector / Scalar 三大计算单元及其分工
- **CANN 基础概念**：了解算子、Tiling、Kernel 等基本概念

若暂无上述基础，建议先学习以下课程补齐后再返回本课程：

- **CANN 与昇腾 NPU 基础**：参见 [Ascend C 算子开发系列教程（Kernel 直调）第 1 章](./../ascendc_operator_development_light/01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb)，涵盖人工智能与算子基础、CANN 架构与昇腾 NPU 原理
- **Ascend C 快速入门**：参见 [Ascend C 算子开发系列教程（Kernel 直调）](./../ascendc_operator_development_light/README.md)

## 快速开始：获取 CANNBot 仓库

CANNBot 的 Skills 与 Agents 统一托管在以下仓库（课件中的链接均指向该仓库）：

- 仓库地址：[cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)
- 安装说明：[CANNBot 安装指南](https://gitcode.com/cann/cannbot-skills/blob/master/docs/installation-guide.md)

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上（课程实测记录基于 CANN 9.1） |
| CLI 工具 | OpenCode / Claude Code / Codex / Trae 等任选其一（课程以 OpenCode 为例） |
| Python | 3.11 |

## 在线体验环境

本课程支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| cann-learning-hub 在线体验 notebook | cann_9.0.0_py3.11-A2-arm | Python 3.11.15 | Notebook 实操篇各册可直接打开运行（[示例：第 1 册](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/CANNBot/notebooks&scanFilePath=tutorials/CANNBot/notebooks/01_cannbot_basics.ipynb)） |
| CANNLab 云开发环境 | cann_9.0.0_py3.11-A2-arm | Python 3.11 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建环境运行 notebook；Notebook 实操篇第 2 册的 OpenCode 与 Flash 插件按该册 2.2 节四步安装 |

> **注意：** 如在本地环境体验，需自行安装配套的 CANN 软件与 CLI 工具，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)，并选择对应 CANN 版本的文档。

## 课程内容

本课程分为**Notebook 实操篇（零基础 + 进阶）**、**初级课程**与**中级课程**三部分：

- **Notebook 实操篇（×6）**：按「认知 → 体验 → 延伸 → 进阶」带你从"什么是编程 Agent"一路走到独立完成「完整版工作流开发 → 性能调优 → 协作上库」全链路；
- **初级课程（第 1～4 课）**：从零开始带你用 CANNBot 生成第一个算子，并覆盖 Ascend C、PyPTO、TileLang-Ascend 三种主流算子开发路径；
- **中级课程（第 5～9 课）**：深入算子自动生成、性能优化、多线程算子、Harness 工程建设与算子测试全流程。

### 🟢 Notebook 实操篇（零基础 ×3 + 进阶 ×3）

面向具备 NPU 基础认知但零 Agent 概念的社区用户（如高校学生），按「认知 → 体验 → 延伸 → 进阶」六册展开，交互式 Notebook 形式，含可运行演示代码与章末练习。进阶篇一个 **xlog1py 算子贯穿「完整版工作流开发 → 性能调优 → 协作上库」全链路**，设计思想优先、实录精编为证。

| 序号 | 主题 | 主要内容 | 前置 | 建议时长 | 课件 |
| :---: | :--- | :--- | :--- | :---: | :--- |
| 1 | 初识 CANNBot | 聊天 AI 与编程 Agent 的区别；Agent / Skill / 多 Agent 协作；三层架构与工作流；六仓生态全景 | 无（需 NPU 基础认知） | ~10 min | [01_cannbot_basics.ipynb](./notebooks/01_cannbot_basics.ipynb) |
| 2 | 第一个算子 | 环境自查；Flash 版四步安装；需求五要素；add_custom 端到端实操（与 [light 教程 02.04 节](./../ascendc_operator_development_light/02_AscendC_basic/02.04_introduction_to_kernel_functions_based_on_add_operator.ipynb)手动开发同规格对照）；产物走读与精度验收；sigmoid_custom 分层课后实践 | 第 1 册 | ~40 min | [02_flash_workflow.ipynb](./notebooks/02_flash_workflow.ipynb) |
| 3 | 玩转与共建 | 调试求助与性能优化；五条开发路径选择；cann-bench 评测；参与贡献的四个入口；学习路线图 | 第 2 册 | ~10 min | [03_capabilities_and_community.ipynb](./notebooks/03_capabilities_and_community.ipynb) |
| 4 | 完整版工作流 | 两套工作流的选型原理与三大设计思想（分权制衡 / 决策权留给人的三个点 / 过程产物分离）；决策者视角使用指南；xlog1py 2.5 小时实录精编（QA 门禁拦下 Inf 缺陷，47/47） | 第 1~3 册 | ~40 min | [04_complete_workflow.ipynb](./notebooks/04_complete_workflow.ipynb) |
| 5 | 性能调优 | 调优三设计思想（数据先于观点 / 分档读法 / 流量思维）与行动模板；16 用例实录精编（y_compact 流量发现 866 万倍）；需求演进与停止条件 | 第 4 册 | ~40 min | [05_performance_optimization.ipynb](./notebooks/05_performance_optimization.ipynb) |
| 6 | 协作上库 | 收束卷：最后一公里自动化与五条红线；PR 八步一图速览（POST ≠ 完成）；AI 边界框架与全课程总结；Issue→PR 课后实操 | 第 5 册 | ~10 min | [06_gitcode_collaboration.ipynb](./notebooks/06_gitcode_collaboration.ipynb) |

> 详见 [Notebook 实操课程](./notebooks/README.md)（含每册详细说明、练习答案与在线体验指引）。

### 🟢 初级课程：CANNBot基础

| 序号 | 主题 | 主要内容 | 前置 | 课件 |
| :---: | :--- | :--- | :--- | :--- |
| 1 | CANNBot 入门：从 0 到 1 生成你的第一个算子 | CANNBot 简介，从零开始生成第一个算子 | 无（具备[前置知识](#前置知识)即可） | [01_cannbot_start.pdf](./slides/01_cannbot_start.pdf) |
| 2 | CANNBot 开发进阶：Ascend C 算子开发实操 | 基于 Ascend C 的算子开发实操 | 第 1 课 | [02_ascend_c_operator.pdf](./slides/02_ascend_c_operator.pdf) |
| 3 | CANNBot 开发进阶：PyPTO 算子开发实操 | 基于 PyPTO 的算子开发实操 | 第 1 课（与第 2 课平行，任选其一） | [03_pypto_operator.pdf](./slides/03_pypto_operator.pdf) |
| 4 | CANNBot 开发进阶：TileLang-Ascend 算子开发实操 | 基于 TileLang-Ascend 的算子开发实操 | 第 1 课（与第 2 课平行，任选其一） | [04_tilelang_operator.pdf](./slides/04_tilelang_operator.pdf) |

### 🟡 中级课程：CANNBot算子开发进阶

| 序号 | 主题 | 主要内容 | 前置 | 课件 |
| :---: | :--- | :--- | :--- | :--- |
| 5 | CANNBot 进阶开发：自动生成 Vector 算子之 RegBase | 自动生成 Vector 算子，RegBase 机制详解 | 第 2 课（需先理解 Ascend C 手动开发流程，才能理解自动生成原理） | [05_vector_regbase.pdf](./slides/05_vector_regbase.pdf) |
| 6 | CANNBot 进阶开发：Vector 算子之排序性能优化 | Vector 算子排序性能优化方法 | 第 5 课 | [06_vector_sort_opt.pdf](./slides/06_vector_sort_opt.pdf) |
| 7 | CANNBot 支持生成单指令多线程算子 | 单指令多线程算子的生成 | 第 2 课 | [07_multi_thread_operator.pdf](./slides/07_multi_thread_operator.pdf) |
| 8 | CANNBot 算子 Harness 工程建设 | 算子 Harness 工程的建设与实践 | 第 2～4 课（至少掌握一种开发路径） | [08_harness_engineering.pdf](./slides/08_harness_engineering.pdf) |
| 9 | CANNBot 算子测试全流程 | 算子测试的完整流程 | 第 8 课 | [09_operator_testing.pdf](./slides/09_operator_testing.pdf) |

## 相关课程

本课程覆盖 Ascend C、PyPTO、TileLang-Ascend 三种算子开发路径，但侧重点是"用 CANNBot 生成与优化算子"。若想在具体技术路线上深入系统学习，可结合以下课程：

| 学习方向 | 建议课程 | 与本课程的关系 |
| :--- | :--- | :--- |
| Ascend C 手动开发 | [Ascend C 算子开发系列教程](./../ascendc_operator_development/README.md) | 本课程第 2、5、6、7 课涉及 Ascend C / Vector 算子，可在此系统学习 SIMD 编程模型与手动开发流程 |
| Ascend C 快速入门 | [Ascend C 算子开发系列教程（Kernel 直调）](./../ascendc_operator_development_light/README.md) | 面向零基础的前置课程，覆盖 CANN / NPU 基础与 Ascend C 入门 |
| PyPTO 开发 | [PyPTO 算子开发系列教程](./../pypto_development/README.md) | 本课程第 3 课介绍 PyPTO 开发，可在此深入学习 PyPTO 编程范式与算子实践 |
| Vector 算子 | [Ascend C 算子开发系列教程第 3 章](./../ascendc_operator_development/03_intermediate_vector_operator_development/README.md) | 本课程第 5、6 课涉及 Vector 算子的 SIMD 与 RegBase，可在此了解更详细的硬件架构与编程细节 |

> **学习路径建议**：零 NPU 基础 → [Ascend C 算子开发系列教程（Kernel 直调）](./../ascendc_operator_development_light/README.md)第 1 章 → [Notebook 实操篇·零基础](./notebooks/README.md)（1~3 册）→ [Notebook 实操篇·进阶](./notebooks/README.md)（4~6 册）→ 本课程第 1～2 课 → 按需深入对应相关课程。

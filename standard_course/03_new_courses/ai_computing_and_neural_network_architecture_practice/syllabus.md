# 《AI 计算与神经网络计算架构实践》课程大纲

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | AI 计算与神经网络计算架构实践 |
| **课程类型** | 高校学分课·实践型（专业选修，16 周） |
| **学时安排** | **64 学时**（理论 28 + 实践 36）：第一部分 20 学时（第 1~5 周）＋ 第二部分 36 学时（第 6~14 周）＋ 第三部分 8 学时（第 15~16 周）；第一、二部分每讲 2h 理论 + 2h 实践，第三部分为纯项目实践 |
| **授课对象** | 计算机/人工智能相关专业本科高年级、研究生 |
| **先修要求** | Python/C/C++ 基础；线性代数基础；了解深度学习基本概念 |
| **配套讲义** | 第一部分 7 个 PPT；第二部分双路线（A2/A3 8 个、Ascend 950 12 个；第三部分结业大作业 1 份 |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测） |
| **结业标准** | 通过各讲 CANNJudge 判题（至少通过基础判题）；提交结业大作业（作业 1 算子开发 ＋ 作业 2 模型集成替换与实训报告）并通过答辩；作业 3 开源贡献为加分项 |
| **双架构路线** | 路线 A（Atlas A2/A3）与路线 B（Ascend 950）二选一或并行开课，每讲学时与内容对应 |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握昇腾生态与 CANN 分层架构；② 掌握大模型训练（预训练/SFT/RL）、部署推理与调试调优的核心方法与流程；③ 系统掌握 Ascend C SIMD/SIMT 编程模型（核函数、多级内存、同步机制、多层级编程 API：C API/Tensor API/基础 API）；④ 掌握算子调试调优方法论、PyTorch 接入与图模式集成流程 |
| **能力目标** | ① 独立开发 Memory 矢量（路线 A）/Reg 矢量（路线 B）、矩阵、融合算子并完成调试调优；② 实现 SIMD/SIMT 混合编程与性能优化（路线 B）；③ 完成大模型训推性能瓶颈分析并输出优化报告；④ 基于 Agent（CANN Bot）辅助算子开发与优化；⑤ 通过 CANNJudge 泛化算子判题 |
| **素养目标** | 形成「架构-编程-调优」系统观与「测量→分析→优化→验证」性能闭环习惯；具备 CANN embodied AI SIG 开源协作意识与工程汇报能力 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **讲得透**：CANN 全栈架构、大模型训推链路、Ascend C 编程模型与矢量/矩阵/融合算子的实现原理
- **写得出**：Memory 矢量（路线 A）/Reg 矢量（路线 B）、矩阵、融合算子独立开发并通过 CANNJudge 判题
- **调到优**：运用 profiling 与仿真工具定位瓶颈，借鉴最佳实践逐步优化并产出数据报告
- **接得进**：算子接入 PyTorch 单算子调用与 AclGraph/GE 图
- **用得好**：借助 CANN Bot 加速开发与问题排查，AI 生成代码必经人工验证
- **融得入**：通过 embodied AI SIG 社区实战任务参与开源协作（提交 PR 并被合并）

### 2.3 课程内容规划

| 模块 | 讲次 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| AI 计算导论与大模型训推 | 第 1~5 讲 | 昇腾生态与 CANN 架构；大模型训练（预训练/SFT/RL）；部署与推理（Qwen3-1.7B）；调试调优与最佳实践（入图/PD 分离/KV 池化/自动融合） | Qwen3 训推全链路跑通 ＋ 性能分析作业 |
| Ascend C 算子编程（双路线） | 第 6~14 讲 | A2/A3：编程模型→Memory 矢量→矩阵→融合→调试调优→PyTorch 单算子→图模式→CANN Bot；950：编程模型→Reg 矢量→矩阵（Tensor API）→融合＋SIMT→调试调优→优化案例→单算子＋入图→SIMD&SIMT 高级→CANN Bot | 矢量/矩阵/融合算子独立实现 ＋ 每讲判题 |
| 算子集成与端到端优化 | 第 15~16 讲 | AddRmsNorm ＋ QuantMatmul 开发（支持 Qwen3）→ Qwen3 算子替换与实训报告 → 开源贡献 | 结业大作业三项 |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 编程语言 | Python / C / C++ **较熟练**（指针与内存管理、结构体、模板基础；算子开发以 C++ 为主） |
| 系统操作 | Linux 命令行基础（目录/文件/权限/环境变量） |
| 数学基础 | 线性代数（矩阵运算、分块乘法思想）；了解深度学习基本概念 |

### 3.2 领域前置（建议完成）

- 课前预习 cann-learning-hub 的 `quick_start/cann_basics` 章节（AI 基础概念、NPU 硬件架构、CANN 软件栈）；
- PyTorch 基础（张量操作、模型定义与 forward 流程，第 2~5 讲、第 11~12 讲直接涉及）；
- 矩阵乘法维度规则与分块思想（第 8 讲直接涉及）。

### 3.3 环境前置

- **在线环境（推荐）**：CANNLab 云端开发环境；训推实践需对应系列环境；
- **本地环境（可选）**：已部署 CANN 的昇腾开发环境（Atlas A2/A3 或 Ascend 950 系列）；无硬件时可先用仿真运行验证代码正确性；
- **社区账号（第 15~16 周开源贡献需要）**：注册 GitCode 账号并完成 SSH/HTTPS 配置，用于提交 PR。

---

## 四、教学安排与理论讲义（PPT）（16 周，每讲 2h 理论 + 2h 实践）

> 讲义题目与课程大纲中课程内容保持一致，「教材 PPT」列为相对本 syllabus 的讲义文件链接；第一、三部分讲义分别位于 `part1_introduction_to_ai_computing_and_llm_training_inference_practice_20h/`、`part3_llm_operator_development_and_inference_optimization_practice_8h/`；第二部分（Ascend C 算子编程）讲义已统一迁至原子课程素材目录 [`../../00_atomic_courses/04_ops_programming/01_ascendc/`](../../00_atomic_courses/04_ops_programming/01_ascendc/)（含 a2a3/ 与 ascend950/ 两个子目录）。路线 A（Atlas A2/A3）与路线 B（Ascend 950）二选一或并行开课。时长标注「2h+2h」指理论 2 学时 ＋ 配套实践 2 学时。

### 4.1 第一部分：AI 计算导论与大模型训练和推理实践（20 学时，第 1~5 周）

| 周次 | 讲次 | 主题 | 内容要点 | 教材 PPT |
|------|------|------|---------|---------|
| 1 | 第 1 讲 | 昇腾 AI 产业生态与 CANN 架构基础 | AI 基础概念概述；CANN 软件栈与架构体系认知；PyTorch NPU 开发快速入门 | 2h+2h, [1_artificial_intelligence_basics.pptx](./part1_introduction_to_ai_computing_and_llm_training_inference_practice_20h/1_artificial_intelligence_basics.pptx) |
| 2 | 第 2 讲 | 基于 CANN 如何训练大模型 | 模型训练基础概念；预训练、有监督微调（SFT）与强化学习（RL）的核心区别；基于 CANN 的大模型基础训练方法与流程 | 2h+2h, [2_llm_training_with_cann.pptx](./part1_introduction_to_ai_computing_and_llm_training_inference_practice_20h/2_llm_training_with_cann.pptx) |
| 3 | 第 3 讲 | 基于 CANN 如何部署和推理大模型 | 模型推理的基本概念与应用场景；掌握大模型基础部署与推理的实现方法 | 2h+2h, [3_llm_deployment_and_inference_with_cann.pptx](./part1_introduction_to_ai_computing_and_llm_training_inference_practice_20h/3_llm_deployment_and_inference_with_cann.pptx) |
| 4 | 第 4 讲 | 基于 CANN 的大模型训推优化实践（1）：概述与入图推理优化 | 训推优化全景与方法论；算子入图原理、入图推理优化实践、性能对比 | 2h+2h, [4_1 训推优化概述](./part1_introduction_to_ai_computing_and_llm_training_inference_practice_20h/4_1_llm_inference_optimization_overview_with_cann.pptx)、[4_2 入图推理优化](./part1_introduction_to_ai_computing_and_llm_training_inference_practice_20h/4_2_llm_inference_optimization_via_operator_graph_integration.pptx) |
| 5 | 第 5 讲 | 基于 CANN 的大模型训推优化实践（2）：PD 分离 & KV 池化 & 算子自动融合训练 | PD 分离架构；KV 池化原理与推理优化实战；算子自动融合原理与训练实践 | 2h+2h, [4_3 PD 分离 & KV 池化](./part1_introduction_to_ai_computing_and_llm_training_inference_practice_20h/4_3_llm_inference_optimization_practice_with_cann_pd_disaggregation_and_kv_pooling.pptx)、[4_4 自动融合训练](./part1_introduction_to_ai_computing_and_llm_training_inference_practice_20h/4_4_llm_training_optimization_practice_with_cann_automatic_operator_fusion.pptx) |

### 4.2 第二部分·路线 A（Atlas A2/A3 系列）：Ascend C 算子编程模型与实践（36 学时，第 6~14 周）

| 周次 | 讲次 | 主题 | 内容要点 | 教材 PPT |
|------|------|------|---------|---------|
| 6 | 第 6 讲 | 异构计算与算子编程导论、Ascend C SIMD 算子编程模型 | Ascend C 编程语言定位与核心特性；异构计算系统组成与基本交互流程；编程模型核心要素（核函数定义、多级内存管理、同步机制、算子分类与多层级编程 API）；编译（含仿真运行）与执行全流程；SIMD 算子开发快速入门 | 2h+2h, [01 导论](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/01_a2a3_heterogeneous_computing_and_ascend_c_operator_programming_introduction.pptx)、[02 SIMD 编程模型](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/02_a2a3_ascend_c_simd_programming_model.pptx) |
| 7 | 第 7 讲 | Ascend C SIMD Memory 矢量编程 | Memory 矢量编程的核心概念（UB 缓冲区、队列与数据搬运）；掩码与尾块处理；基于 Memory 矢量 API 开发算子的基本方法 | 2h+2h, [03 Memory 矢量编程](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/03_a2a3_ascend_c_simd_memory_vector_operator_programming.pptx) |
| 8 | 第 8 讲 | Ascend C SIMD 矩阵编程 | 矩阵编程的核心概念与应用场景（Cube 单元、矩阵分块 Tiling、数据布局）；GEMM 完整开发流程 | 2h+2h, [04 矩阵编程](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/04_a2a3_ascend_c_simd_matrix_operator_programming.pptx) |
| 9 | 第 9 讲 | Ascend C SIMD 融合算子编程 | 算子融合的核心思想与基本概念；手工/自动 AIC/AIV 分离式编程的实现方法 | 2h+2h, [05 融合算子编程](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/05_a2a3_ascend_c_simd_fused_operator_programming.pptx) |
| 10 | 第 10 讲 | Ascend C 算子调试调优与最佳实践 | 算子常见功能调试方法与问题定位思路；性能采集与分析工具的使用；算子性能调优的基本方法论；SIMD 算子常见性能问题的调试与优化 | 2h+2h, [06 调试调优与最佳实践](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/06_a2a3_ascend_c_operator_debugging_tuning_and_best_practices.pptx) |
| 11 | 第 11 讲 | Ascend C PyTorch 单算子调用与实践 | PyTorch 单算子调用的基本原理与方法；算子注册与绑定；调用验证与测试 | 2h+2h, [07 PyTorch 单算子调用](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/07_a2a3_ascend_c_operator_pytorch_single_operator_call.pptx) |
| 12 | 第 12 讲 | Ascend C 接入 GE、AclGraph、PyTorch 图模式实践 | GE、AclGraph、PyTorch 图模式的核心概念；将 Ascend C 算子接入图模式的完整流程；端到端验证 | 2h+2h, [08 图模式接入](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/08_a2a3_ascend_c_operator_graph_integration_pytorch_aclgraph_ge.pptx) |
| 13 | 第 13 讲 | Ascend C 算子优化实践 | Ascend C算子性能优化实践 | 2h+2h, [Ascend C 算子优化实践] (待补充) |
| 14 | 第 14 讲 | 基于 CANN Bot 的 Ascend C 算子开发与实践 | CANN Bot 的基本概念与核心能力；基于 Agent 的算子自动化开发与优化方法 | 2h+2h, [CANN Bot](../../01_bootcamp/two_days_course/05_cannbot_highlights_open_source_community_edition_0.5h.pptx)（暂借用启航营讲义） |

### 4.3 第二部分·路线 B（Ascend 950 系列）：Ascend C 算子编程模型与实践（36 学时，第 6~14 周）

| 周次 | 讲次 | 主题 | 内容要点 | 教材 PPT |
|------|------|------|---------|---------|
| 6 | 第 6 讲 | 异构计算与算子编程导论、Ascend C SIMD 算子编程模型 | 同路线 A 第 6 讲；SIMD & SIMT 双执行模型概述 | 2h+2h, [01 导论](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_01_heterogeneous_computing_and_ascend_c_operator_programming_introduction.pptx)、[02_1 编程模型](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_02_ascend_c_simd_programming_1_programming_model.pptx) |
| 7 | 第 7 讲 | Ascend C SIMD Reg 矢量编程 | Reg 矢量编程的核心概念与优势（寄存器级计算）；基于 C API、Tensor API 及基础 API 开发 Reg 矢量算子的基本方法 | 2h+2h, [02_2 Reg 矢量](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_02_ascend_c_simd_programming_2_reg_vector_operator.pptx) |
| 8 | 第 8 讲 | Ascend C SIMD 矩阵编程（Tensor API） | 基于 Tensor API 的矩阵算子开发；矩阵分块与数据布局；GEMM 实现流程 | 2h+2h, [02_3 矩阵算子](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_02_ascend_c_simd_programming_3_matrix_operator.pptx) |
| 9 | 第 9 讲 | Ascend C SIMD 融合算子编程、Ascend C SIMT 编程模型 | 算子融合的核心思想与实现方法；SIMT 执行模型与线程级编程；SIMD/SIMT 适用场景对比 | 2h+2h, [02_4 融合算子](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_02_ascend_c_simd_programming_4_fused_operator.pptx)、[03 SIMT 编程](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_03_ascend_c_simt_programming.pptx) |
| 10 | 第 10 讲 | Ascend C 算子调试调优与最佳实践 | 算子常见功能调试方法与问题定位思路；性能采集与分析工具的使用；SIMD&SIMT 算子常见性能问题的调试与优化 | 2h+2h, [04 调试调优与最佳实践](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_04_ascend_c_operator_debugging_tuning_and_best_practices.pptx) |
| 11 | 第 11 讲 | Ascend C SIMD&SIMT 算子优化案例 | 典型 SIMD 算子性能优化案例解析；SIMD/SIMT 混合算子优化思路与实战；算子性能优化的通用方法与实践技巧 | 2h+2h, [05 优化案例](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_05_ascend_c_simd_simt_typical_operator_optimization_cases.pptx) |
| 12 | 第 12 讲 | Ascend C PyTorch 单算子调用与接入 GE、AclGraph、PyTorch 图模式理论与实践 | PyTorch 单算子调用的基本原理与方法；GE、AclGraph、PyTorch 图模式接入完整流程；端到端验证 | 2h+2h, [06 PyTorch 单算子调用](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_06_ascend_c_operator_pytorch_single_operator_call.pptx)、[09 图模式接入](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_09_ascend_c_operator_graph_integration_pytorch_aclgraph_ge.pptx) |
| 13 | 第 13 讲 | Ascend C SIMD 与 SIMT 高级编程 | Ascend C 高级编程特性与适用场景；SIMD/SIMT 混合编程的典型方法与实现技巧 | 2h+2h, [07 SIMD&SIMT 高级编程](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_07_ascend_c_simd_simt_advanced_programming.pptx) |
| 14 | 第 14 讲 | 基于 CANN Bot 的 Ascend C 算子开发与实践 | 同路线 A 第 14 讲 | 2h+2h, [CANN Bot](../../01_bootcamp/two_days_course/05_cannbot_highlights_open_source_community_edition_0.5h.pptx)（暂借用启航营讲义） |

### 4.4 第三部分：大模型算子开发与推理优化实践（8 学时，第 15~16 周）

| 周次 | 讲次 | 主题 | 内容要点 | 教材 PPT |
|------|------|------|---------|---------|
| 15~16 | 第 15~16 讲 | 推理模型算子集成与端到端优化实践 | 独立完成至少 1 个矢量算子与 1 个矩阵算子的开发，并集成至推理模型中开展端到端优化实践；结合 CANN embodied AI SIG 社区实战任务，参与开源贡献 | 项目实践 8h, [final_project.pptx](./part3_llm_operator_development_and_inference_optimization_practice_8h/final_project.pptx) |

---

## 五、每节课的实践内容（CANN-Learning-Hub 承载）

> 每讲实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，与理论课配对，按「跟练 → 变体改造 → 独立实现 → 判题」推进；「实践路径」列为教程仓（本仓库）`quick_start/`、`tutorials/` 下的实际目录，进入对应目录即可跟练，路线 B 专属实践路径待补充。

### 5.1 第一部分：AI 计算导论与大模型训推

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 1 讲 | NPU 环境初体验 | **易**：`cann_basics` 4 个 Notebook 跟练；**中**：MindStudio 复跑同教程并对比；**难**：手绘 CANN 四层架构图并标注本课涉及层 | `quick_start/cann_basics/` |
| 第 2 讲 | 大模型训练体验 | **易**：SFT 章节跟练；**中**：换数据集完成微调；**难**：对比预训练/SFT/RL 资源占用并出报告 | `tutorials/sft_training_pipeline`、`tutorials/rl_training_pipeline` |
| 第 3 讲 | 大模型基线推理 | **易**：baseline 推理跑通；**中**：换模型完成推理与精度校验；**难**：ATC 离线编译并对比在线推理延迟 | `tutorials/llm_inference/qwen3_1.7B` |
| 第 4~5 讲 | 调试调优实践 | **易**：Profiling 章节跟练；**中**：定位 Top3 算子并解释瓶颈类型（计算/访存）；**难**：PD 分离 & KV 池化 A/B 实测并写优化报告 | `tutorials/llm_inference/qwen3_1.7B` |

### 5.2 第二部分·路线 A（Atlas A2/A3）

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 6 讲 | Ascend C 快速入门 | **易**：导论篇跟练；**中**：改写示例核函数参数并仿真运行；**难**：跑通编译全流程并解释各阶段产物 | `tutorials/ascendc_operator_development_light/01_basic_overview` |
| 第 7 讲 | Memory 矢量编程 | **易**：矢量算子跟练（Add）；**中**：Add 改 Softmax/减法变体；**难**：Add 泛化算子判题（任意 data_len，含尾块掩码处理） | `tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 第 8 讲 | 矩阵算子编程 | **易**：Matmul basic 跟练；**中**：修改 Tiling 分块参数对比性能；**难**：Matmul 判题（性能达基线 x%） | `tutorials/ascendc_operator_development_light/03_simple_operator_practice` |
| 第 9 讲 | 融合算子编程 | **易**：融合算子开发跟练；**中**：手工 AIC/AIV 分离改造；**难**：Matmul+LeakyRelu 融合并实测访存收益 | `tutorials/ascendc_operator_development_light/03_simple_operator_practice` |
| 第 10 讲 | 调试调优实战 | **易**：Profile/仿真调优跟练；**中**：解读剖析报告并指出瓶颈；**难**：慢算子诊断并优化达标（附前后数据） | `tutorials/ascendc_operator_development_light/04_debug` |
| 第 11 讲 | PyTorch 单算子调用 | **易**：单算子调用跟练；**中**：自定义算子注册调用；**难**：调用精度/性能双校验报告 | `tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 第 12 讲 | 图模式接入 | **易**：入图流程跟练；**中**：自定义算子入图验证；**难**：算子入图（PyTorch-AclGraph-GE）并端到端验证 | `tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 第 13 讲 | Ascend C算子优化实践 | 待补充 | `tutorials/ascendc_operator_development` |
| 第 14 讲 | CANN Bot 智能开发 | **易**：CANNBot 教程跟练；**中**：Agent 生成算子并人工校验；**难**：生成-校验-优化闭环完成一道判题 | `tutorials/CANNBot` |

### 5.3 第二部分·路线 B（Ascend 950）

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 6 讲 | Ascend C 快速入门 | **易**：导论篇跟练；**中**：改写示例核函数参数并仿真运行；**难**：跑通编译全流程并解释各阶段产物 | 待补充 |
| 第 7 讲 | Reg 矢量编程 | **易**：Reg 矢量算子跟练（Add）；**中**：Add 改 Softmax/减法变体；**难**：Add 泛化算子判题（任意 data_len，含尾块掩码处理） | 待补充 |
| 第 8 讲 | 矩阵编程（Tensor API） | **易**：Tensor API 矩阵算子跟练（Matmul）；**中**：修改 Tiling 分块参数对比性能；**难**：Matmul 判题（性能达基线 x%） | 待补充 |
| 第 9 讲 | 融合与 SIMT 编程 | **易**：融合算子与 SIMT 篇跟练；**中**：实现 SIMT Gather 算子；**难**：Matmul+LeakyRelu 融合判题 | 待补充 |
| 第 10 讲 | 调试调优实战 | **易**：Profile/仿真调优跟练；**中**：解读剖析报告并指出瓶颈；**难**：慢算子诊断并优化达标（附前后数据） | 待补充 |
| 第 11 讲 | SIMD&SIMT 优化案例 | **易**：优化案例跟练；**中**：复现一个 SIMD/SIMT 混合优化；**难**：自选算子做极限优化进班级排行 | 待补充 |
| 第 12 讲 | 单算子调用与入图 | **易**：单算子调用/入图跟练；**中**：自定义算子注册调用并入图验证；**难**：算子入图（PyTorch-AclGraph-GE）并端到端验证 | 待补充 |
| 第 13 讲 | SIMD 与 SIMT 高级编程 | **易**：混合编程跟练；**中**：SIMD 改 SIMT 同题实现；**难**：离散访存算子（Gather/Scatter 类）优化 | 待补充 |
| 第 14 讲 | CANN Bot 智能开发 | **易**：CANNBot 教程跟练；**中**：Agent 生成算子并人工校验；**难**：生成-校验-优化闭环完成一道判题 | `tutorials/CANNBot` |

### 5.4 第三部分：端到端优化实践

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 15~16 讲 | 结业大作业 | **易**：完成 AddRmsNorm（矢量）+ QuantMatmul（矩阵）算子开发，支持 Qwen3 功能跑通；**中**：替换 Qwen3 模型中原算子并验证正确性，撰写实训报告；**难**：端到端性能优化 ＋ embodied AI SIG 社区实战任务/开源贡献（PR 被合并） | —（判题与作业由 CANNJudge 承载，见第六章） |

---

## 六、课后习题与作业（CANNJudge 承载）

> 课后习题用于每讲结束后的即时巩固（在线题库 / 判题），小作业用于实践产出验收（判题型，易=跟练跑通 / 中=变体改造 / 难=独立实现）；结业大作业为课程最终产出。**当前各讲课后习题、小作业及结业判题作业的 CANNJudge 题目链接均缺失**，待平台上线后补充（各讲缺失状态见 6.1「CANNJudge 链接」列）。

### 6.1 每讲课后习题与小作业

> 「作业参考资源」列基于各讲内容标注具体参考来源，**并非全部位于同一仓库**：无前缀相对路径位于 CANN asc-devkit 仓 `examples/` 目录（<https://gitcode.com/cann/asc-devkit/tree/master/examples>，主要为算子开发类作业）；`learning-hub:` 前缀路径位于 cann-learning-hub 教程仓（<https://gitcode.com/cann/cann-learning-hub/>，主要为训推、工具链与教程类作业）；无参考资源的作业标 `—`；第 6~14 讲参考资源中 **A = 路线 A（Atlas A2/A3），B = 路线 B（Ascend 950）**。

| 讲次 | 课后习题（CANNJudge 在线题库） | 小作业（CANNJudge 判题） | 作业参考资源 | CANNJudge 链接 |
|------|------------------------------|------------------------|-------------|---------------|
| 第 1 讲 | 昇腾生态与 CANN 分层架构概念题 | PyTorch NPU 入门实践练习 | learning-hub: `quick_start/cann_basics/` | 待补充 |
| 第 2 讲 | 预训练/SFT/RL 核心区别概念题 | 复用 SFT checkpoint，完成一轮可解释的 Wordle GRPO 短跑 | learning-hub: `tutorials/sft_training_pipeline`、`tutorials/rl_training_pipeline` | 待补充 |
| 第 3 讲 | CANN 部署推理流程概念题 | 基于 CANNLab 部署 Qwen3-1.7B 模型 | learning-hub: `tutorials/llm_inference/qwen3_1.7B` | 待补充 |
| 第 4~5 讲 | 推理优化手段配对题（场景 → 优化手段） | 基于 Qwen3 模型完成初步的性能分析 | learning-hub: `tutorials/llm_inference/qwen3_1.7B` | 待补充 |
| 第 6 讲 | Ascend C 算子快速入门 | SIMD Hello World & Add 算子快速入门 | A：`01_simd_cpp_api/00_introduction`；B：`02_simd_c_api/00_introduction` | 待补充 |
| 第 7 讲 | Ascend C 矢量算子编程 | Add 算子、Softmax 算子判题 | A：`01_simd_cpp_api/00_introduction`；B：`01_simd_cpp_api/07_tensor_api/experimental/reg_vector_compute` | 待补充 |
| 第 8 讲 | Ascend C 矩阵算子编程 | Matmul 算子判题 | A：`01_simd_cpp_api/00_introduction`；B：`01_simd_cpp_api/07_tensor_api` | 待补充 |
| 第 9 讲 | Ascend C 融合算子编程 | Matmul+LeakyRelu 算子判题 | A：`01_simd_cpp_api/00_introduction/03_fusion_operation/matmul_leakyrelu_basic_api`；B：`01_simd_cpp_api/07_tensor_api`、`03_simt_api` | 待补充 |
| 第 10 讲 | 调试工具链操作题（Profile/仿真） | Profile 使用方法、仿真性能统计方法、SIMD 算子最佳实践 | A/B：`01_simd_cpp_api/01_utilities/04_profiling`、`01_simd_cpp_api/01_utilities/08_simulator`、`01_simd_cpp_api/05_best_practices` | 待补充 |
| 第 11 讲 | Ascend C算子调用 | PyTorch 单算子调用实践 | A：`01_simd_cpp_api/02_features/00_framework/00_pytorch`；B：待补充 | 待补充 |
| 第 12 讲 | AclGraph/GE 入图流程题 | Ascend C 算子入图实践 | A/B：`01_simd_cpp_api/02_features/00_framework/00_pytorch`、`04_aclgraph`、`03_ge` | 待补充 |
| 第 13 讲 | Ascend C 算子优化实践 | 待补充 | 待补充| 待补充 |
| 第 14 讲 | CANN Bot 功能概念题 | 基于 CANN Bot 开发 AddRmsNorm | A/B：learning-hub: `tutorials/CANNBot` | 待补充 |
| 第 15~16 讲 | —（结业冲刺） | —（进入结业大作业） | — | — |

> **路线 B（Ascend 950）小作业差异**：第 9 讲为「Matmul+LeakyRelu、SIMT Gather 算子」；第 10 讲为「Profile 使用方法、仿真性能统计方法、SIMD&SIMT 算子最佳实践」；第 12 讲为「PyTorch 单算子调用实践、Ascend C 算子入图实践」；第 13 讲为「Gather 与 Add 混合」；第 11 讲待补充。

### 6.2 结业大作业（三项）

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发矢量算子 **AddRmsNorm**、矩阵算子 **QuantMatmul**，支持 Qwen3 模型，功能跑通 | CANNJudge 判题通过（正确性） |
| **作业 2** | **必选** | 将新开发的 AddRmsNorm、QuantMatmul 算子**替换 Qwen3 模型中原来算子**，提交实训报告（含：任务概述、环境配置、算子开发过程与模型接入过程、推理结果、总结反思） | 整网集成端到端跑通 ＋ 实训报告质量 |
| **作业 3** | **开源贡献** | 向开源仓（如 **CANN Embodied-AI SIG 仓**）提交 PR 并被合并 | PR 提交与合并记录 |

---

## 七、学习建议

### 7.1 学习节奏

- **第一部分（第 1~5 讲）**：建立「训练→推理→调优」问题域，重点理解大模型训推全流程与性能优化手段，每周按时完成 cann-learning-hub 练习与 CANNJudge 课后作业
- **第二部分（第 6~14 讲）**：建立「算子开发」能力域，按讲次递进：导论→矢量→矩阵→融合→调优→框架接入→工程化→Agent；建议教程跟练（learning-hub）与判题作业（CANNJudge）同步推进
- **第三部分（第 15~16 讲）**：汇合为「算子开发 ＋ 模型集成替换 ＋ 端到端优化」项目域，将前两部分能力融会贯通

### 7.2 成功要素

1. **双路线按需选择**：初次学习 Ascend C 建议路线 A（A2/A3，资料完善、门槛较低）；有 SIMD 基础或希望学习最新架构特性建议路线 B（Ascend 950，支持 SIMD/SIMT 双模式）；学有余力者可对比两条路线差异，加深对架构演进的理解
2. **跟练与判题同步**：教程练手（会不会）→ 判题独立实现（对不对、快不快），不要只跟练不判题
3. **结业项目尽早启动**：建议第 10 周启动选题，第 12 周前完成两个算子开发与功能跑通（作业 1）→ 第 13~14 周完成 Qwen3 算子替换与端到端验证（作业 2）→ 第 15~16 周完善实训报告并争取开源贡献（作业 3）
4. **善用调试工具与仿真**：无硬件时可用仿真运行验证代码正确性，性能数据需在真实硬件上获取
5. **AI 辅助开发必须人工验证**：CANN Bot 可用于代码生成、错误分析、优化建议，但所有 AI 生成代码必须人工验证正确性与性能

### 7.3 后续学习路径

完成本课程后，可根据兴趣选择深入方向：

| 方向 | 推荐路径 | 目标 |
|------|---------|------|
| **另一架构路线** | 路线 A 学员补学 950（SIMD 编程模型→Reg 矢量→矩阵/融合→SIMT→混合编程）；路线 B 学员补学 A2/A3 | 双平台算子开发能力 |
| **算子极致性能** | A2/A3 极致性能三课（L4-20~22）→ 950 极致性能（L4-23~25） | 独立开发达到理论峰值 90%+ 的高性能算子 |
| **多语言算子范式** | PyPTO（L4-31~35）→ TileLang（L4-36~40）→ PyAsc（L4-41~44） | 掌握多种算子编程范式 |
| **大模型系统** | 训推优化进阶、分布式训练与推理服务化 | 大模型系统优化能力 |

---

## 八、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| CANN 社区主站 | <https://gitcode.com/cann> | CANN 开源社区，获取全部代码与文档 |
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现（每讲作业参考资源所在仓） |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |

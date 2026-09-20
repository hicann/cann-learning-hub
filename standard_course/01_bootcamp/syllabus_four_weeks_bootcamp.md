# 《启航营（四周）》课程大纲

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 启航营（四周）：CANN 全栈算子开发实战（训推全链路 · 矢量 / 矩阵 / 融合算子 · 极致性能 · 社区实践） |
| **课程类型** | 长期深度集训营（高校暑期学校 / 企业脱产培训 / 竞赛集训） |
| **学时安排** | **20 个工作日 ≈ 120 小时**（理论约 35h + 实践约 55h + 结业大作业与社区任务约 30h；上午理论 / 下午实践为主，含多个纯实践半日与全天实践日） |
| **授课对象** | 有一定编程基础、希望系统掌握 CANN 大模型训推全链路与 Ascend C 矢量 / 矩阵 / 融合算子开发的工程师（含高校学生，无需先修其他启航营） |
| **先修要求** | Python/C/C++ 基础；了解 Makefile/GIT 操作等 |
| **配套讲义** | 部分讲次复用 [two_weeks_course/](./two_weeks_course/)（13 份 PPT）；SFT 性能优化、推理优化高级实践、融合算子、极致性能案例等专属讲义建设中 |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测） |
| **结业标准** | 独立完成 `add_rms_norm` 算子开发（QWen3-1.7B 场景）、PyTorch 整网集成与性能优化，并完成一项社区任务（提交 PR / issue 或参加算子竞赛） |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握昇腾生态与 CANN 全栈架构；② 掌握大模型训练方法（SFT + RL 基础、SFT 性能优化）；③ 掌握大模型部署推理与推理优化（最佳实践 + 高级实践）；④ 系统掌握 Ascend C SIMD 编程模型与矢量、矩阵、融合三类算子开发；⑤ 掌握功能与性能调试方法及最佳实践；⑥ 掌握算子接入 PyTorch 与 AclGraph / GE 图的全流程；⑦ 了解 CANNBot 智能开发模式与极致性能优化方法论 |
| **能力目标** | ① 独立完成矢量（Softmax）、矩阵（Matmul）、融合（Matmul+LeakyReLU）三类算子开发与判题；② 将算子接入 PyTorch 与 AclGraph / GE 图，完成端到端验证；③ 借鉴案例方法逐步优化算子性能并产出优化数据；④ 完成 `add_rms_norm` 开发 → QWen3-1.7B 整网集成 → 持续优化的完整闭环；⑤ 完成一项真实社区任务（PR / issue / 算子竞赛） |
| **素养目标** | 形成「训推认知 → 算子实现 → 框架与图接入 → 极致性能 → 社区实践」的全栈工程思维，建立「测量 → 分析 → 优化 → 验证」性能闭环习惯与开源协作意识 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **讲得透**：CANN 全栈架构、大模型训推链路、Ascend C SIMD 编程模型与三类算子的实现原理
- **写得出**：矢量、矩阵、融合三类算子的独立开发并通过 CANNJudge 判题
- **调到优**：运用功能 / 性能调试工具定位瓶颈，借鉴案例方法逐步逼近极致性能
- **接得进**：算子接入 PyTorch 单算子调用与 AclGraph / GE 图，完成端到端验证
- **用得好**：借助 CANNBot（含 skill 定制）加速开发与问题排查，AI 生成代码必经人工验证
- **融得入**：通过社区任务参与真实开源协作（PR / issue / 竞赛），具备持续学习与贡献能力

### 2.3 课程内容规划

| 模块 | 讲次 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| 大模型训推全链路 | 第 1~8 讲 | 昇腾生态与 CANN 架构；训练（SFT+RL 基础 / SFT 性能优化）；部署推理；推理优化（最佳实践 / 高级实践） | Qwen3 训练与推理全链路跑通 + 判题 |
| Ascend C 编程基础 | 第 9~12 讲 | 异构计算导论；SIMD 编程模型；Memory 矢量算子编程 | Softmax 矢量算子独立实现 |
| 矩阵与融合算子 | 第 13~16 讲 | 矩阵算子（Matmul）；融合算子（Matmul+LeakyReLU，基于基础 API） | Matmul 与融合算子独立实现 |
| 调试调优与最佳实践 | 第 17~18 讲 | 功能与性能调试概述；Ascend C 最佳实践 | 性能瓶颈定位与优化报告 |
| 框架与图接入 | 第 19~22 讲 | 算子接入 PyTorch；算子接入 AclGraph / GE 图 | 算子端到端调用 + 入图验证 |
| 智能开发 | 第 23 讲 | 基于 CANNBot 的算子开发实践（含更改 skill 优化算子） | CANNBot 辅助开发与 skill 优化记录 |
| 极致性能案例 | 第 24~26 讲 | 矢量（Add / Softmax）、Matmul、融合算子逐步实现极致性能的案例剖析 | 极致性能优化方法论 + 优化数据报告 |
| 结业项目与社区实践 | 第 27~28 讲 | `add_rms_norm` 开发、整网集成、持续优化；社区任务（PR / issue / 竞赛） | 结业大作业 + 社区贡献 |

> **与三营的递进关系**：一天营 = 矢量单算子功能跑通；两天营 = 矢量算子 + 训推认知入门（Qwen3-1.7B 场景）；两周营 = 双算子 + 框架集成（10 天紧凑版）；**四周营 = 训推全链路 + 矢量 / 矩阵 / 融合全类型算子 + AclGraph / GE 图接入 + CANNBot 专题 + 极致性能案例 + 社区实践**（20 天完整版），是启航营体系中内容最完整的一档。

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 编程语言 | Python / C / C++ **较熟练**（指针与内存管理、结构体、模板基础；算子开发以 C++ 为主） |
| 系统操作 | Linux 命令行操作（目录 / 文件 / 权限 / 进程 / 环境变量） |
| 工程工具 | Makefile / CMake 基本规则；GIT 分支与协作基本操作（第 28 讲社区任务需提交 PR） |

### 3.2 领域前置（建议完成）

- 课前预习 [cann-learning-hub 快速入门](../../quick_start) 的 `cann_basics` 章节（AI 基础概念、NPU 硬件架构、CANN 软件栈）；
- PyTorch 基础（张量操作、模型定义与 forward 流程，第 2~8 讲、第 19~22 讲直接涉及）；
- 矩阵运算基础（矩阵乘法维度规则、分块乘法思想，第 13~16 讲直接涉及）。

### 3.3 环境前置

- **在线环境（推荐）**：CANNLab 云端开发环境（cann_9.0.0 py3.11-A3-arm）；训练与推理实践需 A3 系列环境；
- **本地环境（可选）**：已部署 CANN 9.0.0+ 的昇腾开发环境（Atlas A3 训练/推理系列）；
- **社区账号（第 4 周必需）**：注册 GitCode 账号并完成 SSH / HTTPS 配置，用于社区任务提交 PR / issue。

---

## 四、教学安排与理论讲义（PPT）（20 天，上午理论 + 下午实践）

### 第 1 周（W1）：大模型训推全链路与 Ascend C 入门（D1~D5）

| 时段 | 讲次 | 主题 | 内容要点 | 教材PPT |
|------|------|------|---------|------|
| **D1 上午** | 第 1 讲 | 昇腾 AI 产业生态与 CANN 架构基础 | 昇腾生态全景；CANN 全栈分层；Atlas 产品线 | 2h + 2h, [PPT](./two_weeks_course/1_artificial_intelligence_basics.pptx) |
| **D1 下午** | 第 2 讲 | 基于 CANN 如何训练大模型：SFT + RL 基础 | SFT/RL 训练概念；torchtitan 训练流程；Qwen3 基线 | 2h + 2h, [PPT](./two_weeks_course/02_03_llm_training_with_cann_2h.pptx) |
| **D2 上午** | 第 3 讲 | 基于 CANN 的大模型训练：SFT 性能优化 | 训练性能瓶颈分析；并行与显存优化；端到端吞吐提升 | 2h, 讲义建设中 |
| **D2 下午** | 第 4 讲 | 大模型训练实践 | SFT 基线跑通 → 性能优化 → 判题 | 4h 纯实践, 共用第 2~3 讲 PPT |
| **D3 上午** | 第 5 讲 | 基于 CANN 如何部署和推理大模型 | CANN 推理工具链；ATC 模型转换；基线推理流程 | 2h + 2h, [PPT](./two_weeks_course/04_llm_deployment_and_inference_with_cann_2h.pptx) |
| **D3 下午** | 第 6 讲 | 基于 CANN 的大模型推理优化与最佳实践 | 量化 / 算子融合 / 图模式 / KV Cache；Profiling 定位 | 2h + 2h, [PPT](./two_weeks_course/05_llm_training_and_inference_optimization_overview_with_cann_2h.pptx) |
| **D4 上午** | 第 7 讲 | 基于 CANN 的大模型推理优化高级实践 | 高级量化与融合策略；推理服务化与吞吐优化 | 2h, 讲义建设中 |
| **D4 下午** | 第 8 讲 | 大模型推理实践 | 基线推理 → 优化手段验证 → 判题 | 4h 纯实践, 共用第 5~7 讲 PPT |
| **D5 上午** | 第 9 讲 | 异构计算与 Ascend C 算子编程导论 | Host/Device 异构；NPU 架构概述；Ascend C 快速入门 | 2h + 2h, [PPT](./two_weeks_course/06_a2a3_heterogeneous_computing_and_ascend_c_operator_programming_introduction.pptx) |
| **D5 下午** | 第 10 讲 | Ascend C(A2/A3) SIMD 编程模型介绍 | 核函数定义；内存层级；同步机制；算子编译与 Stream | 2h + 2h, [PPT](./two_weeks_course/07_a2a3_ascend_c_simd_programming_model.pptx) |

### 第 2 周（W2）：矢量 / 矩阵 / 融合算子与调试接入（D6~D10）

| 时段 | 讲次 | 主题 | 内容要点 | 教材PPT |
|------|------|------|---------|------|
| **D6 上午** | 第 11 讲 | Ascend C(A2/A3) Memory 矢量算子编程模型 | 数据搬运接口；矢量计算接口；Softmax 实现剖析 | 2h, [PPT](./two_weeks_course/08_a2a3_ascend_c_simd_memory_vector_operator_programming.pptx) |
| **D6 下午** | 第 12 讲 | Memory 矢量算子（Softmax）编程实践 | Softmax 跟练 → 独立实现 → 判题 | 4h 纯实践, 共用第 11 讲 PPT |
| **D7 上午** | 第 13 讲 | Ascend C(A2/A3) 矩阵算子编程实践 | Cube 单元；Matmul 高阶 API；分块与数据布局 | 2h, [PPT](./two_weeks_course/10_a2a3_ascend_c_simd_matrix_operator_programming.pptx) |
| **D7 下午** | 第 14 讲 | 矩阵算子（Matmul）编程实践 | GEMM 跟练 → 独立实现 → 判题 | 4h 纯实践, 共用第 13 讲 PPT |
| **D8 上午** | 第 15 讲 | Ascend C(A2/A3) 融合算子编程实践（**基于基础 API**） | 融合算子模式；多算子协同；Matmul+LeakyReLU 剖析 | 2h + 2h, 讲义建设中 |
| **D8 下午** | 第 16 讲 | （简单）融合算子（Matmul+LeakyReLU）编程实践 | 融合算子跟练 → 独立实现 → 判题 | 4h 纯实践, 共用第 15 讲讲义 |
| **D9 上午** | 第 17 讲 | Ascend C 功能与性能调试概述 | CPU 仿真调试；NPU 板上调试；msprof 性能分析；Profile 使用方法与仿真性能统计 | 2h + 1h, [PPT](./two_weeks_course/12_a2a3_ascend_c_operator_debugging_tuning_and_best_practices.pptx) |
| **D9 下午** | 第 18 讲 | Ascend C 最佳实践 | 常见问题定位；性能优化套路；工程规范；UB Bank 冲突案例、数据搬运 DataCopy 案例 | 1h + 2h, 共用第 17 讲 PPT |
| **D10 上午** | 第 19 讲 | Ascend C 算子如何接入 PyTorch | 算子注册；单算子调用；Softmax/Matmul 接入演示 | 2h + 1h, [PPT](./two_weeks_course/14_a2a3_ascend_c_operator_pytorch_single_operator_call.pptx) |
| **D10 下午** | 第 20 讲 | Ascend C 算子接入 PyTorch 实践（Softmax/Matmul 等算子） | 端到端集成 → 验证 → 判题 | 3h 纯实践, 共用第 19 讲 PPT |

### 第 3 周（W3）：图接入、CANNBot 与极致性能案例（D11~D15）

| 时段 | 讲次 | 主题 | 内容要点 | 教材PPT |
|------|------|------|---------|------|
| **D11 上午** | 第 21 讲 | Ascend C 算子如何接入 AclGraph、GE 图（Softmax/Matmul 等算子） | AclGraph 构图与执行；GE 图接入；入图验证 | 2h（讲授 + 跟练）, [PPT](./two_weeks_course/15_a2a3_ascend_c_operator_graph_integration_pytorch_aclgraph_ge.pptx) |
| **D11 下午** | 第 22 讲 | 算子接入 AclGraph、GE 图实践（Softmax/Matmul 等算子） | 入图端到端 → 图执行验证 → 判题 | 3h 纯实践, 共用第 21 讲 PPT |
| **D12 全天** | 第 23 讲 | 基于 CANNBot 的算子开发实践 | CANNBot 生成 / 调试 / 优化；**尝试更改 skill 优化算子** | 2h + 4h, [PPT](./two_weeks_course/16_cannbot_highlights_open_source_community_edition_0.5h.pptx) |
| **D13 全天** | 第 24 讲 | 案例分析与实践：矢量算子（Add / Softmax 等）如何逐步实现极致性能 | 优化路径案例剖析（VF 粒度 / 双发射 / 访存优化）；复现与超越 | 2h 案例 + 4h 实践, 讲义建设中 |
| **D14 全天** | 第 25 讲 | 案例分析与实践：Matmul 算子如何逐步实现极致性能 | 分块 / 数据布局 / 流水优化路径剖析；复现与超越 | 2h 案例 + 4h 实践, 讲义建设中 |
| **D15 全天** | 第 26 讲 | 案例分析与实践：融合算子（Matmul+Gelu 等）如何逐步实现极致性能 | 以 Matmul+Gelu 融合算子为例：多算子协同 / 流水 / 零拷贝优化路径剖析；复现与超越 | 2h 案例 + 4h 实践, 讲义建设中 |

### 第 4 周（W4）：结业大作业与社区实践（D16~D20）

| 时段 | 讲次 | 主题 | 内容要点 | 教材PPT |
|------|------|------|---------|------|
| **D16~D17** | 第 27 讲 | 结业大作业：AddRMSNorm 算子开发、整网集成、实践 | `add_rms_norm` 开发（作业 1）→ PyTorch QWen3-1.7B 整网集成（作业 2）→ 持续性能优化（作业 3） | 全天 × 2, [PPT](./two_weeks_course/17_a2a3_two_week_bootcamp_final_project.pptx)（结业说明参考） |
| **D18~D20** | 第 28 讲 | 社区任务实战 | 挑选任意社区任务（cann-learning-hub / cann-samples 等），提交 PR 或 issue 等；或参加算子竞赛 | 全天 × 3, —（辅导答疑） |

> **关键节点说明：**
> - **D2 / D4 / D6 / D7 / D8 下午为 4h 纯实践，D10 / D11 下午为 3h 纯实践**（统一按「CANN-LearningHub 跟练 + CANNJudge 练习」推进）；
> - **D12 为 CANNBot 全天专题**（2h 理论 + 其余时间实践，含「更改 skill 优化算子」进阶玩法）；
> - **D13~D15 为极致性能案例三部曲**（矢量 → Matmul → 融合，案例剖析 + 复现实践）；
> - **D16~D17 为结业大作业冲刺期**（作业 1~3：开发 → 集成 → 优化）；
> - **D18~D20 为社区任务实践期**（作业 4：提交 PR / issue 或参加算子竞赛）。

---

## 五、每节课的实践内容（CANN-Learning-Hub 承载）

> 每讲实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，与理论课配对；纯实践半日 / 全天按「跟练 → 独立实现 → 判题」推进。

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 1 讲 | NPU 环境初体验 | 跑通 PyTorch NPU HelloWorld，验证环境与算力 | `tutorials/02_ai_frameworks/01_pytorch_npu_quickstart/` |
| 第 2 讲 | 大模型训练体验 | Qwen3-1.7B SFT 基线训练跑通 | `tutorials/02_ai_frameworks/02_training_techniques/08_cann_sft_rl_basics/` |
| 第 3~4 讲 | SFT 性能优化实践 | SFT 性能分析与优化 → 判题（D2 下午 4h 纯实践） | `tutorials/02_ai_frameworks/02_training_techniques/09_cann_sft_rl_advanced_varlen_cp/`、`10_cann_sft_rl_expert/` |
| 第 5 讲 | 大模型基线推理 | Qwen3 部署与基线推理跑通 | `tutorials/02_ai_frameworks/03_inference_techniques/01_llm_deployment_inference_basics/` |
| 第 6 讲 | 推理优化实践 | Profiling 采集 + 量化 / 融合 / 图模式等优化手段验证 | `tutorials/02_ai_frameworks/03_inference_techniques/02_llm_inference_optimization/` |
| 第 7~8 讲 | 推理优化高级实践 | 高级优化手段验证 → 判题（D4 下午 4h 纯实践） | `tutorials/02_ai_frameworks/03_inference_techniques/08_cann_llm_inference_advanced/` |
| 第 9 讲 | Ascend C 快速入门 | Add 算子跟练跑通 | `tutorials/04_ops_programming/01_ascendc/01_introduction/` |
| 第 10 讲 | 编程模型上机 | 向量加法算子完整实现 | `tutorials/04_ops_programming/01_ascendc/02_a2a3_simd_programming_model/` |
| 第 11~12 讲 | 矢量算子编程 | Softmax 跟练 → 独立实现 → 判题（D6 下午 4h 纯实践） | `tutorials/04_ops_programming/01_ascendc/03_a2a3_simd_memory_vector/` |
| 第 13~14 讲 | 矩阵算子编程 | Matmul 跟练 → GEMM 独立实现 → 判题（D7 下午 4h 纯实践） | `tutorials/04_ops_programming/01_ascendc/04_a2a3_simd_matmul/` |
| 第 15~16 讲 | 融合算子编程 | Matmul+LeakyReLU 跟练 → 独立实现 → 判题（D8 下午 4h 纯实践） | `tutorials/04_ops_programming/01_ascendc/05_a2a3_simd_fused_operator/` |
| 第 17~18 讲 | 调试调优实战 | 仿真 / 板调 + msprof 性能分析 → 瓶颈定位与优化 | `tutorials/04_ops_programming/01_ascendc/08_a2a3_debug_tuning/` |
| 第 19~20 讲 | PyTorch 接入 | 算子注册与单算子调用 → 端到端验证（D10 下午 3h 纯实践） | `tutorials/04_ops_programming/01_ascendc/18_pytorch_single_operator_call/` |
| 第 21~22 讲 | AclGraph / GE 图接入 | 算子入图 → 图执行端到端验证（D11 下午 3h 纯实践） | `tutorials/04_ops_programming/01_ascendc/28_operator_graph_integration/` |
| 第 23 讲 | CANNBot 智能开发 | CANNBot 生成 / 调试 / 优化算子 + 更改 skill 优化算子记录 | `tutorials/04_ops_programming/05_cannbot/01_cannbot_introduction_practice/`、`02_cannbot_knowledge_base_skills/` |
| 第 24 讲 | 矢量算子极致性能 | Add / Softmax 逐步极致优化实战 + 性能数据报告 | `tutorials/04_ops_programming/01_ascendc/20_a2a3_vector_extreme_performance/` |
| 第 25 讲 | 矩阵算子极致性能 | Matmul 逐步极致优化实战 + 优化数据报告 | `tutorials/04_ops_programming/01_ascendc/21_a2a3_matmul_extreme_performance/` |
| 第 26 讲 | 融合算子极致性能 | Matmul+Gelu 等融合算子逐步极致优化实战 + 优化数据报告 | `tutorials/04_ops_programming/01_ascendc/22_a2a3_fused_extreme_performance/` |
| 第 27 讲 | 结业大作业 | `add_rms_norm` 开发 + QWen3-1.7B 整网集成 + 持续优化（判题由 CANNJudge 承载，见第六章） | — |
| 第 28 讲 | 社区任务 | 挑选任意社区任务，提交 PR / issue 等（见第六章作业 4） | —（社区仓库，如 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/)、[cann-samples](https://gitcode.com/cann/cann-samples/)） |

> **实践目录说明：** 「实践路径」列为 tutorials 规划目录（见 [00_atomic_courses/directory.md](../00_atomic_courses/directory.md)），实体目录建设完成后替换为可点击链接。

---

## 六、课后习题与作业（CANNJudge 承载）

> 课后习题用于每讲结束后的即时巩固（在线题库 / 判题），小作业用于实践产出验收（判题型，易=跟练跑通 / 中=变体改造 / 难=独立实现）；结业大作业为课程最终产出。**当前各讲课后习题、小作业及结业判题作业的 CANNJudge 题目链接均缺失**，待平台上线后补充（各讲缺失状态见 6.1「CANNJudge 链接」列）。

### 6.1 每讲课后习题与小作业

> 「作业参考资源」列基于各讲内容标注具体参考来源，**并非全部位于同一仓库**：无前缀相对路径位于 CANN asc-devkit 仓 `examples/` 目录（<https://gitcode.com/cann/asc-devkit/tree/master/examples>，主要为算子开发类作业）；`learning-hub:` 前缀路径位于 cann-learning-hub 教程仓（<https://gitcode.com/cann/cann-learning-hub/>，主要为训推、工具链与教程类作业）；无参考资源的作业标 `—`。

| 讲次 | 课后习题（CANNJudge 在线题库） | 小作业（CANNJudge 判题） | 作业参考资源 | CANNJudge 链接 |
|------|------------------------------|------------------------|-------------|---------------|
| 第 1 讲 | 昇腾生态与 CANN 分层架构概念题 | NPU 环境验证：HelloWorld + 算力信息提交 | learning-hub: `tutorials/02_ai_frameworks/01_pytorch_npu_quickstart/` | 待补充 |
| 第 2 讲 | SFT / RL 训练概念题 | Qwen3-1.7B SFT 基线跑通（训练流程执行） | learning-hub: `tutorials/02_ai_frameworks/02_training_techniques/08_cann_sft_rl_basics/` | 待补充 |
| 第 3~4 讲 | SFT 性能优化方法题 | SFT 性能优化实践 + 优化前后数据简报 | learning-hub: `tutorials/02_ai_frameworks/02_training_techniques/09_cann_sft_rl_advanced_varlen_cp/`、`10_cann_sft_rl_expert/` | 待补充 |
| 第 5 讲 | CANN 部署推理流程概念题 | Qwen3 基线推理跑通（baseline notebook） | learning-hub: `tutorials/02_ai_frameworks/03_inference_techniques/01_llm_deployment_inference_basics/` | 待补充 |
| 第 6 讲 | 推理优化手段配对题（场景 → 优化手段） | Profiling 采集 + op_statistic 瓶颈分析简报 | learning-hub: `tutorials/02_ai_frameworks/03_inference_techniques/02_llm_inference_optimization/` | 待补充 |
| 第 7~8 讲 | 高级优化手段概念题 | 推理优化手段验证 + 判题 | learning-hub: `tutorials/02_ai_frameworks/03_inference_techniques/08_cann_llm_inference_advanced/` | 待补充 |
| 第 9 讲 | 异构计算与 Ascend C 概念题 | SIMD Hello World & Add 算子快速入门判题（**易**：功能跑通） | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 10 讲 | 编程模型概念题（核函数 / 内存 / 同步） | Add 算子泛化性支持判题（任意 data_len） | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 11~12 讲 | 矢量算子编程接口题 | Softmax 独立实现判题（**中**） | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 13~14 讲 | Matmul 高阶 API 跟练 | GEMM 算子独立实现判题 | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 15~16 讲 | 融合算子模式概念题 | Matmul+LeakyReLU 融合算子独立实现判题（**难**） | `01_simd_cpp_api/00_introduction/03_fusion_operation/matmul_leakyrelu_basic_api` | 待补充 |
| 第 17 讲 | 调试工具链操作题（仿真 / 板调 / msprof） | Profile 使用方法、仿真性能统计方法实践 | `01_simd_cpp_api/01_utilities/04_profiling`、`01_simd_cpp_api/01_utilities/08_simulator` | 待补充 |
| 第 18 讲 | 最佳实践案例题（UB Bank / DataCopy） | UB Bank 冲突案例、数据搬运 DataCopy 案例分析 | `01_simd_cpp_api/05_best_practices/04_memory_access` | 待补充 |
| 第 19~20 讲 | 算子注册与调用流程题 | PyTorch 单算子调用实践：Softmax / Matmul 算子端到端判题 | `01_simd_cpp_api/02_features/00_framework/00_pytorch` | 待补充 |
| 第 21~22 讲 | AclGraph / GE 入图流程题 | Ascend C 算子入图实践：端到端验证 | `01_simd_cpp_api/02_features/00_framework/00_pytorch`、`04_aclgraph`、`03_ge` | 待补充 |
| 第 23 讲 | CANNBot 功能概念题 | CANNBot 生成算子 + 更改 skill 优化记录 | learning-hub: `tutorials/04_ops_programming/05_cannbot/01_cannbot_introduction_practice/`、`02_cannbot_knowledge_base_skills/` | 待补充 |
| 第 24 讲 | 矢量极致优化方法题 | Add / Softmax 算子性能优化报告（基线对比） | `01_simd_cpp_api/05_best_practices/00_vector_compute/add_high_performance` | 待补充 |
| 第 25 讲 | 分块 / 流水策略题 | Matmul 算子优化数据报告（基线对比） | `01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_basic_api_high_performance` | 待补充 |
| 第 26 讲 | 融合优化方法题 | Matmul+Gelu 融合算子优化数据报告（基线对比） | `01_simd_cpp_api/05_best_practices/03_fusion_compute/matmul_gelu_high_performance` | 待补充 |
| 第 27 讲 | —（结业冲刺） | —（进入结业大作业） | — | — |
| 第 28 讲 | —（社区实践） | —（作业 4 承载） | — | — |

### 6.2 结业大作业（四档，全部必选）

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发矢量算子 **`add_rms_norm`**（只需要支持 QWen3-1.7B 模型） | CANNJudge 判题通过（正确性） |
| **作业 2** | **必选** | 将 `add_rms_norm` 算子接入 **PyTorch QWen3-1.7B** | 整网集成端到端跑通，推理结果正确 |
| **作业 3** | **必选** | **持续优化** `add_rms_norm` 算子，优化推理模型性能 | 性能提升数据 + 优化路径剖析（profiling 证据 + 改动说明） |
| **作业 4** | **必选** | 挑选任意**社区任务**、或**算子竞赛** | PR / issue 提交记录，或竞赛参赛证明 |

> **与两周营衔接：** 两周营 = 双算子 + 集成 + 可选优化（10 天）；四周营在 20 天周期内完成「训推全链路 → 三类算子 → 图接入 → 极致性能」的完整训练，结业作业全部必选，并以真实社区贡献（作业 4）收尾，直接衔接开源社区与算子竞赛。

---

## 七、学习建议

### 7.1 四周学习节奏

- **W1「训推全链路 + 入门」**：训练（基础 → 性能优化）与推理（部署 → 优化 → 高级实践）两大链路交替「理论 + 纯实践」，建立大模型系统视角；D5 转入 Ascend C 编程入门
- **W2「算子三部曲 + 接入」**：矢量 → 矩阵 → 融合三类算子逐类攻克（每类均为「上午理论 + 下午纯实践」），调试调优后完成 PyTorch 接入
- **W3「图接入 + 智能开发 + 极致性能」**：算子入 AclGraph / GE 图，CANNBot 全天专题（含 skill 定制），再以三个案例日剖析极致性能优化路径
- **W4「结业 + 社区」**：D16~D17 完成 `add_rms_norm` 开发、整网集成与优化（作业 1~3），D18~D20 投入社区任务（作业 4）

### 7.2 成功要素

1. **纯实践半日是分水岭**：D2 / D4 / D6 / D7 / D8 / D10 / D11 下午的纯实践是从「听懂」到「会做」的关键，务必独立完成，不要只跟练
2. **极致性能案例要形成方法论**：D13~D15 三部曲不是看热闹，要记录每一步优化的 profiling 数据，沉淀「测量 → 分析 → 优化 → 验证」的完整闭环，直接服务于结业作业 3
3. **图接入是工程化分水岭**：算子从「单算子能跑」到「入图可用」是生产级能力的标志，D11 起的 AclGraph / GE 实践务必吃透
4. **善用 CANNBot 但不依赖**：D12 专题含「更改 skill 优化算子」进阶玩法，可用于代码生成、错误分析、优化建议，但所有 AI 生成代码必须人工验证正确性与性能
5. **社区任务尽早选题**：建议 W3 期间开始浏览社区 issue 与任务列表，D18 前确定选题，避免最后一刻仓促提交
6. **结对学习 + 代码评审**：四周周期长、内容深，建议结对互助与互相 code review

### 7.3 后续学习路径

完成四周启航营后，可根据兴趣选择深入方向：

| 方向 | 推荐路径 | 目标 |
|------|---------|------|
| **Ascend 950 算子编程** | 950 SIMD 编程模型 → Reg 矢量 → 矩阵 / 融合 → SIMT（L4-09~17）→ SIMD&SIMT 混合编程（L4-26~27） | 950 双平台算子开发能力 |
| **算子极致性能** | A2/A3 极致性能三课（L4-20~22）→ 950 极致性能（L4-23~25） | 独立开发达到理论峰值 90%+ 的高性能算子 |
| **算子工程化** | 算子入图（L4-28）→ Aclnn 工程化开发（L4-29）→ 通信算子自定义开发（L4-30） | 算子库级工程化能力 |
| **多语言算子开发** | PyPTO（L4-31~35）→ TileLang（L4-36~40）→ PyAsc（L4-41~44） | 掌握多种算子编程范式 |
| **框架与大模型系统** | L2 层训练 / 推理 / 图框架进阶课程 | 框架开发与系统优化专家 |

---

## 八、进一步学习参考

### 8.1 进阶方向（原子课程衔接）

| 进阶方向 | 对应原子课程（[目录规划](../00_atomic_courses/directory.md)） |
|---------|------------------------------------------------|
| A2/A3 矩阵算子编程 | L4-04（`01_ascendc/04_a2a3_simd_matmul/`）、L4-06（典型矩阵算子实践） |
| A2/A3 融合算子编程 | L4-05（`01_ascendc/05_a2a3_simd_fused_operator/`） |
| Ascend C 矢量、矩阵、融合算子调试调优与最佳实践 | L4-08（`01_ascendc/08_a2a3_debug_tuning/`） |
| Ascend 950 Ascend C 算子编程以及进一步性能优化 | L4-09~17（950 SIMD / SIMT 系列）、L4-23~25（950 极致性能）、L4-26~27（SIMD&SIMT 混合编程） |
| A2/A3 算子极致性能 | L4-20~22（`01_ascendc/20~22_a2a3_*_extreme_performance/`） |
| 算子工程化与多语言范式 | L4-28~30（入图 / Aclnn / 通信算子）、PyPTO L4-31~35、TileLang L4-36~40、PyAsc L4-41~44 |

### 8.2 资源链接

| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标 / 参考实现 / 社区任务选题） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（本课程实践作业载体 / 社区任务选题） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |
| 昇腾社区 | <https://www.hiascend.com/> | 官方文档、论坛、课程、活动 |
| CANN 开发者论坛 | <https://bbs.huaweicloud.com/forum/forum-1109-1.html> | 问题求助、经验分享、技术交流 |

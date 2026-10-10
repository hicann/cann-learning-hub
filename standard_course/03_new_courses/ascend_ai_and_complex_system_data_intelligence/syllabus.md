# 《昇腾 AI 与复杂系统数据智能》课程大纲

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 昇腾 AI 与复杂系统数据智能 |
| **课程类型** | 高校学分课·行业综合型（专业选修，16 周） |
| **学时安排** | **约 32 课时**（必修 26 ＋ 选修 6 按需；另结业大作业课后 8，每讲另配约 2h 跟练实践）：第一部分 4 讲（第 1~4 周）＋ 第二部分 6 讲（第 5~10 周）＋ 第三部分必修 3 讲（第 11~13 周）＋ 选修 3 讲（第 14~16 周，按需）＋ 结业大作业（第 13~16 周课后推进，期末答辩） |
| **授课对象** | 数据科学/人工智能相关专业本科高年级、研究生 |
| **先修要求** | Python 编程能力；基本机器学习/深度学习概念；了解行业数据特点 |
| **配套讲义** | 第一部分 4 个 PPT（建设中）；第二部分 8 个 PPT；第三部分必修 4 个 PPT ＋ 选修 6 个 PPT（Ascend C 950）；结业大作业 1 份 |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测） |
| **结业标准** | 通过各讲 CANNJudge 判题；从作业 1/作业 2 中任选其一提交结业大作业并通过答辩；作业 3 开源贡献为加分项 |
| **后续衔接** | 《AI 计算与神经网络计算架构实践》第二部分（36 课时完整算子体系，双架构路线） |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握昇腾生态与 CANN 软件栈架构；② 掌握 ATC 离线模型编译与推理部署的完整流程；③ 理解大模型训练与推理的核心优化技术（PD 分离、KV 池化、算子自动融合训练等）；④ 掌握 Ascend C 算子编程的基本方法与调试调优手段 |
| **能力目标** | ① 操作 NPU 环境完成模型离线编译、部署与推理验证，输出推理性能报告；② 开发基础矢量算子（Add/ReLU/LayerNorm 等），完成调试、PyTorch 单算子调用与入图验证；③ 针对数据智能场景（推荐/检索/时序等）进行推理优化实践并量化收益；④ 通过 CANNJudge 基础/进阶算子判题 |
| **素养目标** | 建立数据智能系统的「数据-模型-算力」协同思维与「测量→分析→优化→验证」性能闭环习惯；具备复杂系统性能分析能力、工程化实践素养与开源协作意识 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **看得懂**：CANN 分层架构与大模型训推全链路（ATC 编译→训练→部署推理→训推优化）
- **用得上**：ATC 离线编译、部署与推理验证工具链，独立完成模型转换与推理验证
- **跑得通**：Qwen3-1.7B 训推全流程与性能瓶颈分析，输出性能报告
- **写得出**：基础矢量算子独立开发并通过 CANNJudge 判题
- **调得动**：运用调试工具链定位算子瓶颈，按最佳实践优化
- **接得进**：算子接入 PyTorch 单算子调用与 AclGraph/GE 图，完成端到端验证
- **融得入**：向昇腾 CANN 社区提交 PR 并被合并，参与开源协作

### 2.3 课程内容规划

| 模块 | 讲次 | 核心内容 | 关键产出 | 原子课映射 |
|------|------|---------|---------|-----------|
| 复杂系统行业导论与数据智能 | 第 1~4 讲 | 复杂系统与数据智能概述；行业场景（推荐/检索/时序等）系统架构；数据-模型-算力协同；性能工程方法与课程主线 | 行业认知与结业选题方向 | — |
| 大模型训练与推理 | 第 5~10 讲 | 昇腾生态与 CANN 架构；ATC 离线模型编译与推理；大模型训练（SFT/RL）；部署与推理（Qwen3-1.7B）；训推优化实践（概述与入图推理优化、PD 分离 & KV 池化 & 自动融合训练） | Qwen3 训推全链路跑通 ＋ 每讲作业 | L1-14~21、L2-13~18 |
| Ascend C 算子编程实践（必修 ＋ 选修） | 第 11~13 讲 ＋ 选修 | 矢量算子编程概述；算子调试调优与最佳实践；PyTorch 单算子调用与入图；选修：Ascend C 950 编程模型、Reg 矢量/矩阵（Tensor API）/融合与 SIMT 算子编程 | 基础矢量算子独立实现 ＋ 判题 | L4-01~14 |
| 结业大作业 | 第 14 讲（第 13~16 周课后） | 昇腾 NPU 模型迁移与部署 或 算子开发（AddRmsNorm 或 QuantMatmul，支持 Qwen3）；开源贡献 | 结业大作业 ＋ 社区贡献 | 综合应用 |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 编程语言 | Python **较熟练**（数据处理、脚本编写；算子开发以 C++ 为主，课程内入门） |
| 系统操作 | Linux 命令行基础（目录/文件/环境变量） |
| 领域基础 | 基本机器学习/深度学习概念（训练/推理/模型评估） |

### 3.2 领域前置（建议完成）

- 课前预习 cann-learning-hub 的 [quick_start/cann_basics](../../../quick_start/cann_basics/) 章节（AI 基础概念、NPU 硬件架构、CANN 软件栈）；
- PyTorch 基础（张量操作、模型定义与 forward 流程，第 7~10、13 讲直接涉及）。

### 3.3 环境前置

- **在线环境（推荐）**：CANNLab 云端开发环境；
- **本地环境（可选）**：已部署 CANN 的昇腾开发环境（Atlas A2/A3 系列）；
- **社区账号（结业作业开源贡献需要）**：注册 GitCode 账号并完成 SSH/HTTPS 配置，用于提交 PR。

---

## 四、教学安排与理论讲义（PPT）（16 周）

> 讲义题目与课程大纲中课程内容保持一致，「教材 PPT」列为相对本 syllabus 的讲义文件链接；第二部分讲义位于 `part1_llm_training_and_inference/`、第三部分讲义位于 `part2_ascend_c_operator_programming_practice/`（目录名沿用历史编号），第一部分专属讲义建设中。时长标注「2h+2h」指理论 2 课时 ＋ 配套跟练实践约 2h，「2h」为纯理论讲授。

### 4.1 第一部分：复杂系统行业导论与数据智能（4 讲，第 1~4 周，共 8 课时）

> 本部分为行业导论模块，以理论讲授与案例研讨为主（每讲 2h），帮助学员建立行业系统认知与「数据-模型-算力」协同思维，为后续技术模块学习与结业选题铺垫；专属讲义建设中。

| 周次 | 讲次 | 主题 | 内容要点 | 教材 PPT |
|------|------|------|---------|---------|
| 1 | 第 1 讲 | 复杂系统与数据智能导论 | 复杂系统基本特征；数据智能技术范式；产业趋势与课程定位 | 2h, 讲义建设中 |
| 2 | 第 2 讲 | 行业数据智能系统架构 | 推荐/检索/时序等典型场景的系统架构、数据流与业务指标 | 2h, 讲义建设中 |
| 3 | 第 3 讲 | 数据-模型-算力协同 | AI 基础设施与算力底座；昇腾生态在行业系统中的定位 | 2h, 讲义建设中 |
| 4 | 第 4 讲 | 复杂系统性能工程与课程主线 | 「测量→分析→优化→验证」性能闭环方法；课程实践主线与结业要求解读 | 2h, 讲义建设中 |

### 4.2 第二部分：大模型训练与推理（6 讲，第 5~10 周）

| 周次 | 讲次 | 主题 | 内容要点 | 教材 PPT |
|------|------|------|---------|---------|
| 5 | 第 5 讲 | 昇腾 AI 产业生态与 CANN 架构基础 | 产业生态全景；CANN 四层架构；环境搭建与认知 | 2h+2h, [1_AI 基础](../../00_atomic_courses/02_framework/01_overview/artificial_intelligence_basics.pptx) |
| 6 | 第 6 讲 | ATC 离线模型编译与推理 | ATC 工具使用；模型转换流程；离线推理验证 | 2h+2h, [2_ATC 离线编译与推理](./part1_llm_training_and_inference/2_atc_offline_model_compilation_and_inference.pptx) |
| 7 | 第 7 讲 | 基于 CANN 如何训练大模型 | 训练基础概念与流程（预训练/SFT/RL）；基础训练方法与实践 | 2h+2h, [3_大模型训练](./part1_llm_training_and_inference/3_llm_special_topic_training_with_cann.pptx) |
| 8 | 第 8 讲 | 基于 CANN 如何部署和推理大模型 | 部署流程；推理验证；性能基准测试 | 2h+2h, [4_大模型部署与推理](./part1_llm_training_and_inference/4_llm_special_topic_deployment_and_inference_with_cann.pptx) |
| 9 | 第 9 讲 | 基于 CANN 的大模型训推优化实践（1）：概述与入图推理优化 | 训推优化全景与方法论；算子入图原理、入图推理优化实践、性能对比 | 2h+2h, [5_1 训推优化概述](./part1_llm_training_and_inference/5_1_llm_special_topic_training_and_inference_optimization_overview_with_cann.pptx)、[5_2 入图推理优化](./part1_llm_training_and_inference/5_2_llm_special_topic_inference_optimization_via_operator_graph_integration.pptx) |
| 10 | 第 10 讲 | 基于 CANN 的大模型训推优化实践（2）：PD 分离 & KV 池化 & 算子自动融合训练 | PD 分离架构；KV 池化原理与推理优化实战；算子自动融合原理与训练实践 | 2h+2h, [5_3 PD 分离 & KV 池化](./part1_llm_training_and_inference/5_3_llm_special_topic_inference_optimization_practice_with_cann_pd_disaggregation_and_kv_pooling.pptx)、[5_4 自动融合训练](./part1_llm_training_and_inference/5_4_llm_special_topic_training_optimization_practice_with_cann_automatic_operator_fusion.pptx) |

### 4.3 第三部分：Ascend C 算子编程实践（必修 3 讲，第 11~13 周；选修 3 讲，第 14~16 周按需）

**必修（算子闭环底线）：**

| 周次 | 讲次 | 主题 | 内容要点 | 教材 PPT |
|------|------|------|---------|---------|
| 11 | 第 11 讲 | Ascend C 矢量算子编程概述 | 异构计算原理；Ascend C 编程模型；核函数结构；矢量算子编程入门 | 2h+2h, [矢量算子编程导论](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3_ascend_c_simd_vector_operator_development.pptx) |
| 12 | 第 12 讲 | Ascend C 算子调试调优与最佳实践 | 调试工具链；性能分析方法；常见瓶颈与优化；最佳实践总结 | 2h+2h, [调试调优与最佳实践](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/06_a2a3_ascend_c_operator_debugging_tuning_and_best_practices.pptx) |
| 13 | 第 13 讲 | Ascend C 算子 PyTorch 单算子调用与入图 | 单算子调用原理与方法；算子入图流程（PyTorch/AclGraph/GE）；端到端验证 | 2h+2h, [PyTorch 单算子调用](../../00_atomic_courses/04_ops_programming/01_ascendc/a2a3/07_a2a3_ascend_c_operator_pytorch_single_operator_call.pptx)、[图模式接入](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_09_ascend_c_operator_graph_integration_pytorch_aclgraph_ge.pptx) |

**选修·Ascend C 950 算子编程（第 14~16 周按需选学，讲义位于原子课程素材目录 [`ascend950/`](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/)）：**

| 讲次 | 主题 | 教材 PPT |
|------|------|---------|
| 选修 1 | Ascend C 950 导论、SIMD 编程模型与 Reg 矢量编程 | 2h+2h, [01 导论](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_01_heterogeneous_computing_and_ascend_c_operator_programming_introduction.pptx)、[02_1 编程模型](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_02_ascend_c_simd_programming_1_programming_model.pptx)、[02_2 Reg 矢量](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_02_ascend_c_simd_programming_2_reg_vector_operator.pptx) |
| 选修 2 | Ascend C 950 矩阵算子编程（Tensor API） | 2h+2h, [02_3 矩阵算子](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_02_ascend_c_simd_programming_3_matrix_operator.pptx) |
| 选修 3 | Ascend C 950 融合算子编程与 SIMT 编程模型 | 2h+2h, [02_4 融合算子](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_02_ascend_c_simd_programming_4_fused_operator.pptx)、[03 SIMT 编程](../../00_atomic_courses/04_ops_programming/01_ascendc/ascend950/Ascend950_03_ascend_c_simt_programming.pptx) |

### 4.4 第四部分：结业大作业（综合实践 & 答辩）

| 讲次 | 主题 | 内容要点 | 教材 PPT |
|------|------|---------|---------|
| 第 14 讲 | 结业大作业 | 昇腾 NPU 模型迁移与部署（作业 1）或 算子开发 AddRmsNorm/QuantMatmul（作业 2）；开源贡献（作业 3） | 项目实践, [final_project.pptx](./final_project.pptx) |

---

## 五、每节课的实践内容（CANN-Learning-Hub 承载）

> 每讲实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，与理论课配对，按「跟练 → 变体改造 → 独立实现 → 判题」推进；「实践路径」列为教程仓（本仓库）`quick_start/`、`tutorials/` 下的实际目录，进入对应目录即可跟练。

### 5.1 第一部分：复杂系统行业导论与数据智能

> 本部分以理论讲授与行业案例研讨为主，不设平台跟练实践；建议学员在第 5 讲开课前完成 `quick_start/cann_basics` 章节（AI 基础概念、NPU 硬件架构、CANN 软件栈）预习，为第二部分上机实践做准备。

### 5.2 第二部分：大模型训练与推理

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 5 讲 | NPU 环境初体验 | **易**：`cann_basics` 4 Notebook 跟练；**中**：梳理昇腾产业生态图谱并标注本课定位；**难**：CANN 四层架构逐层实验验证 | `quick_start/cann_basics/` |
| 第 6 讲 | ATC 离线编译与推理 | **易**：ATC 转换一个 ONNX 模型并推理；**中**：转换参数调优（精度/性能选项）对比；**难**：离线 vs 在线推理延迟/吞吐实测报告 | `tutorials/ge_development/03_graph_compilation` |
| 第 7 讲 | 大模型训练体验 | **易**：`sft_training_pipeline/01` 跟练；**中**：换数据集微调；**难**：`rl_training_pipeline` RL 训练管线跑通与瓶颈定位 | `tutorials/sft_training_pipeline`、`tutorials/rl_training_pipeline` |
| 第 8 讲 | 大模型基线推理 | **易**：`qwen3_1.7B` baseline 推理跑通；**中**：换模型部署校验；**难**：并发推理压测与 SLO 报告 | `tutorials/llm_inference/qwen3_1.7B` |
| 第 9 讲 | 入图推理优化 | **易**：入图开关 A/B 对比；**中**：定位入图生效算子集合；**难**：手工改图 vs 自动入图性能对比 | `tutorials/llm_inference/qwen3_1.7B` |
| 第 10 讲 | PD 分离 & KV 池化 & 自动融合 | **易**：Profiling 章节跟练；**中**：PD 分离/KV 池化开关实测；**难**：自动融合收益量化与训练吞吐报告 | `tutorials/llm_inference/qwen3_1.7B` |

### 5.3 第三部分：Ascend C 算子编程实践

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 11 讲 | 矢量算子编程 | **易**：`01_basic_overview` 基础算子跟练；**中**：Add 改 ReLU/GELU 变体；**难**：Add/ReLU 泛化算子判题（任意 shape/数据类型） | `tutorials/ascendc_operator_development_light/01_basic_overview`、`tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 第 12 讲 | 调试调优实战 | **易**：调优章节跟练；**中**：剖析报告解读并指出瓶颈；**难**：慢算子诊断优化达标（附前后对比） | `tutorials/ascendc_operator_development_light/04_debug` |
| 第 13 讲 | 单算子调用与入图 | **易**：单算子调用/入图跟练；**中**：自定义算子注册调用与入图验证；**难**：矢量算子判题（含 Tiling 与性能要求） | `tutorials/ascendc_operator_development_light/02_AscendC_basic`、`tutorials/ge_development` |
| 选修 | Ascend C 950 算子编程 | **易**：950 选修章节跟练；**中**：Reg 矢量（950）与 Memory 矢量（A2/A3）同题对比；**难**：融合与 SIMT 混合算子进阶判题 | 待补充（950 专属实践路径建设中） |

### 5.4 第四部分：结业大作业

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 14 讲（第 13~16 周课后推进，期末答辩） | 结业大作业 | **易**：完成大作业基线（模型迁移或算子开发）；**中**：算子融入推理模型并验证；**难**：向昇腾 CANN 社区提交 PR 并被合并 | —（判题与作业由 CANNJudge 承载，见第六章） |

---

## 六、课后习题与作业（CANNJudge 承载）

> 课后习题用于每讲结束后的即时巩固（在线题库 / 判题），小作业用于实践产出验收（判题型，易=跟练跑通 / 中=变体改造 / 难=独立实现）；结业大作业为课程最终产出。**当前各讲课后习题、小作业及结业判题作业的 CANNJudge 题目链接均缺失**，待平台上线后补充（各讲缺失状态见 6.1「CANNJudge 链接」列）。

### 6.1 每讲课后习题与小作业

> 「作业参考资源」列基于各讲内容标注具体参考来源，**并非全部位于同一仓库**：无前缀相对路径位于 CANN asc-devkit 仓 `examples/` 目录（<https://gitcode.com/cann/asc-devkit/tree/master/examples>，主要为算子开发类作业）；`learning-hub:` 前缀路径位于 cann-learning-hub 教程仓（<https://gitcode.com/cann/cann-learning-hub/>，主要为训推、工具链与教程类作业）；无参考资源的作业标 `—`。

| 讲次 | 课后习题（CANNJudge 在线题库） | 小作业（CANNJudge 判题） | 作业参考资源 | CANNJudge 链接 |
|------|------------------------------|------------------------|-------------|---------------|
| 第 1~4 讲 | —（行业导论） | —（以行业案例研讨为主，无判题作业） | — | — |
| 第 5 讲 | 昇腾生态与 CANN 分层架构概念题 | PyTorch NPU 入门实践练习 | learning-hub: `quick_start/cann_basics/` | 待补充 |
| 第 6 讲 | ATC 编译与离线推理流程概念题 | 待补充 | learning-hub: `tutorials/ge_development/03_graph_compilation` | 待补充 |
| 第 7 讲 | 预训练/SFT/RL 训练概念题 | 待补充 | learning-hub: `tutorials/sft_training_pipeline`、`tutorials/rl_training_pipeline` | 待补充 |
| 第 8 讲 | CANN 部署推理流程概念题 | 待补充 | learning-hub: `tutorials/llm_inference/qwen3_1.7B` | 待补充 |
| 第 9 讲 | 入图推理优化概念题 | 待补充 | learning-hub: `tutorials/llm_inference/qwen3_1.7B` | 待补充 |
| 第 10 讲 | PD 分离/KV 池化/自动融合概念题 | 待补充 | learning-hub: `tutorials/llm_inference/qwen3_1.7B` | 待补充 |
| 第 11 讲 | 异构计算与 Ascend C 编程模型概念题 | SIMD Hello World & Add 算子快速入门 | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 12 讲 | 调试工具链操作题（Profile/仿真） | Profile 使用方法、仿真性能统计方法、SIMD 算子最佳实践 | `01_simd_cpp_api/01_utilities/04_profiling`、`01_simd_cpp_api/01_utilities/08_simulator`、`01_simd_cpp_api/05_best_practices` | 待补充 |
| 第 13 讲 | 算子注册与入图流程题 | PyTorch 单算子调用实践、Ascend C 算子入图实践 | `01_simd_cpp_api/02_features/00_framework/00_pytorch`、`04_aclgraph`、`03_ge` | 待补充 |
| 选修 | — | 学有余力者可挑战进阶判题（题号待补充） | learning-hub: `tutorials/ascendc_operator_development_light/02_AscendC_basic`、`tutorials/ascendc_operator_development_light/03_simple_operator_practice` | 待补充 |

### 6.2 结业大作业（三项）

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **可选**（与作业 2 任选其一） | 昇腾 NPU 模型迁移与部署：在昇腾 NPU 上完成一个 AI 模型的部署与推理 | 工具链使用能力；端到端跑通 ＋ 性能报告 |
| **作业 2** | **可选**（与作业 1 任选其一） | 开发矢量算子 **AddRmsNorm** 或矩阵算子 **QuantMatmul**，支持 Qwen3 模型，功能跑通 | CANNJudge 判题通过（正确性） |
| **作业 3** | **开源贡献** | 向昇腾 CANN 社区提交 PR 并被合并 | PR 提交与合并记录 |

---

## 七、学习建议

### 7.1 学习节奏

- **第一部分（第 1~4 讲）**：以行业认知为主，建立「数据-模型-算力」协同思维与性能闭环意识，结合自身方向（推荐/检索/时序等）酝酿结业选题
- **第二部分（第 5~10 讲）**：以「会用会优化」为主，重点掌握大模型训推流程与优化手段（ATC 编译→训练→部署推理→入图/PD 分离/KV 池化/自动融合），按时完成 cann-learning-hub 练习
- **第三部分（第 11~13 讲）**：以「会写会调」为主，重点掌握 Ascend C 矢量算子开发闭环（开发→调试→调用→入图）；学有余力者加选修（Ascend C 950：编程模型/Reg 矢量/矩阵/融合与 SIMT）
- **结业大作业（第 13~16 周课后）**：建议结合具体业务场景（推荐/检索/时序等）选题，从作业 1/作业 2 中任选其一，将训推优化与算子开发融会贯通

### 7.2 成功要素

1. **跟练与判题同步**：教程练手（会不会）→ 判题独立实现（对不对、快不快），不要只跟练不判题
2. **算子从简单起步**：先从简单算子（Add/ReLU）开始，理解每一步原理，再逐步过渡到复杂算子（AddRmsNorm/QuantMatmul）
3. **选修按需选择**：选修（Ascend C 950 算子编程：编程模型/Reg 矢量/矩阵/融合与 SIMT）面向学有余力者，为衔接《AI 计算与神经网络计算架构实践》36 课时完整算子体系做准备
4. **结业选题结合场景**：从自身方向（推荐/检索/时序等行业场景）确定题目，并争取作业 3 开源贡献加分

### 7.3 后续学习路径

完成本课程后，可根据兴趣选择深入方向：

| 方向 | 推荐路径 | 目标 |
|------|---------|------|
| **完整算子体系** | 衔接《AI 计算与神经网络计算架构实践》第二部分（36 课时，A2/A3 与 Ascend 950 双路线） | 矢量/矩阵/融合全类型算子开发能力 |
| **算子极致性能** | A2/A3 极致性能三课（L4-20~22）→ 950 极致性能（L4-23~25） | 高性能算子开发 |
| **大模型系统优化** | 分布式训练、推理服务化、超节点架构 | 大模型系统优化能力 |
| **行业应用** | 参与昇腾社区行业 SIG（embodied AI、推荐系统等）实战贡献 | 行业落地与开源协作 |

---

## 八、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| CANN 社区主站 | <https://gitcode.com/cann> | CANN 开源社区，获取全部代码与文档 |
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现（作业参考资源所在仓） |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |

# 《启航营（两天）》课程大纲

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 启航营（两天）：昇腾 AI 与 CANN 开发速成 |
| **课程类型** | 短期集训营（企业培训 / 高校夏令营 / 竞赛备赛） |
| **学时安排** | 2 天 × 上午/下午两段 ≈ 16 小时（理论 7h + 实践 7h，每讲理论与实践 1:1 配对；另含结业作业 2~3h） |
| **授课对象** | 零基础新入行者、竞赛新手、跨领域转型开发者 |
| **先修要求** | Python/C/C++ 基础；了解Makefile/GIT操作等 |
| **配套讲义** | [two_days_course/](./two_days_course/)（7 讲 PPT） |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测） |
| **结业标准** | 独立完成矢量算子 `add_rms_norm` 开发（Qwen3-1.7B 模型场景）并通过正确性判题 |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 了解昇腾 AI 产业生态与 CANN 软件栈分层架构；② 掌握大模型部署推理与推理优化基本方法；③ 理解 Ascend C 矢量算子开发流程与 PyTorch 接入方法；④ 了解 CANNBot 智能开发模式 |
| **能力目标** | ① 能在 NPU 上完成大模型部署与推理；② 能独立开发矢量算子 `add_rms_norm` 并通过判题；③ 能将算子接入 PyTorch 框架（进阶）；④ 能通过持续优化提升推理模型性能（进阶） |
| **素养目标** | 建立异构计算思维与「硬件-软件协同」意识，具备独立查阅 CANN 官方文档与社区资源的能力 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **说清楚**：昇腾 AI 生态与 CANN 软件栈的分层关系，NPU 与 GPU 在架构上的核心差异; Ascend C 基本编程模型；
- **跑起来**：在 NPU 环境中完成大模型推理部署，并使用基础优化手段提升推理性能
- **写得出**：使用 Ascend C 独立开发一个矢量算子，完成从编码、编译到调试的全流程
- **接得进**：将自定义算子接入 PyTorch 框架，实现端到端模型推理（进阶）

### 2.3 课程内容规划

| 模块 | 讲次 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| 生态与架构认知 | 第 1 讲 | 昇腾生态全景；CANN 分层架构；NPU vs GPU 架构差异 | CANN 全栈认知 + NPU 初体验 |
| 大模型部署与推理 | 第 2~3 讲 | CANN 推理工具链与部署流程；推理优化（量化 / 算子融合 / 图模式 / KV Cache / Profiling 定位） | Qwen3 基线推理 + 优化实践 |
| Ascend C 算子开发 | 第 4~5 讲 | 矢量算子开发全流程（支撑推理模型优化）；算子接入 PyTorch；CANNBot 智能开发 | 矢量算子独立实现 + 框架接入 |
| 结业冲刺 | 第 6 讲 | `add_rms_norm` 开发、集成与展示 | 结业算子功能跑通（Qwen3-1.7B 场景） |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 编程语言 | Python / C / C++ 基础语法（变量、循环、函数、指针与内存概念） |
| 系统操作 | Linux 命令行基本操作（目录 / 文件 / 权限 / 进程） |
| 工程工具 | Makefile 基本规则；GIT clone / commit / push 基本操作 |

### 3.2 领域前置（建议完成）

- 建议课前预习 [cann-learning-hub 快速入门](../../quick_start) 的 `cann_basics` 章节（AI 基础概念、NPU 硬件架构、CANN 软件栈）；
- 有 PyTorch 模型训练 / 推理使用经验更佳（第 2~3 讲大模型部署推理、第 5 讲算子接入 PyTorch 会直接涉及）。

### 3.3 环境前置

- **在线环境（推荐）**：CANNLab 云端开发环境（cann_9.0.0 py3.11-A3-arm），浏览器访问即可；
- **本地环境（可选）**：已部署 CANN 9.0.0+ 的昇腾开发环境（Atlas A3 训练/推理系列）；大模型推理实践需较大显存，建议 A3 及以上环境。

---

## 四、教学安排与理论讲义（PPT）（2 天，按上午/下午排课）

| 时段 | 讲次 | 主题 | 内容要点 | 教材PPT |
|------|------|------|---------|------|
| **D1 上午** | 第 1 讲 | 昇腾 AI 产业生态与 CANN 架构基础 | 昇腾生态全景；CANN 分层架构；NPU/GPU 差异 | 1h + 1h, [PPT](../00_atomic_courses/02_framework/01_overview/artificial_intelligence_basics_light.pptx) |
| **D1 上午** | 第 2 讲 | 基于 CANN 如何部署和推理大模型 | CANN 推理工具链；ATC 模型转换；基线推理流程 | 1h + 1h, [PPT](./two_days_course/02_llm_deployment_and_inference_with_cann_1h.pptx) |
| **D1 下午** | 第 3 讲 | 基于 CANN 的大模型推理优化与最佳实践 | 量化 / 算子融合 / 图模式 / KV Cache；Profiling 瓶颈定位 | 2h + 2h, [PPT](./two_days_course/03_llm_training_and_inference_optimization_overview_with_cann_2h.pptx) |
| **D2 上午** | 第 4 讲 | 如何开发 Ascend C 矢量算子（支撑推理模型优化） | Ascend C 编程模型；矢量算子开发全流程；推理算子场景 | 2h + 2h, [PPT](../00_atomic_courses/04_ops_programming/01_ascendc/a2a3_ascend_c_simd_vector_operator_development.pptx) |
| **D2 下午** | 第 5 讲 | Ascend C 算子如何接入 PyTorch ＋ 基于 CANNBot 的 Ascend C 算子开发介绍 | 算子注册与单算子调用；CANNBot 辅助开发 | 1h + 1h, [PPT1](./two_days_course/05_a2a3_ascend_c_operator_pytorch_single_operator_call_0.5h.pptx)、[PPT2](./two_days_course/05_cannbot_highlights_open_source_community_edition_0.5h.pptx) |
| **D2 下午** | 第 6 讲 | 结业作业辅导与展示 | 结业要求解读；作业冲刺与成果展示 | 2~3h（结业作业说明参考 [PPT](./two_days_course/06_a2a3_two_day_bootcamp_final_project.pptx)） |

> **关键节点说明：**
> - 每讲理论课后立即配套等时长实践（1:1 配对，实践内容见第五章）；第 5 讲由两个 0.5h 主题（PyTorch 接入 + CANNBot）组成，实践各半；
> - 第 4 讲算子开发实践与第 2~3 讲推理实践形成呼应（算子支撑推理模型优化）；
> - 第 6 讲结业作业 2~3h，可借助 CANNBot 辅助开发与调试，未完成部分可课后补交。

---

## 五、每节课的实践内容（CANN-Learning-Hub 承载）

> 每讲实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，与理论课 1:1 配对（各 1~2h）。

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 1 讲 | NPU 环境初体验 | 跑通 PyTorch NPU HelloWorld + MNIST 单 epoch 训练 | `quick_start/cann_basics/` |
| 第 2 讲 | 大模型基线推理 | Qwen3-1.7B 部署与基线推理跑通 | `tutorials/llm_inference/qwen3_1.7B` |
| 第 3 讲 | 推理优化实践 | Profiling 采集 + 量化 / 融合 / 图模式等优化手段验证 | `tutorials/llm_inference/qwen3_1.7B` |
| 第 4 讲 | 矢量算子开发 | Add 算子跟练 → 变体改造 → 独立实现（支撑推理模型优化场景） | `tutorials/ascendc_operator_development_light/01_basic_overview`、`tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 第 5 讲 | 算子接入与智能开发 | 自定义算子 PyTorch 注册与调用；CANNBot 生成算子并对比 | `tutorials/ascendc_operator_development_light/02_AscendC_basic`、`tutorials/CANNBot` |
| 第 6 讲 | 结业作业冲刺 | `add_rms_norm` 开发、集成与展示（判题由 CANNJudge 承载，见第六章） | — |

> **实践目录说明：** 「实践路径」列为 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 教程仓（本仓库）`quick_start/`、`tutorials/` 下的实际教程目录，进入对应目录即可跟练 Notebook。

---

## 六、课后习题与作业（CANNJudge 承载）

> 课后习题用于每讲结束后的即时巩固（在线题库 / 判题），小作业用于实践产出验收（判题型，易=跟练跑通 / 中=变体改造 / 难=独立实现）；结业大作业为课程最终产出。**当前各讲课后习题、小作业及结业判题作业的 CANNJudge 题目链接均缺失**，待平台上线后补充（各讲缺失状态见 6.1「CANNJudge 链接」列）。

### 6.1 每讲课后习题与小作业

> 「作业参考资源」列基于各讲内容标注具体参考来源，**并非全部位于同一仓库**：无前缀相对路径位于 CANN asc-devkit 仓 `examples/` 目录（<https://gitcode.com/cann/asc-devkit/tree/master/examples>，主要为算子开发类作业）；`learning-hub:` 前缀路径位于 cann-learning-hub 教程仓（<https://gitcode.com/cann/cann-learning-hub/>，主要为训推、工具链与教程类作业）；无参考资源的作业标 `—`。

| 讲次 | 课后习题（CANNJudge 在线题库） | 小作业（CANNJudge 判题） | 作业参考资源 | CANNJudge 链接 |
|------|------------------------------|------------------------|-------------|---------------|
| 第 1 讲 | Pytorch NPU 入门练习 | torch.matmul 调用实验 | torch API链接 | 待补充CANN Judge链接 |
| 第 2 讲 | CANN 部署推理流程概念题 | - | - | 待补充 |
| 第 3 讲 | 推理优化手段配对题（场景 → 优化手段） | - | - | 待补充 |
| 第 4 讲 | Ascend C 矢量算子 | SIMD Hello World和Add算子快速入门、Softmax进阶(可选) |  `01_simd_cpp_api/00_introduction` | 待补充CANN Judge链接 |
| 第 5 讲 | Ascend C单算子接入Pytorch| Add、Softmax单算子接入Pytorch| `01_simd_cpp_api/02_features/00_framework/00_pytorch`| 待补充 |
| 第 6 讲 | —（结业冲刺） | —（进入结业大作业） | — | — |

### 6.2 结业大作业（四档）

> **`add_rms_norm` 结业作业由两部分构成**：**第一部分 · 功能跑通**（作业 2，CANNJudge 判题通过）→ **第二部分 · 持续优化性能**（作业 4，性能提升幅度 + 优化路径说明）；作业 3 PyTorch 集成为独立可选作业维度。

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发矢量算子 **`Sigmoid`** | CANNJudge 判题通过 |
| **作业 2** | **可选** | 开发矢量算子 **`add_rms_norm`**（仅需支持 Qwen3-1.7B 模型）—— 第一部分：功能跑通 | CANNJudge 判题通过 |
| **作业 3** | 可选 | 将 `add_rms_norm` 算子**接入 PyTorch Qwen3-1.7B 模型** | 单算子调用正确性 + 端到端模型验证 |
| **作业 4** | 可选 | **持续优化** `add_rms_norm` 算子，优化推理模型性能 —— 第二部分：持续优化性能 | 性能提升幅度 + 优化路径说明（访存/融合等） |

---

## 七、学习建议

- **D1「听懂 + 跟练」**：重点是理解概念、跑通示例，不追求独立实现；上午建立昇腾/CANN 整体认知并跑通大模型推理，下午深入推理优化与 A/B 实验。
- **D2「动手 + 独立」**：重点是独立开发算子、完成结业作业；上午集中攻克 Ascend C 算子开发并通过 Add 判题，下午完成结业算子与集成优化，展示成果。
- **成功要素**：动手优先（每讲实践必须亲手完成）；善用工具（CANNJudge 即时反馈 + CANNBot 辅助调试）；数据驱动（性能优化基于测量数据，记录每次改动收益）；同伴互助（邻座讨论 + 成果展示互评）；文档查阅（养成查阅官方文档习惯）。
- **学有余力**完成可选作业可作为两周营先修凭证，后续可深入算子开发（矩阵/融合/极致性能）、框架集成（TorchAir/AclGraph）、推理优化（KV Cache/量化/服务部署）或系统架构（NPU 架构/超节点互联）方向。

---

## 八、进一步学习参考
| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标 / 参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（本课程实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |

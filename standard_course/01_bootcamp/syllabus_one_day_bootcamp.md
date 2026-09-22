# 《启航营（一天）》课程大纲

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 启航营（一天）：昇腾 AI 与 CANN 算子开发初体验 |
| **课程类型** | 短期体验集训营（企业活动 / 高校宣讲 / 社区体验日 / 竞赛热身） |
| **学时安排** | 1 天 × 上午/下午两段 ≈ 6.5 小时（理论 2.5h + 实践 1h + 结业作业 2~3h） |
| **授课对象** | 零基础新入行者、竞赛新手、跨领域转型开发者 |
| **先修要求** | Python/C/C++ 基础；了解Makefile/GIT操作等 |
| **配套讲义** | 与两天营共用 [two_days_course/](./two_days_course/)（3 讲 PPT） |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测） |
| **结业标准** | 独立完成矢量算子 `add_rms_norm` 开发并功能跑通 |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 了解昇腾 AI 产业生态与 CANN 软件栈分层架构；② 理解 Ascend C 矢量算子开发流程与基本编程模型；③ 了解 CANNBot 智能开发模式 |
| **能力目标** | ① 能在 NPU 环境中跑通矢量算子示例；② 能独立开发矢量算子 `add_rms_norm` 并功能跑通；③ 能通过持续优化提升算子性能（进阶）；④ 能借助 CANNBot 辅助算子开发与问题排查 |
| **素养目标** | 建立异构计算思维与「硬件-软件协同」意识，具备独立查阅 CANN 官方文档与社区资源的能力 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **说清楚**：昇腾 AI 生态与 CANN 软件栈的分层关系，Ascend C 矢量算子开发的基本流程；
- **跑起来**：在 NPU 环境中搭建开发环境，跑通矢量算子示例；
- **写得出**：使用 Ascend C 独立开发一个矢量算子，完成功能跑通；
- **调得优**：借助 CANNJudge 性能反馈与 CANNBot 辅助，持续优化算子性能（进阶）。

### 2.3 课程内容规划

| 模块 | 讲次 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| 生态与架构认知 | 第 1 讲 | 昇腾 AI 产业生态全景；CANN 软件栈分层架构；Atlas 硬件产品线；开发者资源导航 | 建立 CANN 全栈认知 |
| Ascend C 矢量算子开发 | 第 2 讲 | 异构编程模型；核函数 / Tiling / 流水；矢量算子开发流程 | Add 算子跟练 → 独立实现 |
| 智能开发范式 | 第 3 讲 | CANNBot 功能与用法；Ascend C 算子自动生成 | 借助 CANNBot 完成一次算子生成 |
| 结业冲刺 | 第 4 讲 | `add_rms_norm` 需求解读；功能实现与验证 | 结业算子功能跑通 |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 编程语言 | Python / C / C++ 基础语法（变量、循环、函数、指针与内存概念） |
| 系统操作 | Linux 命令行基本操作（目录 / 文件 / 权限 / 进程） |
| 工程工具 | Makefile 基本规则；GIT clone / commit / push 基本操作 |

### 3.2 领域前置（无硬性要求，零基础可参加）

- 无 AI / 异构计算背景要求；课程从生态与架构认知起步，逐步过渡到算子开发；
- 建议课前预习 [cann-learning-hub 快速入门](../../quick_start) 的 `cann_basics` 章节（AI 基础概念、NPU 硬件架构、CANN 软件栈），可显著提升跟课效率。

### 3.3 环境前置

- **在线环境（推荐）**：CANNLab 云端开发环境，浏览器访问即可，无需本地 NPU；
- **本地环境（可选）**：已部署 CANN 8.5.0+ 的昇腾开发环境（Atlas A2/A3 系列）。

---

## 四、教学安排与理论讲义（PPT）（1 天，按上午/下午排课）

| 时段 | 讲次 | 主题 | 内容要点 | 教材PPT |
|------|------|------|---------|------|
| **D1 上午** | 第 1 讲 | 昇腾 AI 产业生态与 CANN 架构基础 | 昇腾生态全景；CANN 分层架构；Atlas 产品线；资源导航 | 1h, [PPT](./two_days_course/01_artificial_intelligence_basics_light.pptx) |
| **D1 上午** | 第 2 讲 | 如何开发 Ascend C 矢量算子 | 异构计算模型；核函数 / Tiling / 流水；开发全流程；上机跟练（Add 算子跟练 → 变体改造 → 过渡独立实现） | 1h 理论 + 1h 实践, [PPT](./two_days_course/04_a2a3_ascend_c_simd_vector_operator_development_2h.pptx)（2h 版讲义，一天营节选核心内容；实践内容见第五章） |
| **D1 下午** | 第 3 讲 | 基于 CANNBot 的 Ascend C 算子开发介绍 | CANNBot 简介；算子自动生成演示；辅助调试 | 0.5h, [PPT](./two_days_course/05_cannbot_highlights_open_source_community_edition_0.5h.pptx) |
| **D1 下午** | 第 4 讲 | 结业作业冲刺与展示 | `add_rms_norm` 需求解读；功能实现与验证 | 2~3h（结业作业说明参考 [PPT](./two_days_course/06_a2a3_two_day_bootcamp_final_project.pptx)） |

> **关键节点说明：**
> - 第 1 讲上午 1h 纯理论；第 2 讲上午 1h 理论 ＋ 1h 实践（跟练 Add → 过渡到独立实现）；
> - 第 3 讲 0.5h 介绍 CANNBot 后，结业作业可借助 CANNBot 辅助开发与调试；
> - 第 4 讲结业作业冲刺 2~3h，未完成部分可课后补交。

---

## 五、每节课的实践内容（CANN-Learning-Hub 承载）

> 每讲实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，与理论课配对（一天营重点在第 2 讲的 1h 上机实践）。

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 1 讲 | NPU 环境初体验 | 跑通 PyTorch NPU HelloWorld，验证环境与算力 | `quick_start/cann_basics/` |
| 第 2 讲 | Ascend C 矢量算子上机 | Add 算子跟练 → 变体改造 → 过渡独立实现（1h） | `tutorials/ascendc_operator_development_light/01_basic_overview`、`tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 第 3 讲 | CANNBot 算子生成体验 | 用 CANNBot 生成一个矢量算子，与手写版本对比 | `tutorials/CANNBot` |
| 第 4 讲 | 结业作业冲刺 | `add_rms_norm` 功能实现与验证（判题由 CANNJudge 承载，见第六章） | — |

> **实践目录说明：** 「实践路径」列为 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 教程仓（本仓库）`quick_start/`、`tutorials/` 下的实际教程目录，进入对应目录即可跟练 Notebook。

---

## 六、课后习题与作业（CANNJudge 承载）

> 课后习题用于每讲结束后的即时巩固（在线题库 / 判题），小作业用于实践产出验收（判题型，易=跟练跑通 / 中=变体改造 / 难=独立实现）；结业大作业为课程最终产出。**当前各讲课后习题、小作业及结业判题作业的 CANNJudge 题目链接均缺失**，待平台上线后补充（各讲缺失状态见 6.1「CANNJudge 链接」列）。

### 6.1 每讲课后习题与小作业

> 「作业参考资源」列基于各讲内容标注具体参考来源，**并非全部位于同一仓库**：无前缀相对路径位于 CANN asc-devkit 仓 `examples/` 目录（<https://gitcode.com/cann/asc-devkit/tree/master/examples>，主要为算子开发类作业）；`learning-hub:` 前缀路径位于 cann-learning-hub 教程仓（<https://gitcode.com/cann/cann-learning-hub/>，主要为训推、工具链与教程类作业）；无参考资源的作业标 `—`。

| 讲次 | 课后习题（CANNJudge 在线题库） | 小作业（CANNJudge 判题） | 作业参考资源 | CANNJudge 链接 |
|------|------------------------------|------------------------|-------------|---------------|
| 第 1 讲 | Pytorch NPU 入门练习 | torch.matmul 调用实验 | torch API链接 | 待补充CANN Judge链接 |
| 第 2 讲 | Ascend C 矢量算子 | SIMD Hello World 和 Add 算子快速入门 |  `01_simd_cpp_api/00_introduction` | 待补充CANN Judge链接 |
| 第 3 讲 | CANNBot 功能与使用概念题 | 用 CANNBot 生成一个矢量算子并与手写版本对比（提交对比记录） | learning-hub: `tutorials/CANNBot` | 待补充 |

### 6.2 结业大作业（三档）

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发矢量算子 **`sigmoid`** ，支持 Qwen3 大模型，功能跑通 | CANNJudge 判题通过（正确性门禁：多形状/多类型/精度达标） |
| **作业 2** | 可选 | **持续优化** `add_rms_norm` 算子性能 | 性能提升幅度 + 优化路径说明（VF粒度/双发射/访存/融合等） |
| **作业 3** | 可选 | **持续优化** `add_rms_norm` 算子性能 | 性能提升幅度 + 优化路径说明（VF粒度/双发射/访存/融合等） |

---

## 七、学习建议

- **上午「听懂 + 跟练」**：重点是理解概念、跑通示例，不追求独立实现；先建立昇腾/CANN 整体认知，再掌握 Ascend C 矢量算子开发流程。
- **下午「独立 + 冲刺」**：重点是独立完成结业算子开发；矢量算子实践从跟练过渡到独立实现，随后冲刺结业作业 `add_rms_norm` 功能跑通。
- **成功要素**：动手优先（每讲实践必须亲手完成）；善用工具（CANNJudge 即时反馈 + CANNBot 辅助开发与调试）；文档查阅（养成查阅官方文档习惯）；节奏紧凑（一天营时间有限，未完成的结业部分课后补交）。
- **学有余力**完成可选作业可作为两天营/两周营先修凭证，后续可深入算子开发（矩阵/融合/极致性能）、框架集成（PyTorch/TorchAir/AclGraph）或推理优化（KV Cache/量化/服务部署）方向。

---

## 八、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标 / 参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（本课程实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |

# 《启航营（两天）》课程大纲

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 启航营（两天）：昇腾 AI 与 CANN 开发速成 |
| **课程类型** | 短期集训营（企业培训 / 高校夏令营 / 竞赛备赛） |
| **学时安排** | 2 天 × 上午/下午两段 ≈ 14 小时（理论 7h + 实践 7h，每讲理论与实践 1:1 配对） |
| **授课对象** | 零基础新入行者、竞赛新手、跨领域转型开发者 |
| **先修要求** | Python/C/C++ 基础；了解Makefile/GIT操作等 |
| **配套讲义** | [two_days_course/](./two_days_course/)（7 讲 PPT） |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测） |
| **结业标准** | 独立完成矢量算子 `add_rms_norm` 开发并通过正确性判题 |

---

## 二、课程目标（OBE 成果导向）

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

---

## 三、教学安排与实践（2 天，按上午/下午排课）

| 时段 | 讲次 | 主题 | 教材PPT | 实践(cann-learning-hub) | 实践（cann-judge) |
|------|------|------|------|------|------------|
| **D1 上午** | 第 1 讲 | 昇腾 AI 产业生态与 CANN 架构基础 |1h, [PPT](./two_days_course/01_artificial_intelligence_basics_light.pptx) | [人工智能基础](../../quick_start/cann_basics)  | — |
| **D1 上午** | 第 2 讲 | 基于 CANN 部署和推理大模型 | 1h, [PPT](./two_days_course/02_llm_deployment_and_inference_with_cann_1h.pptx)  | [Qwen3-1.7B 推理优化实践](../../tutorials/llm_inference/qwen3_1.7B) | — |
| **D1 下午** | 第 3 讲 | 大模型推理优化与最佳实践 | 2h, [PPT](./two_days_course/03_llm_training_and_inference_optimization_overview_with_cann_2h.pptx) | [Qwen3-1.7B 推理优化实践](../../tutorials/llm_inference/qwen3_1.7B) | — |
| **D2 上午** | 第 4 讲 | Ascend C 矢量算子开发 | 2h, [PPT](./two_days_course/04_a2a3_ascend_c_simd_vector_operator_development_2h.pptx) |  [算子开发系列](../../tutorials/ascendc_operator_development_light) | [CANNJudge 刷题](https://cannjudge.cn) |
| **D2 下午** | 第 5 讲 | 算子接入 PyTorch ＋ CANNBot 智能开发 | 1h, [PPT1](./two_days_course/05_a2a3_ascend_c_operator_pytorch_single_operator_call_0.5h.pptx)、[PPT2](./two_days_course/05_cannbot_highlights_open_source_community_edition_0.5h.pptx) |[PyTorch框架下Kernel直调](../../tutorials/ascendc_operator_development_light/02_AscendC_basic)| |
| **D2 下午** | 第 6 讲 | 结业作业辅导与展示 | — | — |[AddRmsNorm算子开发](https://cannjudge.cn/public/vector/addrmsnorm) |

---

## 四、结业作业（三档）

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发矢量算子 **`add_rms_norm`**（x + bias → RMSNorm → 输出） | CANNJudge 判题通过（正确性门禁：多形状/多类型/精度达标/性能不低于参考 80%） |
| **作业 2** | 可选 | 将 `add_rms_norm` 算子**接入 PyTorch 框架** | 单算子调用正确性 + 端到端模型验证 |
| **作业 3** | 可选 | **持续优化** `add_rms_norm`，优化推理模型性能 | 性能提升幅度 + 优化路径说明（VF粒度/双发射/访存/融合等） |

---

## 五、考核方式

> 以下为参考框架，具体考核方式由任课老师根据实际情况确定。

| 考核项 | 占比 | 说明 |
|--------|------|------|
| **课堂实践** | 30% | 第 1~5 讲上机实践完成度 |
| **结业作业 1（必选）** | 50% | CANNJudge 判题通过即得分；性能超出基线可额外加分 |
| **结业作业 2（可选）** | +10% | PyTorch 集成正确性 + 端到端验证 |
| **结业作业 3（可选）** | +10% | 性能提升幅度 + 优化路径说明质量 |
| **合计** | 100% + 20% 加分 | 可选作业加分上限 20% |

**评分等级：** 优秀（90~100+，必选通过+至少一项可选高质量）/ 良好（75~89）/ 合格（60~74，必选通过）/ 不合格（<60）

---

## 六、学习建议

- **D1「听懂 + 跟练」**：重点是理解概念、跑通示例，不追求独立实现；上午建立昇腾/CANN 整体认知并跑通大模型推理，下午深入推理优化与 A/B 实验。
- **D2「动手 + 独立」**：重点是独立开发算子、完成结业作业；上午集中攻克 Ascend C 算子开发并通过 Add 判题，下午完成结业算子与集成优化，展示成果。
- **成功要素**：动手优先（每讲实践必须亲手完成）；善用工具（CANNJudge 即时反馈 + CANNBot 辅助调试）；数据驱动（性能优化基于测量数据，记录每次改动收益）；同伴互助（邻座讨论 + 成果展示互评）；文档查阅（养成查阅官方文档习惯）。
- **学有余力**完成可选作业可作为两周营先修凭证，后续可深入算子开发（矩阵/融合/极致性能）、框架集成（TorchAir/AclGraph）、推理优化（KV Cache/量化/服务部署）或系统架构（NPU 架构/超节点互联）方向。

---

## 七、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标 / 参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（本课程实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |
| 昇腾社区 | <https://www.hiascend.com/> | 官方文档、论坛、课程、活动 |
| CANN 开发者论坛 | <https://bbs.huaweicloud.com/forum/forum-1109-1.html> | 问题求助、经验分享、技术交流 |

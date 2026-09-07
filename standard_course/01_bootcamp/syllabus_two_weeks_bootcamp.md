# 《启航营（两周）》课程大纲

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 启航营（两周）：CANN 算子开发系统入门 |
| **课程类型** | 短期强化集训营（高校夏令营 / 企业内训 / 竞赛集训） |
| **学时安排** | **10 个工作日 ≈ 60 小时**（理论 20h + 实践 40h；D3–D8 上午理论 2h + 下午实践 2~4h） |
| **授课对象** | 零基础新入行者、竞赛新手、跨领域转型开发者 |
| **先修要求** | Python/C/C++ 基础；了解Makefile/GIT操作等 |
| **配套讲义** | [two_weeks_course/](./two_weeks_course/)（13 讲 PPT） |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测） |
| **结业标准** | 独立完成 `add_rms_norm` + `quant_matmul` 双算子开发，通过判题并接入 PyTorch 端到端验证 |

---

## 二、课程目标（OBE 成果导向）

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握 CANN 软件栈与昇腾硬件架构；② 掌握大模型训练 / 推理 / 优化基本方法；③ 系统掌握 Ascend C SIMD 编程模型（**Memory 矢量 C API / 基础 API 双路线 + 矩阵算子**）与调试调优方法；④ 掌握算子接入 PyTorch 全流程（单算子调用 + 入图集成）；⑤ 了解 CANNBot 智能开发模式 |
| **能力目标** | ① 独立完成矢量 / 矩阵算子的开发、调试与调优（双 API 栈）；② 算子接入 PyTorch 并完成端到端验证；③ 通过 CANNJudge 矢量 + 矩阵双判题；④ 完成 **`add_rms_norm` + `quant_matmul` 双算子**结业项目 |
| **素养目标** | 形成「架构认知 → 编程实现 → 调试调优 → 框架集成」一体化工程思维，具备独立查阅文档与解决算子开发问题的能力 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **讲得透**：昇腾 NPU 架构（三核异构 / 存储层次 / 数据搬运）与 CANN 软件栈的分层关系
- **写得出**：使用 Ascend C C API 与基础 API 双路线独立开发矢量算子与矩阵算子
- **调得动**：使用 profiling 工具定位性能瓶颈，通过 VF 粒度 / 双发射 / 访存优化等手段提升算子性能
- **接得进**：将自定义算子接入 PyTorch 框架，完成单算子调用与入图端到端验证
- **用得好**：借助 CANNBot 等 AI 辅助工具加速算子开发与问题排查

---

## 三、教学安排与实践（10 天，上午理论 + 下午实践）

| 时段 | 讲次 | 主题 | 教材PPT | 实践(cann-learning-hub) | 实践（cann-judge) |
|------|------|------|------|------|------------|
| **D1 上午** | 第 1 讲 | 昇腾 AI 产业生态与 CANN 架构基础 | 2h, [PPT](./two_weeks_course/1_artificial_intelligence_basics.pptx) | 待补充链接 | 待补充链接 |
| **D1 下午** | 第 2 讲 | 基于 CANN 训练大模型 | 2h, [PPT](./two_weeks_course/02_03_llm_training_with_cann_2h.pptx) | 待补充链接 | 待补充链接 |
| **D2 上午** | 第 3 讲 | 基于 CANN 部署和推理大模型 | 2h, [PPT](./two_weeks_course/04_llm_deployment_and_inference_with_cann_2h.pptx) | 待补充链接 | 待补充链接 |
| **D2 下午** | 第 4 讲 | 大模型推理优化与最佳实践 | 2h, [PPT](./two_weeks_course/05_llm_training_and_inference_optimization_overview_with_cann_2h.pptx) | 待补充链接 | 待补充链接 |
| **D3 上午** | 第 5 讲 | 异构计算与 Ascend C 算子编程导论 | 2h, [PPT](./two_weeks_course/06_a2a3_heterogeneous_computing_and_ascend_c_operator_programming_introduction.pptx) | 待补充链接 | 待补充链接 |
| **D3 下午** | 第 6 讲 | SIMD 编程模型介绍 | 2h, [PPT](./two_weeks_course/07_a2a3_ascend_c_simd_programming_model.pptx) | 待补充链接 | 待补充链接 |
| **D4 上午** | 第 7 讲 | Memory 矢量算子编程实践（**基于 C API**） | 2h, [PPT](./two_weeks_course/08_a2a3_ascend_c_simd_memory_vector_operator_programming.pptx) | 待补充链接 | 待补充链接 |
| **D4 下午** | — | **纯实践日：C API 矢量算子** | — | 待补充链接 | 待补充链接 |
| **D5 上午** | 第 9 讲 | Memory 矢量算子编程实践（**基于基础 API**） | 2h, [PPT](./two_weeks_course/08_a2a3_ascend_c_simd_memory_vector_operator_programming.pptx) | 待补充链接 | 待补充链接 |
| **D5 下午** | — | **纯实践日：双 API 对照** | — | 待补充链接 | 待补充链接 |
| **D6 上午** | 第 11 讲 | 矩阵算子编程实践（基于基础 API） | 2h, [PPT](./two_weeks_course/10_a2a3_ascend_c_simd_matrix_operator_programming.pptx) | 待补充链接 | 待补充链接 |
| **D6 下午** | — | **纯实践日：矩阵算子** | — | 待补充链接 | 待补充链接 |
| **D7 上午** | 第 13 讲 | 调试调优与最佳实践 | 2h, [PPT](./two_weeks_course/12_a2a3_ascend_c_operator_debugging_tuning_and_best_practices.pptx) | 待补充链接 | 待补充链接 |
| **D7 下午** | — | **纯实践日：性能优化** | — | 待补充链接 | 待补充链接 |
| **D8 上午** | 第 15/16 讲 | 算子接入 PyTorch（单算子调用 + 入图验证） | 1h + 1h, [PPT1](./two_weeks_course/14_a2a3_ascend_c_operator_pytorch_single_operator_call.pptx)、[PPT2](./two_weeks_course/15_a2a3_ascend_c_operator_graph_integration_pytorch_aclgraph_ge.pptx) | 待补充链接 | 待补充链接 |
| **D8 下午** | — | 实践：PyTorch 集成 | — | 待补充链接 | 待补充链接 |
| **D9 上午** | 第 17 讲 | 基于 CANNBot 的 Ascend C 算子开发介绍 | 1h, [PPT](./two_weeks_course/16_cannbot_highlights_open_source_community_edition_0.5h.pptx) | 待补充链接 | 待补充链接 |
| **D9 下午–D10** | 第 18 讲 | 结业项目 + 答辩 | —, [PPT](./two_weeks_course/17_a2a3_two_week_bootcamp_final_project.pptx) | 待补充链接 | 待补充链接 |

> **关键节点说明：**
> - **D4 / D6 / D7 下午为 4h 纯实践日**（无新理论，按跟练 → 独立实现 → 判题推进）；
> - **D5 为 C API vs 基础 API 同题双实现对照日**，需提交双实现对照报告（工时 / 代码量 / 性能）；
> - **D9–D10 为结业项目冲刺期**，完成双算子 + 框架集成 + 可选优化，最终答辩展示。

---

## 四、结业作业（三档）

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发矢量算子 **`add_rms_norm`** ＋ 矩阵算子 **`quant_matmul`** | CANNJudge 双算子判题通过（正确性 + 性能不低于参考 80%） |
| **作业 2** | **必选** | 双算子**接入 PyTorch 模型** | 单算子调用正确性 + 入图端到端验证 |
| **作业 3** | 可选 | **持续优化**双算子，提升推理性能 | 性能提升幅度 + 优化路径剖析证据（profiling 数据 + 改动说明） |

> **与两天营衔接：** 同源递进——两天营 = 矢量单算子入门；两周营 = 双算子（矢量 + 矩阵）+ 框架集成 + 性能优化，是两天营的系统性延伸。

---

## 五、考核方式

> 以下为参考框架，具体考核方式由任课老师根据实际情况确定。

| 考核项 | 占比 | 说明 |
|--------|------|------|
| **每日实践** | 30% | D1–D8 每日实践完成度；含 D5 双 API 对照报告、D7 优化数据 |
| **CANNJudge 判题** | 30% | 矢量算子判题（40%）+ 矩阵算子判题（60%） |
| **结业项目** | 40% | 必选双算子 + PyTorch 集成（基础分）；可选性能优化（加分） |
| **合计** | 100% | — |

**评分等级：** 优秀（90~100+，必选全通过 + 可选优化有显著收益）/ 良好（75~89）/ 合格（60~74，必选全通过）/ 不合格（<60）

---

## 六、学习建议

### 6.1 两周学习节奏

- **D1–D2「认知 + 体验」**：建立昇腾/CANN 整体认知，跑通大模型训练与推理，理解 AI 计算的上层视角
- **D3–D7「核心 + 攻坚」**：系统学习 Ascend C 算子开发，从导论 → SIMD 编程模型 → 矢量算子（双 API）→ 矩阵算子 → 调试调优，逐步深入；D4/D5/D6/D7 下午纯实践是能力提升的关键期
- **D8–D10「集成 + 产出」**：完成算子 PyTorch 集成，借助 CANNBot 加速开发，最终完成双算子结业项目并答辩

### 6.2 成功要素

1. **纯实践日是分水岭**：D4/D5/D6/D7 下午的 4h 纯实践是从「听懂」到「会做」的关键，务必独立完成，不要只跟练
2. **双 API 对照日要深度思考**：D5 的 C API vs 基础 API 对照不仅是写两份代码，更要理解两种抽象层次的权衡（控制力 vs 开发效率），为后续选择合适的开发方式建立直觉
3. **性能优化要有数据支撑**：D7 的优化实践和结业可选作业都要基于 profiling 数据，记录每次改动的性能变化，形成「测量 → 分析 → 优化 → 验证」的闭环
4. **善用 CANNBot 但不依赖**：D9 引入 CANNBot 后，可用于代码生成、错误分析、优化建议，但所有 AI 生成代码必须人工验证，尤其是正确性与性能
5. **同伴互助与代码评审**：两周营时间较长，建议结对学习，互相 code review，从同伴的代码中学习不同的实现思路

### 6.3 后续学习路径

完成两周启航营后，可根据兴趣选择深入方向：

| 方向 | 推荐路径 | 目标 |
|------|---------|------|
| **算子极致性能** | Ascend C 高级原子课程（矢量极致性能 / 矩阵极致性能 / 融合算子极致性能）→ SIMD&SIMT 混合编程 | 独立开发达到理论峰值 90%+ 的高性能算子 |
| **算子工程化** | Aclnn 算子工程化开发 → 算子入图（PyTorch/AclGraph/GE）→ 通信算子自定义开发 | 具备算子库级别的工程化开发能力 |
| **框架深度开发** | TorchNPU 深度实践 → TorchAir / AclGraph → 分布式训练框架 → 图融合 PASS 开发 | 深度参与 AI 框架层开发与优化 |
| **大模型系统优化** | 大模型推理优化高级 → KV Cache / 量化 / 推理服务 → 分布式训练（3D 并行）→ 超节点架构 | 成为大模型系统级性能优化专家 |

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

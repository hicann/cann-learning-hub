# 《启航营（两周）》课程大纲

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 启航营（两周）：CANN 算子开发系统入门 |
| **课程类型** | 短期强化集训营（高校夏令营 / 企业内训 / 竞赛集训） |
| **学时安排** | **10 个工作日 ≈ 60 小时**（理论 20h + 实践 40h；D1–D8 上午理论 2h + 下午实践 2~4h，D7 全天为调试调优上下两讲） |
| **授课对象** | 零基础新入行者、竞赛新手、跨领域转型开发者 |
| **先修要求** | Python/C/C++ 基础；了解Makefile/GIT操作等 |
| **配套讲义** | [two_weeks_course/](./two_weeks_course/)（13 份 PPT，支撑 18 讲次，部分讲次共用讲义） |
| **实践平台** | cann-learning-hub（教程型跟练 Notebook）＋ CANNJudge（判题型自动评测） |
| **结业标准** | 独立完成与 **Qwen3-1.7B 大模型匹配**的 `add_rms_norm` + `quant_matmul` 双算子开发，通过判题并接入 PyTorch 端到端验证 |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握 CANN 软件栈与昇腾硬件架构；② 掌握大模型训练 / 推理 / 优化基本方法；③ 系统掌握 Ascend C SIMD 编程模型（**Memory 矢量 C API / 基础 API 双路线 + 矩阵算子**）与调试调优方法；④ 掌握算子接入 PyTorch 全流程（单算子调用 + 入图集成）；⑤ 了解 CANNBot 智能开发模式 |
| **能力目标** | ① 独立完成矢量 / 矩阵算子的开发、调试与调优（双 API 栈）；② 算子接入 PyTorch 并完成端到端验证；③ 通过 CANNJudge 矢量 + 矩阵双判题；④ 完成与 **Qwen3-1.7B 大模型匹配**的 **`add_rms_norm` + `quant_matmul` 双算子**结业项目 |
| **素养目标** | 形成「架构认知 → 编程实现 → 调试调优 → 框架集成」一体化工程思维，具备独立查阅文档与解决算子开发问题的能力 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **讲得透**：昇腾 NPU 架构（三核异构 / 存储层次 / 数据搬运）与 CANN 软件栈的分层关系
- **写得出**：使用 Ascend C C API 与基础 API 双路线独立开发矢量算子与矩阵算子
- **调得动**：使用 profiling 工具定位性能瓶颈，通过 VF 粒度 / 双发射 / 访存优化等手段提升算子性能
- **接得进**：将自定义算子接入 PyTorch 框架，完成单算子调用与入图端到端验证
- **用得好**：借助 CANNBot 等 AI 辅助工具加速算子开发与问题排查

### 2.3 课程内容规划

| 模块 | 讲次 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| 生态与训推认知 | 第 1~4 讲 | 昇腾生态与 CANN 架构；大模型训练（SFT/RL）；部署推理；推理优化 | 训推全链路认知与体验 |
| Ascend C 编程基础 | 第 5~6 讲 | 异构计算导论；SIMD 编程模型（核函数/内存/同步/编译） | 向量加法算子完整实现 |
| 矢量算子（双 API 路线） | 第 7~10 讲 | C API 矢量编程；基础 API（Tensor API）矢量编程；双 API 同题对照 | 矢量算子独立实现 + 双 API 对照报告 |
| 矩阵算子 | 第 11~12 讲 | Matmul 高阶 API；Cube 单元与分块策略 | GEMM 算子独立实现 |
| 调试调优 | 第 13~14 讲 | 调试工具链（仿真/板调）；性能分析与优化方法 | 性能瓶颈定位与优化实践 |
| 框架集成与智能开发 | 第 15~17 讲 | PyTorch 单算子调用与入图；CANNBot 智能开发 | 算子端到端集成 |
| 结业项目 | 第 18 讲 | 双算子开发 + 集成 + 优化 + 答辩 | `add_rms_norm` + `quant_matmul`（均与 Qwen3-1.7B 匹配） |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 编程语言 | Python / C / C++ **较熟练**（指针与内存管理、结构体、模板基础；两周营算子开发以 C++ 为主） |
| 系统操作 | Linux 命令行操作（目录 / 文件 / 权限 / 进程 / 环境变量） |
| 工程工具 | Makefile / CMake 基本规则；GIT 分支与协作基本操作 |

### 3.2 领域前置（建议完成）

- 课前预习 [cann-learning-hub 快速入门](../../quick_start) 的 `cann_basics` 章节（AI 基础概念、NPU 硬件架构、CANN 软件栈）；
- PyTorch 基础（张量操作、模型定义与 forward 流程，第 2~4 讲、第 15~16 讲直接涉及）；
- 矩阵运算基础（矩阵乘法维度规则、分块乘法思想，第 11~12 讲直接涉及）。

### 3.3 环境前置

- **在线环境（推荐）**：CANNLab 云端开发环境（cann_9.0.0 py3.11-A3-arm）；训练与推理实践需 A3 系列环境；
- **本地环境（可选）**：已部署 CANN 9.0.0+ 的昇腾开发环境（Atlas A3 训练/推理系列）。

---

## 四、教学安排与理论讲义（PPT）（10 天，上午理论 + 下午实践）

| 时段 | 讲次 | 主题 | 内容要点 | 教材PPT |
|------|------|------|---------|------|
| **D1 上午** | 第 1 讲 | 昇腾 AI 产业生态与 CANN 架构基础 | 昇腾生态全景；CANN 全栈分层；Atlas 产品线 | 2h, [PPT](./two_weeks_course/1_artificial_intelligence_basics.pptx) |
| **D1 下午** | 第 2 讲 | 基于 CANN 如何训练大模型 | SFT/RL 训练概念；torchtitan 训练流程；Qwen3 基线 | 2h, [PPT](./two_weeks_course/02_03_llm_training_with_cann_2h.pptx) |
| **D2 上午** | 第 3 讲 | 基于 CANN 如何部署和推理大模型 | CANN 推理工具链；ATC 模型转换；基线推理流程 | 2h, [PPT](./two_weeks_course/04_llm_deployment_and_inference_with_cann_2h.pptx) |
| **D2 下午** | 第 4 讲 | 基于 CANN 的大模型推理优化与最佳实践 | 量化 / 算子融合 / 图模式 / KV Cache；Profiling 定位 | 2h, [PPT](./two_weeks_course/05_llm_training_and_inference_optimization_overview_with_cann_2h.pptx) |
| **D3 上午** | 第 5 讲 | 异构计算与 Ascend C 算子编程导论 | Host/Device 异构；NPU 架构概述；Ascend C 快速入门 | 2h, [PPT](./two_weeks_course/06_a2a3_heterogeneous_computing_and_ascend_c_operator_programming_introduction.pptx) |
| **D3 下午** | 第 6 讲 | Ascend C(A2/A3) SIMD 编程模型介绍 | 核函数定义；内存层级；同步机制；算子编译与 Stream | 2h, [PPT](./two_weeks_course/07_a2a3_ascend_c_simd_programming_model.pptx) |
| **D4 上午** | 第 7 讲 | Ascend C(A2/A3) Memory 矢量算子编程实践（**基于基础 API**） | 基础 API编程范式；LocalMemoryAllocator；模板化开发 | 2h, [PPT](./two_weeks_course/08_a2a3_ascend_c_simd_memory_vector_operator_programming.pptx) |
| **D4 下午** | 第 8 讲 | Memory 矢量算子（Softmax）编程实践 |  Softmax 跟练 → 独立实现 → 判题 | 4h, 共用第 7 讲 PPT |
| **D5 上午** | 第 9 讲 | Ascend C(A2/A3) 矩阵算子编程实践（基于基础 API） | Cube 单元；Matmul 高阶 API；分块与数据布局 | 2h, [PPT](./two_weeks_course/10_a2a3_ascend_c_simd_matrix_operator_programming.pptx) |
| **D5 下午** | 第 10 讲 | 矩阵算子编程实践 · 纯实践 | GEMM 跟练 → 独立实现 → 判题 | 4h, 共用第 11 讲 PPT |
| **D6 上午** | 第 11 讲 | Ascend C(A2/A3) 融合算子编程实践（**基于基础 API**） | 融合算子模式；多算子协同；Matmul+LeakyReLU 剖析 | 2h + 2h, [PPT](./two_weeks_course/09_a2a3_ascend_c_simd_fused_operator_programming.pptx) |
| **D6 下午** | 第 12 讲 | （简单）融合算子（Matmul+LeakyReLU）编程实践 | 融合算子跟练 → 独立实现 → 判题 | 4h 纯实践, 共用第 12 讲讲义 |
| **D7 上午** | 第 13 讲 | Ascend C 调试调优与最佳实践（1） | CPU 仿真调试；NPU 板上调优；常见问题定位 | 2h, [PPT](./two_weeks_course/12_a2a3_ascend_c_operator_debugging_tuning_and_best_practices.pptx) |
| **D7 下午** | 第 14 讲 | Ascend C 调试调优与最佳实践（2） | msprof 性能分析；Roofline；瓶颈优化方法 | 2h, 共用第 13 讲 PPT |
| **D8 上午** | 第 15 讲 | Ascend C 算子如何接入 PyTorch（单算子调用 + 入图验证） | 算子注册；单算子调用；AclGraph/GE 入图 | 2h, [PPT1](./two_weeks_course/14_a2a3_ascend_c_operator_pytorch_single_operator_call.pptx)、[PPT2](./two_weeks_course/15_a2a3_ascend_c_operator_graph_integration_pytorch_aclgraph_ge.pptx) |
| **D8 下午** | 第 16 讲 | Ascend C 算子接入 PyTorch 实践 · 纯实践 | 端到端集成 → 入图验证 → 判题 | 4h, 共用第 15 讲 PPT |
| **D9 上午** | 第 17 讲 | 基于 CANNBot 的 Ascend C 算子开发介绍 | CANNBot 生成 / 调试 / 优化；辅助开发实践 | 1h, [PPT](./two_weeks_course/16_cannbot_highlights_open_source_community_edition_0.5h.pptx) |
| **D9 下午–D10** | 第 18 讲 | 结业项目 + 答辩 | 双算子开发；框架集成；性能优化；成果答辩 | 2~3h + 冲刺, [PPT](./two_weeks_course/17_a2a3_two_week_bootcamp_final_project.pptx) |

> **关键节点说明：**
> - **D4 / D5 / D6 / D8 下午为 4h 纯实践**（无新理论，按跟练 → 独立实现 → 判题推进）；
> - **D5 为 C API vs 基础 API（Tensor API）同题双实现对照日**，需提交双实现对照报告（工时 / 代码量 / 性能）；
> - **D7 为调试调优（1）（2）上下两讲**，各 2h 理论 + 2h 实践，共用同一份讲义分两个专题展开；
> - **D9–D10 为结业项目冲刺期**，完成双算子 + 框架集成 + 可选优化，最终答辩展示。

---

## 五、每节课的实践内容（CANN-Learning-Hub 承载）

> 每讲实践由 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 的跟练 Notebook 承载，与理论课配对（各 2h；纯实践日为 4h）。

| 讲次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 第 1 讲 | NPU 环境初体验 | 跑通 PyTorch NPU HelloWorld，验证环境与算力 | `quick_start/cann_basics/` |
| 第 2 讲 | 大模型训练体验 | Qwen3-1.7B SFT 基线训练跑通 | `tutorials/sft_training_pipeline`、`tutorials/rl_training_pipeline` |
| 第 3 讲 | 大模型基线推理 | Qwen3 部署与基线推理跑通 | `tutorials/llm_inference/qwen3_1.7B` |
| 第 4 讲 | 推理优化实践 | Profiling 采集 + 量化 / 融合 / 图模式等优化手段验证 | `tutorials/llm_inference/qwen3_1.7B` |
| 第 5 讲 | Ascend C 快速入门 | Add 算子跟练跑通 | `tutorials/ascendc_operator_development_light/01_basic_overview` |
| 第 6 讲 | 编程模型上机 | 向量加法算子完整实现 | `tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 第 7~8讲 | 基础 API 矢量编程 | 基础 API跟练  | `tutorials/ascendc_operator_development_light/02_AscendC_basic`（基础 API 路线） |
| 第 9~10 讲 | 矩阵算子编程 | 基础 API 跟练 → GEMM 独立实现（D6 下午 4h 纯实践） | `tutorials/ascendc_operator_development_light/03_simple_operator_practice` |
| 第 11~12 讲 | 融合算子编程 | Matmul+LeakyReLU 跟练 → 独立实现 → 判题（D8 下午 4h 纯实践） | `tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 第 13~14 讲 | 调试调优实战 | 仿真 / 板调 + msprof 性能分析 → 瓶颈定位与优化 | `tutorials/ascendc_operator_development_light/04_debug` |
| 第 15~16 讲 | PyTorch 集成 | 算子注册与单算子调用 → 入图端到端验证（D8 下午 4h 纯实践） | `tutorials/ascendc_operator_development_light/02_AscendC_basic`、`tutorials/ge_development` |
| 第 17 讲 | CANNBot 智能开发 | CANNBot 生成算子 + 人工验证记录 | `tutorials/CANNBot` |
| 第 18 讲 | 结业项目 | 双算子开发 + 框架集成 + 性能优化 + 答辩（判题由 CANNJudge 承载，见第六章） | — |

> **实践目录说明：** 「实践路径」列为 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub/) 教程仓（本仓库）`quick_start/`、`tutorials/` 下的实际教程目录，进入对应目录即可跟练 Notebook。

---

## 六、课后习题与作业（CANNJudge 承载）

> 课后习题用于每讲结束后的即时巩固（在线题库 / 判题），小作业用于实践产出验收（判题型，易=跟练跑通 / 中=变体改造 / 难=独立实现）；结业大作业为课程最终产出。**当前各讲课后习题、小作业及结业判题作业的 CANNJudge 题目链接均缺失**，待平台上线后补充（各讲缺失状态见 6.1「CANNJudge 链接」列）。

### 6.1 每讲课后习题与小作业

> 「作业参考资源」列基于各讲内容标注具体参考来源，**并非全部位于同一仓库**：无前缀相对路径位于 CANN asc-devkit 仓 `examples/` 目录（<https://gitcode.com/cann/asc-devkit/tree/master/examples>，主要为算子开发类作业）；`learning-hub:` 前缀路径位于 cann-learning-hub 教程仓（<https://gitcode.com/cann/cann-learning-hub/>，主要为训推、工具链与教程类作业）；无参考资源的作业标 `—`。

| 讲次 | 课后习题（CANNJudge 在线题库） | 小作业（CANNJudge 判题） | 作业参考资源 | CANNJudge 链接 |
|------|------------------------------|------------------------|-------------|---------------|
| 第 1 讲 | Pytorch NPU 入门练习 | torch.matmul 调用实验 | torch API链接 | 待补充CANN Judge链接 |
| 第 2 讲 | SFT / RL 训练概念题 | Qwen3-1.7B SFT 基线跑通（训练流程执行） | learning-hub: `tutorials/sft_training_pipeline`、`tutorials/rl_training_pipeline` | 待补充 |
| 第 3 讲 | CANN 部署推理流程概念题 | Qwen3 基线推理跑通（baseline notebook） | learning-hub: `tutorials/llm_inference/qwen3_1.7B` | 待补充 |
| 第 4 讲 | 推理优化手段配对题（场景 → 优化手段） | Profiling 采集 + op_statistic 瓶颈分析简报 | learning-hub: `tutorials/llm_inference/qwen3_1.7B` | 待补充 |
| 第 5 讲 | Ascend C 快速入门 | SIMD Hello World和Add算子快速入门 | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 6 讲 | Ascend C SIMD编程模型 | Add 算子改造| `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 7~8 讲 | Ascend C矢量算子开发 | Add/Relu/Softmax算子 | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 9~10 讲 | Ascend C矩阵算子开发 | Matmul算子 | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 11~12 讲 | Ascend C融合算子开发 | Matmul+Relu/Gelu/LeaklyRelu 算子 | `01_simd_cpp_api/00_introduction` | 待补充 |
| 第 13~14 讲 | 调试工具链操作题（仿真 / 板调 / msprof） | 性能瓶颈定位 + 优化前后数据采集报告 | `01_simd_cpp_api/01_utilities/04_profiling`、`01_simd_cpp_api/01_utilities/08_simulator` | 待补充 |
| 第 15~16 讲 | 算子注册与入图流程题 | 自定义算子 PyTorch 端到端调用 + 入图验证 | `01_simd_cpp_api/02_features/00_framework/00_pytorch` | 待补充 |
| 第 17 讲 | CANNBot 功能概念题 | CANNBot 生成算子 + 人工验证记录 | learning-hub: `tutorials/CANNBot` | 待补充 |
| 第 18 讲 | —（结业冲刺） | —（进入结业大作业） | — | — |

### 6.2 结业大作业（三档）

> **结业双算子（`add_rms_norm` / `quant_matmul`）均需与 Qwen3-1.7B 大模型匹配**；`add_rms_norm` 结业作业由两部分构成：**第一部分 · 功能跑通**（作业 1，CANNJudge 判题通过）→ **第二部分 · 持续优化性能**（作业 3，性能提升 + 优化路径剖析）。

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发与 **Qwen3-1.7B 大模型匹配**的矢量算子 **`add_rms_norm`** ＋ 矩阵算子 **`quant_matmul`** —— 第一部分：功能跑通 | CANNJudge 双算子判题通过（正确性 + 性能不低于参考 80%） |
| **作业 2** | **必选** | 双算子**接入 PyTorch 模型** | 单算子调用正确性 + 入图端到端验证 |
| **作业 3** | 可选 | **持续优化**双算子，提升推理性能 —— 第二部分：持续优化性能 | 性能提升幅度 + 优化路径剖析证据（profiling 数据 + 改动说明） |

> **与两天营衔接：** 同源递进——两天营 = 矢量单算子入门；两周营 = 双算子（矢量 + 矩阵）+ 框架集成 + 性能优化，是两天营的系统性延伸。

---

## 七、学习建议

### 7.1 两周学习节奏

- **D1–D2「认知 + 体验」**：建立昇腾/CANN 整体认知，跑通大模型训练与推理，理解 AI 计算的上层视角
- **D3–D7「核心 + 攻坚」**：系统学习 Ascend C 算子开发，从导论 → SIMD 编程模型 → 矢量算子（双 API）→ 矩阵算子 → 调试调优（上下两讲），逐步深入；D4/D5/D6 下午纯实践是能力提升的关键期
- **D8–D10「集成 + 产出」**：完成算子 PyTorch 集成，借助 CANNBot 加速开发，最终完成双算子结业项目并答辩

### 7.2 成功要素

1. **纯实践日是分水岭**：D4/D5/D6/D8 下午的 4h 纯实践是从「听懂」到「会做」的关键，务必独立完成，不要只跟练
2. **双 API 对照日要深度思考**：D5 的 C API vs 基础 API 对照不仅是写两份代码，更要理解两种抽象层次的权衡（控制力 vs 开发效率），为后续选择合适的开发方式建立直觉
3. **性能优化要有数据支撑**：D7 的优化实践和结业可选作业都要基于 profiling 数据，记录每次改动的性能变化，形成「测量 → 分析 → 优化 → 验证」的闭环
4. **善用 CANNBot 但不依赖**：D9 引入 CANNBot 后，可用于代码生成、错误分析、优化建议，但所有 AI 生成代码必须人工验证，尤其是正确性与性能
5. **同伴互助与代码评审**：两周营时间较长，建议结对学习，互相 code review，从同伴的代码中学习不同的实现思路

### 7.3 后续学习路径

完成两周启航营后，可根据兴趣选择深入方向：

| 方向 | 推荐路径 | 目标 |
|------|---------|------|
| **算子极致性能** | Ascend C 高级原子课程（矢量极致性能 / 矩阵极致性能 / 融合算子极致性能）→ SIMD&SIMT 混合编程 | 独立开发达到理论峰值 90%+ 的高性能算子 |
| **算子工程化** | Aclnn 算子工程化开发 → 算子入图（PyTorch/AclGraph/GE）→ 通信算子自定义开发 | 具备算子库级别的工程化开发能力 |
| **框架深度开发** | TorchNPU 深度实践 → TorchAir / AclGraph → 分布式训练框架 → 图融合 PASS 开发 | 深度参与 AI 框架层开发与优化 |
| **大模型系统优化** | 大模型推理优化高级 → KV Cache / 量化 / 推理服务 → 分布式训练（3D 并行）→ 超节点架构 | 成为大模型系统级性能优化专家 |

---

## 八、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标 / 参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（本课程实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |

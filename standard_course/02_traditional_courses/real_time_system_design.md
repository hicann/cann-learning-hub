# 传统课程 · 《实时系统设计》：课程大纲

> 本文件为「实时系统设计」（传统课程清单·课程十，中国科学技术大学）**单课程落地文件**；总规划见本目录 [README.md](./README.md)。
> 课程定位：**实时/嵌入式传统课 + CANN 单元植入**——第一部分嵌入式/实时系统传统主体不动，第二部分以 6 次 × 3 课时（共 18 课时）植入 Ascend C 算子编程（前 5 次 Atlas A2/A3 主线：2 课时理论 + 1 课时实践；第 6 次 Ascend 950 新增特性：3 课时理论），第三部分以「算子开发 + 时延确定性分析」大作业收束，把实时课的核心概念（WCET、抖动、确定性）直接迁移到 NPU 实测。
> 开课成本：第二部分理论讲义、实践与作业**全部复用《AI 计算与神经网络计算架构实践》成熟素材**（见 [syllabus.md](../03_new_courses/ai_computing_and_neural_network_architecture_practice/syllabus.md)），教师零改造即可开课。

---

## 一、课程基本信息

| 项目 | 内容 |
|------|------|
| **课程名称** | 实时系统设计（Real-Time System Design） |
| **课程类型** | 传统课程 · 专业选修（嵌入式/软件/自动化方向） |
| **学时安排** | **48 学时**（16 周 × 3 学时）：第一部分 24 学时（第 1~8 周，嵌入式/实时系统传统主体）＋ 第二部分 **18 学时**（第 9~14 周，6 次 × 3 课时 Ascend C 主讲；第 1~5 次为 2 课时理论 + 1 课时实践，第 6 次为 3 课时理论）＋ 第三部分 6 学时（第 15~16 周，大作业与答辩） |
| **授课对象** | 计算机/软件/自动化/微电子相关专业本科高年级 |
| **先修要求** | C 语言、操作系统基础（进程/调度/中断）、计算机组成原理 |
| **配套讲义** | 第一部分沿用现有实时系统讲义；第二部分复用《AI 计算与神经网络计算架构实践》Part2 讲义（第 1~5 次 a2a3 系列，第 6 次 950 系列，见第四章链接） |
| **实践平台** | cann-learning-hub（跟练 Notebook）＋ CANNJudge（判题型自动评测）＋ Atlas A2/A3（或 950）环境 |
| **结业标准** | 完成第二部分算子判题作业（Add/Softmax 级）＋ 第三部分大作业（算子时延确定性分析报告，见 6.2） |

---

## 二、课程目标与课程内容规划

### 2.1 三维目标

| 目标维度 | 达成要求 |
|---------|---------|
| **知识目标** | ① 掌握实时系统核心理论：调度算法（RMS/EDF）、优先级反转与继承、WCET 与抖动；② 理解 NPU 作为「确定性执行架构」的实时友好特性（静态调度、显式内存管理、无乱序/无 cache 抖动）；③ 掌握 Ascend C SIMD 编程模型与 Memory 矢量/矩阵/融合算子开发方法；④ 了解 Ascend 950 新增特性：SIMD Reg 矢量编程、SIMT 编程模型与 SIMD/SIMT 高级编程 |
| **能力目标** | ① 能完成经典实时任务集的可调度性分析；② 能独立开发矢量/矩阵算子并通过 CANNJudge 判题；③ 能用实时系统方法论（时延分布、P99、抖动）实测分析 NPU 算子执行的确定性 |
| **素养目标** | 建立「确定性是设计出来的，不是测出来的」的实时观，并能横跨 CPU/RTOS 与 NPU 两种系统形态讨论时间可预测性 |

### 2.2 结业能力画像

完成本课程后，学员将能够：

- **算得出**：给定周期任务集，完成 RMS/EDF 可调度性判定；给定算子与数据规模，估算执行时间量级；
- **写得对**：用 Ascend C 独立开发矢量算子（Add/Softmax 级）并通过判题；
- **测得准**：对 NPU 算子进行多次采样，输出时延分布（P50/P99/最大抖动），并与 CPU 同算法对比，解释两者确定性差异的微架构根源；
- **说得清**：为什么 NPU 的「软件显式管理 + 静态调度」天然利于 WCET 分析，而 CPU 的 cache/乱序/中断是实时性三大敌人；950 的 SIMD/SIMT 融合设计为复杂控制流场景带来了什么。

### 2.3 课程内容规划

| 模块 | 周次 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| 实时系统理论（传统主体） | 第 1~8 周 | 沿用课程现有嵌入式/实时系统教学大纲 | 调度分析作业 + RTOS 实验 |
| **Ascend C 算子编程（● 主讲）** | 第 9~14 周 | 六次课：导论与编程模型 → Memory 矢量 → 矩阵与融合 → 调试调优 → PyTorch 接入与 CANN Bot → 950 新特性（Reg 矢量/SIMT/高级编程） | Add/Softmax 等算子判题 |
| 大作业收束 | 第 15~16 周 | 算子开发 + 时延确定性分析（实时视角回归） | 结业大作业报告 |

---

## 三、前置内容

### 3.1 知识前置（硬性要求）

| 类别 | 要求 |
|------|------|
| 编程语言 | C 语言较熟练（指针、结构体、内存布局） |
| 系统基础 | 操作系统（进程与调度、中断）、计算机组成原理（流水线、cache、DMA） |
| 工具 | Linux 命令行、Makefile、GIT 基本操作 |

### 3.2 领域前置（建议完成）

- 建议课前预习 cann-learning-hub 的 `quick_start/cann_basics` 章节（NPU 硬件架构、CANN 软件栈）；
- 第 14 周「950 新增特性」前，建议先读 950 SIMD/SIMT 双执行模型概述材料。

### 3.3 环境前置

- **在线环境（推荐）**：CANNLab 云端开发环境（浏览器访问即可）；
- **本地环境（可选）**：已部署 CANN 的昇腾开发环境（Atlas A2/A3 系列；950 内容无配套实践，无硬件依赖）；无硬件时可用仿真运行验证代码正确性。

---

## 四、教学安排与理论讲义（PPT）

### 4.1 第一部分：实时系统设计（24 学时，第 1~8 周，传统主体保留）

沿用课程现有嵌入式/实时系统教学大纲（具体内容待补充），本方案不改动其内容。

> **融入方式（换芯不换课）**：第一部分埋一条「确定性」伏线——传统概念讲授时以一句话对照 NPU/CANN（○/◐ 级，如 ISR 抖动 ↔ NPU 静态下发），第 8 周收拢为「为什么 NPU 天然实时友好」的过渡课，自然进入第二部分。

### 4.2 第二部分：Ascend C 算子编程（18 学时，第 9~14 周，6 次 × 3 课时 ● 主讲）

> 复用《AI 计算与神经网络计算架构实践》Part2 讲义：第 1~5 次（Atlas A2/A3 主线，2 课时理论 + 1 课时实践）+ 第 6 次（Ascend 950 新增特性，3 课时理论）。

| 周次 | 课次 | 主题 | 内容要点 | 形式 | 教材 PPT |
|------|------|------|---------|------|---------|
| 9 | 1 | 异构计算与算子编程导论、Atlas A2/A3 Ascend C SIMD 算子编程模型 | Ascend C 语言定位与核心特性；异构系统组成与交互流程；编程模型核心要素（核函数、多级内存、同步机制、多层级 API）；编译与执行全流程 | 2h 理论 + 1h 实践 | [01 导论](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/a2a3_operator_programming/01_a2a3_heterogeneous_computing_and_ascend_c_operator_programming_introduction.pptx)、[02 SIMD 编程模型](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/a2a3_operator_programming/02_a2a3_ascend_c_simd_programming_model.pptx) |
| 10 | 2 | Ascend C SIMD Memory 矢量编程 | UB 缓冲区、队列与数据搬运；掩码与尾块处理；基于 Memory 矢量 API 开发算子的基本方法 | 2h 理论 + 1h 实践 | [03 Memory 矢量](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/a2a3_operator_programming/03_a2a3_ascend_c_simd_memory_vector_operator_programming.pptx) |
| 11 | 3 | Ascend C SIMD 矩阵编程 + 融合算子编程 | Cube 单元、矩阵分块 Tiling、数据布局；GEMM 完整流程；算子融合思想与 AIC/AIV 分离式编程 | 2h 理论 + 1h 实践 | [04 矩阵](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/a2a3_operator_programming/04_a2a3_ascend_c_simd_matrix_operator_programming.pptx)、[05 融合](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/a2a3_operator_programming/05_a2a3_ascend_c_simd_fused_operator_programming.pptx) |
| 12 | 4 | Ascend C 算子调试调优与最佳实践 | 常见功能调试方法与问题定位思路；性能采集与分析工具；算子性能调优方法论与常见问题优化 | 2h 理论 + 1h 实践 | [06 调试调优](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/a2a3_operator_programming/06_a2a3_ascend_c_operator_debugging_tuning_and_best_practices.pptx) |
| 13 | 5 | Ascend C PyTorch 单算子调用与实践 + 基于 CANN Bot 的 Ascend C 算子开发与实践 | PyTorch 单算子调用原理、算子注册与绑定、调用验证；CANN Bot 核心能力与 Agent 辅助开发范式 | 2h 理论 + 1h 实践 | [07 PyTorch 单算子](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/a2a3_operator_programming/07_a2a3_ascend_c_operator_pytorch_single_operator_call.pptx)、[CANN Bot](../01_bootcamp/two_days_course/05_cannbot_highlights_open_source_community_edition_0.5h.pptx)（暂借用启航营讲义） |
| 14 | 6 | Ascend C 950 新增特性介绍：SIMD Reg 矢量编程、SIMT 编程、SIMD 与 SIMT 高级编程 | 950 SIMD/SIMT 双执行模型；Reg 矢量编程（寄存器级计算）；SIMT 执行模型与线程级编程；SIMD/SIMT 混合编程方法与高级特性 | 3h 理论 | [02_2 Reg 矢量](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/ascend950_operator_programming/Ascend950_02_ascend_c_simd_programming_2_reg_vector_operator.pptx)、[03 SIMT 编程](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/ascend950_operator_programming/Ascend950_03_ascend_c_simt_programming.pptx)、[07 SIMD&SIMT 高级编程](../03_new_courses/ai_computing_and_neural_network_architecture_practice/part2_ascend_c_operator_programming_model_and_practice_36h/ascend950_operator_programming/Ascend950_07_ascend_c_simd_simt_advanced_programming.pptx) |

### 4.3 第三部分：大作业（6 学时，第 15~16 周）

| 周次 | 主题 | 说明 |
|------|------|------|
| 15 | 大作业开发 | 见 6.2 结业大作业：算子开发判题 + **时延确定性分析报告**（实时视角回归 NPU） |
| 16 | 答辩与总结 | 报告答辩；「实时系统 × AI 计算」专题总结研讨 |

---

## 五、每节课的实践内容（复用《AI 计算与神经网络计算架构实践》实践，CANN-Learning-Hub 承载）

> 第 1~5 次课内含 1 课时实践（与理论同堂配对）；第 6 次（950 新特性）为纯理论课，无配套实践。「实践路径」为 cann-learning-hub 教程仓实际目录。

| 课次 | 实践主题 | 实践内容 | 实践路径（cann-learning-hub） |
|------|---------|---------|------------------------------|
| 1 | Ascend C 快速入门 | **易**：导论篇跟练；**中**：改写示例核函数参数并仿真运行；**难**：跑通编译全流程并解释各阶段产物 | `tutorials/ascendc_operator_development_light/01_basic_overview` |
| 2 | Memory 矢量算子 | **易**：矢量算子跟练（Add）；**中**：Add 改 Softmax/减法变体；**难**：Add 泛化判题（任意 data_len，尾块掩码） | `tutorials/ascendc_operator_development_light/02_AscendC_basic` |
| 3 | 矩阵与融合算子 | **易**：Matmul basic 跟练；**中**：修改 Tiling 分块对比性能；**难**：Matmul 判题 / Matmul+LeakyRelu 融合 | `tutorials/ascendc_operator_development_light/03_simple_operator_practice` |
| 4 | 调试调优实战 | **易**：Profile/仿真调优跟练；**中**：解读剖析报告并指出瓶颈；**难**：慢算子诊断并优化达标（附前后数据） | `tutorials/ascendc_operator_development_light/04_debug` |
| 5 | 框架接入与智能开发 | **易**：单算子调用跟练；**中**：自定义算子注册调用；**难**：CANN Bot 生成-校验-优化闭环一道判题 | `tutorials/ascendc_operator_development_light/02_AscendC_basic`、`tutorials/CANNBot` |
| 6 | —（纯理论课） | — | — |

---

## 六、课后习题与作业（CANNJudge 承载）

### 6.1 第二部分课后习题与小作业

| 课次 | 课后习题 | 小作业（CANNJudge 判题） | 作业参考资源 | CANNJudge 链接 |
|------|---------|------------------------|-------------|---------------|
| 1 | Ascend C 编程模型概念题 | SIMD Hello World 与 Add 算子快速入门 | `01_simd_cpp_api/00_introduction` | 待补充 |
| 2 | Memory 矢量编程概念题 | Add 算子、Softmax 算子判题 | `01_simd_cpp_api/00_introduction` | 待补充 |
| 3 | 矩阵/融合算子概念题 | Matmul 算子判题；Matmul+LeakyRelu 融合判题 | `01_simd_cpp_api/00_introduction/03_fusion_operation/matmul_leakyrelu_basic_api` | 待补充 |
| 4 | 调试工具链操作题 | Profile 使用方法、仿真性能统计、SIMD 算子最佳实践 | `01_simd_cpp_api/01_utilities/04_profiling`、`01_simd_cpp_api/01_utilities/08_simulator`、`01_simd_cpp_api/05_best_practices` | 待补充 |
| 5 | PyTorch 接入流程题 | Add/Softmax 单算子接入 PyTorch；CANN Bot 辅助开发一道题 | `01_simd_cpp_api/02_features/00_framework/00_pytorch`、`tutorials/CANNBot` | 待补充 |
| 6 | 950 SIMD/SIMT 特性概念题 | —（纯理论课，无判题作业） | `01_simd_cpp_api/07_tensor_api`、`03_simt_api` | — |

> 「作业参考资源」列：无前缀相对路径位于 CANN asc-devkit 仓 `examples/` 目录（<https://gitcode.com/cann/asc-devkit/tree/master/examples>）；`tutorials/` 前缀位于 cann-learning-hub 教程仓。

### 6.2 结业大作业（实时特色，三档）

| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发矢量算子（Add 泛化 / Softmax；学有余力可用 950 SIMT 方式实现变体） | CANNJudge 判题通过 |
| **作业 2** | **必选（本课特色）** | **算子时延确定性分析**：对作业 1 算子在 NPU 上多次采样（≥1000 次），输出时延分布（P50/P99/最大抖动/直方图），与 CPU 同算法对比；用第一部分 WCET/抖动语言解释两者确定性差异的微架构根源（cache/乱序/中断 vs 静态调度/显式内存管理） | 实测数据完整性 + 微架构归因分析质量 |
| **作业 3** | 可选加分 | 基于 Atlas 200I DK（或云端环境）搭建一个「实时 AI 推理」小系统：给推理任务设定时延预算，验证达标并分析抖动来源 | 系统完成度 + 时延预算达成分析 |


| 作业 | 档位 | 内容 | 考核点 |
|------|------|------|--------|
| **作业 1** | **必选** | 开发矢量算子 **`Sigmoid`** | CANNJudge 判题通过 |
| **作业 2** | **可选** | 开发矢量算子 **`add_rms_norm`**（x + bias → RMSNorm → 输出，**仅需支持 Qwen3-1.7B 模型**） | CANNJudge 判题通过 |
| **作业 3** | 可选 | 将 `add_rms_norm` 算子**接入 PyTorch Qwen3-1.7B 模型** | 单算子调用正确性 + 端到端模型验证 |
| **作业 4** | 可选 | **持续优化** `add_rms_norm` 算子，优化推理模型性能 | 性能提升幅度 + 优化路径说明（VF粒度/双发射/访存/融合等） |

---

## 七、进一步学习参考

| 资源 | 链接 | 用途 |
|------|------|------|
| Ascend C API 实现与样例 | <https://gitcode.com/cann/asc-devkit> | API 源码、API 使用示例、算子参考实现 |
| Ascend C 编程指南 | <https://asc.gitcode.com> | 编程模型、API 用法与最佳实践 |
| CANN 算子领域样例仓库 | <https://gitcode.com/cann/cann-samples/tree/master/Samples/> | 各领域算子实现样例（对标/参考实现） |
| CANN Learning Hub | <https://gitcode.com/cann/cann-learning-hub/> | 全栈教程与 Notebook 练习（实践作业载体） |
| 教材《Ascend C 异构并行程序设计》 | <https://gitcode.com/HIT1920/AscendCBook> | 异构并行程序设计教材 |

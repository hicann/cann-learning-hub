![CANN 应用调优：Dis-GMRES + MuduoXinyu](./images/readme_cover.png)

---

# CANN 应用调优（CANN 模块 5.2）

## 统一入口与统一考核

本实验在目录层面计为**一个实验（5.2，共 2 课时）**，内部包含**两条子路径**。学习请从统一入口开始，以统一考核收尾：

- 统一实验入口：[09.01_chapter_intro.ipynb](./09.01_chapter_intro.ipynb)
- 统一考核：[09.02_chapter_test.ipynb](./09.02_chapter_test.ipynb)
- 考核答案：[answer/09.02_chapter_test_answer.md](./answer/09.02_chapter_test_answer.md)
- 子路径 A：Dis-GMRES 调优（`01_distributed_gmres/`）：[01.01_chapter_intro.ipynb](./01_distributed_gmres/01.01_chapter_intro.ipynb)
- 子路径 B：MuduoXinyu Attention A/B 调优（`02_muduoxinyu_optimization/`）：[02.01_chapter_intro.ipynb](./02_muduoxinyu_optimization/02.01_chapter_intro.ipynb)

## 课程简介

本课程对应 CANN 教学体系模块 5.2（CANN 应用调优），一个实验、两条子路径：

- **子路径 A：分布式 GMRES（Dis-GMRES）调优**（`01_distributed_gmres/`）：基于 profiling 结果对 Dis-GMRES 的计算、通信、内存与调度进行瓶颈分析与调优；
- **子路径 B：MuduoXinyu Attention A/B 调优**（`02_muduoxinyu_optimization/`）：比较 MuduoXinyu 项目 attention 的两条实现路径——Path A（FP32 多算子 attention 基线）与 Path B（FP16 `aclnnIncreFlashAttentionV4` FlashAttention 单算子），完成真实 NPU 上的构建、A/B 冒烟、正确性门禁与公平性能比较，不把量化作为独立变量。

本实验只包含上述两条子路径，不新增 Vadd 路径；5.1 的故障 Add 仍作为调试载体保留在 5.1，不迁移到 5.2。

## 适用人群与前置要求

面向具备 C++、ACLNN、基础性能统计和模型推理经验的学习者。MuduoXinyu 路径还要求能够构建 NPU 后端并理解自回归解码。

## 学习目标

子路径 A（Dis-GMRES 调优）：

- 基于 profiling 结果识别 Dis-GMRES 的计算、通信、内存与调度瓶颈；
- 通过单变量实验开展计算与通信调优；
- 依据证据提出可验证的调优结论。

子路径 B（MuduoXinyu Attention A/B 调优）：

- 比较 FP32 多算子与 FP16 FlashAttention；
- 用 token exact、fallback 和调用计数检查功能；
- 用 1 次预热 + 3 次统计比较性能，不预设融合一定更快。

## 课程章节目录

### 子路径 A：Dis-GMRES 调优

章节目录：`01_distributed_gmres/`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 1.1 章节介绍 | 课程总览 | [01.01_chapter_intro.ipynb](./01_distributed_gmres/01.01_chapter_intro.ipynb) |
| 1.2 分布式 GMRES Profiling | profiling 采集与解读 | [01.02_distributed_gmres_profiling.ipynb](./01_distributed_gmres/01.02_distributed_gmres_profiling.ipynb) |
| 1.3 Baseline 与瓶颈识别 | 基线建立与瓶颈分析 | [01.03_baseline_and_bottleneck_analysis.ipynb](./01_distributed_gmres/01.03_baseline_and_bottleneck_analysis.ipynb) |
| 1.4 计算与通信调优 | 调优实验设计 | [01.04_compute_and_communication_tuning.ipynb](./01_distributed_gmres/01.04_compute_and_communication_tuning.ipynb) |
| 1.5 内存、调度与扩展性 | 内存与调度分析 | [01.05_memory_scheduling_and_scaling.ipynb](./01_distributed_gmres/01.05_memory_scheduling_and_scaling.ipynb) |
| 1.6 章节实践 | 章节测试 | [01.06_chapter_test.ipynb](./01_distributed_gmres/01.06_chapter_test.ipynb) |

### 子路径 B：MuduoXinyu Attention A/B 调优

章节目录：`02_muduoxinyu_optimization/`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 2.1 章节介绍 | A/B 链路与前置条件 | [02.01_chapter_intro.ipynb](./02_muduoxinyu_optimization/02.01_chapter_intro.ipynb) |
| 2.2 多算子调用 vs FlashAttention | 环境、补丁、构建和真实 A/B | [02.02_multi_operator_vs_flash_attention.ipynb](./02_muduoxinyu_optimization/02.02_multi_operator_vs_flash_attention.ipynb) |
| 2.3 章节测试 | 独立 A/B 综合实践 | [02.03_chapter_test.ipynb](./02_muduoxinyu_optimization/02.03_chapter_test.ipynb) |

MuduoXinyu 章节按 02.01 → 02.02 → 02.03 学习。实验需显式准备干净的 MuduoXinyu 基线仓库、模型和 tokenizer。

## 实验条件与适用范围

- **Dis-GMRES**：GMRES 外层控制在 Host CPU；SpMV/Dot/AXPY 等热点在 AI Core 执行；world1 已通过，HCCL 2/4/8 卡未验证，不得声称多卡扩展性结论。
- **MuduoXinyu**：功能与数据正确性已通过；本轮 Atlas A3 上 Path A mean `166.35666666666665` tokens/s，Path B mean `170.086` tokens/s，Path B 相对 Path A 为 `+2.24176969162644%`（`performance_valid=true`、`performance_beneficial=true`）；该结论只适用于本次 Atlas A3、当前模型/参数与运行顺序，不泛化；其他环境必须重跑公平 A/B，若新实测为负值必须如实报告，不得包装为优化成功。
- 本实验仅含两条子路径；统一考核中的性能判定遵循"基线 → 受控/公平比较 → 正确性门禁 → 性能判定"方法：Dis-GMRES 采用单变量实验，MuduoXinyu 是固定模型、输入、解码参数、环境和统计口径下的复合实现包 A/B（Path A 与 Path B 同时改变算子组织与 dtype），结论只能归因于整体实现包。

## 课程支持的硬件产品与已验证的在线体验环境

- 环境：AI 算力环境；
- 已验证硬件：Atlas A3（SoC Ascend910_9362）；
- CANNLab 镜像模板：`cann_9.0.0-py3.11-A3-arm-20260829`；
- Notebook 内核：`Python 3.11.4 (CANN)`，kernelspec 为 `python3`；
- CANNLab 使用方法：[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)；
- GitCode 在线 Notebook：`-`。

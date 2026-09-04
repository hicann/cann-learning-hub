![算子开发：SpMV + RoPE](./images/readme_cover.png)

---

# 算子开发（CANN 模块 4.3）

## 统一入口与统一考核

本实验在目录层面计为**一个实验（4.3，共 4 课时）**，内部包含**两条子路径**，不是两个实验编号。学习请从统一入口开始，以统一考核收尾：

- 统一实验入口：[06.01_chapter_intro.ipynb](./06.01_chapter_intro.ipynb)
- 统一考核：[06.02_chapter_test.ipynb](./06.02_chapter_test.ipynb)
- 考核答案：[answer/06.02_chapter_test_answer.md](./answer/06.02_chapter_test_answer.md)
- 子路径 A：SpMV 后端开发（`01_ascendc_spmv/`）：[01.01_chapter_intro.ipynb](./01_ascendc_spmv/01.01_chapter_intro.ipynb)
- 子路径 B：RoPE 算子开发（`02_rope_operator/`）：[02.01_chapter_intro.ipynb](./02_rope_operator/02.01_chapter_intro.ipynb)

## 课程简介

本课程对应 CANN 教学体系模块 4.3（算子开发），一个实验、两条子路径：

- **子路径 A：SpMV 后端开发**（`01_ascendc_spmv/`）：基于真实 Ascend C SIMD/Vector Core kernel 与 A3 真机证据，忠实教授 `Ascend-SpMV` 当前可支撑的 backend/precision/partition/persistent 原型，以及构建、正确性与性能验证方法；
- **子路径 B：RoPE 算子开发**（`02_rope_operator/`）：围绕旋转位置编码（RoPE）算子在昇腾平台上的开发，实现 SIMD（AIV / Vector Core）正式路径与 SIMT（Ascend 950 only）模板路径，并通过 Ascend C RTC（运行时编译）打通算子链路。

## 适用人群与前置要求

面向具备 C/C++、稀疏矩阵和 Ascend C 基础的学习者。RoPE 路径还要求理解张量布局、Vector Core 与 Stream 同步。

## 学习目标

子路径 A（SpMV 后端开发）：

- 理解可替换 SpMV backend 的接口与执行模型；
- 理解 mixed precision、nnz-aware partition 与 persistent CSR 的设计；
- 能够用构建与调用证据判断真实执行位置，审计 backend 正确性。

子路径 B（RoPE 算子开发）：

- 理解 RoPE pair-planar 数据布局；
- 运行 A3（SoC Ascend910_9362）SIMD/Vector Core RTC 链路（`dav-2201` 是编译目标说明，910B3 是历史实现/验证背景，均不等同于本轮 A3）；
- 对照 Ascend 950 SIMT 模板理解线程级映射，并明确其未验证状态。

## 课程章节目录

### 子路径 A：SpMV 后端开发

章节目录：`01_ascendc_spmv/`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 1.1 章节介绍 | 课程总览 | [01.01_chapter_intro.ipynb](./01_ascendc_spmv/01.01_chapter_intro.ipynb) |
| 1.2 后端接口与执行模型 | backend 接口设计 | [01.02_backend_interface_and_execution_model.ipynb](./01_ascendc_spmv/01.02_backend_interface_and_execution_model.ipynb) |
| 1.3 精度压缩与分区 | mixed precision 与分区 | [01.03_precision_and_partition.ipynb](./01_ascendc_spmv/01.03_precision_and_partition.ipynb) |
| 1.4 Persistent CSR Context | persistent 设计 | [01.04_persistent_context.ipynb](./01_ascendc_spmv/01.04_persistent_context.ipynb) |
| 1.5 构建、正确性与性能 | 构建与验证方法 | [01.05_build_validation_and_performance.ipynb](./01_ascendc_spmv/01.05_build_validation_and_performance.ipynb) |
| 1.6 章节实践 | 章节测试 | [01.06_chapter_test.ipynb](./01_ascendc_spmv/01.06_chapter_test.ipynb) |

### 子路径 B：RoPE 算子开发

章节目录：`02_rope_operator/`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 2.1 章节介绍 | RoPE 语义、SIMD/SIMT 边界 | [02.01_chapter_intro.ipynb](./02_rope_operator/02.01_chapter_intro.ipynb) |
| 2.2 SIMD 与 SIMT | SIMD RTC 真机执行与 SIMT 模板 | [02.02_rope_simd_simt.ipynb](./02_rope_operator/02.02_rope_simd_simt.ipynb) |
| 2.3 章节测试 | SIMD 真机综合实践 | [02.03_chapter_test.ipynb](./02_rope_operator/02.03_chapter_test.ipynb) |

RoPE 章节快速开始：进入课程目录后执行 `bash 02_rope_operator/src/run_rope_lab.sh --warmup 2 --repeat 5`。

## 实验条件与适用范围

- **SpMV 正式可执行路径是 SIMD/Vector Core**：已有真实 Ascend C SIMD/Vector Core kernel 与 A3 真机证据；当前**没有**可声称通过的 SpMV SIMT 真机路径。
- **RoPE SIMD/RTC 在 A3 通过**（本轮验证硬件为 Atlas A3，SoC Ascend910_9362；`dav-2201` 是编译目标说明，910B3 是历史实现/验证背景，均不等同于本轮 A3）。
- **RoPE SIMT 仅是 Ascend 950 模板**：当前 A3 不支持，状态为 `DEFERRED_UNSUPPORTED_TARGET`，不得写成已在真机通过。

## 课程支持的硬件产品与已验证的在线体验环境

- 已验证硬件：Atlas A3（SoC Ascend910_9362）；
- CANNLab 镜像模板：`cann_9.0.0-py3.11-A3-arm-20260829`；
- Notebook 内核：`Python 3.11.4 (CANN)`，kernelspec 为 `python3`；
- CANNLab 使用方法：[CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)；
- GitCode 在线 Notebook：`-`。

![OpenMP 编程](./images/readme_cover.png)

---

# OpenMP 编程（CANN 模块 3.2）

## 课程简介

本课程对应 CANN 教学体系模块 3.2，面向 CANN 与高性能计算交叉场景，以稀疏矩阵向量乘（SpMV）为主线，学习共享内存并行编程。课程以 CSR 存储格式与 SpMV 基础为起点，介绍 OpenMP 编程模型，完成 OpenMP SpMV 实现，并进行扩展性与内存带宽分析。实验材料基于 Ascend-SpMV 工程的 CPU/OpenMP 子集。

## 适合人群与前置要求

面向具备 C/C++、CSR 稀疏矩阵与并行编程基础概念的学习者，建议先完成 3.1 毕昇编译器安装课程，并确认 `clang++ -v` 输出包含 BiSheng 编译器标识；本课程是 CPU/OpenMP 共享内存并行实验，不涉及 NPU 算子开发。

## 学习目标

- 理解 CSR 存储格式与 SpMV 的基本计算模式；
- 掌握 OpenMP 并行区域、规约与调度等核心概念；
- 使用 OpenMP 实现并验证共享内存 SpMV；
- 用单变量实验分析扩展性与内存带宽，并给出有证据支撑的结论。

## 课程章节目录

章节目录：`本目录`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 2.1 章节介绍 | 课程总览 | [02.01_chapter_intro.ipynb](./02.01_chapter_intro.ipynb) |
| 2.2 CSR 与 SpMV | 存储格式与计算模式 | [02.02_csr_and_spmv_basics.ipynb](./02.02_csr_and_spmv_basics.ipynb) |
| 2.3 OpenMP 编程模型 | 并行模型与核心指令 | [02.03_openmp_programming_model.ipynb](./02.03_openmp_programming_model.ipynb) |
| 2.4 OpenMP SpMV 实现 | 实现与正确性验证 | [02.04_openmp_spmv_implementation.ipynb](./02.04_openmp_spmv_implementation.ipynb) |
| 2.5 扩展性与带宽分析 | 性能分析方法 | [02.05_scaling_and_bandwidth_analysis.ipynb](./02.05_scaling_and_bandwidth_analysis.ipynb) |
| 2.6 章节实践 | 章节测试 | [02.06_chapter_test.ipynb](./02.06_chapter_test.ipynb) |

## 课程支持的硬件产品与已验证的在线体验环境

| 项目 | 说明 |
|------|------|
| 支持硬件 | CANNLab Atlas A3 实例的 Host CPU；U1/L1 矩阵在 `static` 与 `dynamic,16` 调度下均已实测通过；NPU 不参与计算；本课程是 CPU/OpenMP 共享内存并行实验，不是 NPU 算子实验 |
| CANNLab 环境 | A3，SoC `Ascend910_9362`，CANN 9.0，Device 0 |
| 镜像模板 | `cann_9.0.0-py3.11-A3-arm-20260829` |
| Notebook 内核 | `Python 3.11.4 (CANN)`，kernelspec 为 `python3` |
| CANNLab 指南 | [CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md) |
| GitCode 在线 Notebook | - |

## 实验环境说明

- CPU 实验需要鲲鹏毕昇 Host 编译器 `clang++`、CMake 与 OpenMP；本轮已在 CANNLab A3（Ascend910_9362）Host CPU 上实测通过；历史实验记录还包含 16 核/16 进程服务器数据，历史实验记录使用的 CANN 版本为 CANN 9.0.0。
- 使用 `$ASCEND_TOOLKIT_HOME`、`$ASCEND_HOME_PATH` 等环境变量定位实际安装，不假定固定路径。
- Notebook 为教学重组内容，本地开发阶段保持未执行状态；实现、构建方式与参考结果来自对应原实验工程的 README、源码、脚本、CSV 与历史运行记录，不将历史日志或静态检查写成当前机器真机 PASS。参考耗时会随硬件、编译器版本与系统负载变化，学习时应关注正确性与变化趋势。
- 在线体验链接由仓库维护人员配置，当前记为 `-`。

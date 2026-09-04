![MPI 编程](./images/readme_cover.png)

---

# MPI 编程（CANN 模块 3.3）

## 课程简介

本课程对应 CANN 教学体系模块 3.3，以稀疏矩阵向量乘（SpMV）为主线，学习分布式内存并行编程。课程从 MPI 编程模型出发，完成 CSR 数据分区与分布式 SpMV 实现，并开展多进程扩展性分析。实验材料基于 mpi-SpMV 工程，与 3.2 OpenMP 课程形成共享内存 → 分布式内存的递进关系。

## 适合人群与前置要求

面向具备 C/C++、CSR 稀疏矩阵与并行编程基础的学习者，建议先完成 3.2 OpenMP 课程；本课程是 CPU/MPI 分布式内存并行实验，不涉及 HCCL 与多 NPU 通信。

## 学习目标

- 理解 MPI 进程模型、通信子与点对点/集合通信的基本概念；
- 掌握按 CSR 行划分计算任务的数据分布与通信设计；
- 使用 MPI 实现并验证分布式 SpMV；
- 用单变量实验分析多进程扩展性，区分通信与计算开销。

## 课程章节目录

章节目录：`本目录`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 3.1 章节介绍 | 课程总览 | [03.01_chapter_intro.ipynb](./03.01_chapter_intro.ipynb) |
| 3.2 MPI 编程模型 | 进程模型与通信原语 | [03.02_mpi_programming_model.ipynb](./03.02_mpi_programming_model.ipynb) |
| 3.3 CSR 数据分区 | 分区策略与边界数据 | [03.03_csr_data_partitioning.ipynb](./03.03_csr_data_partitioning.ipynb) |
| 3.4 分布式 SpMV 实现 | 实现与正确性验证 | [03.04_distributed_spmv_implementation.ipynb](./03.04_distributed_spmv_implementation.ipynb) |
| 3.5 MPI 扩展性分析 | 多进程性能分析 | [03.05_mpi_scaling_analysis.ipynb](./03.05_mpi_scaling_analysis.ipynb) |
| 3.6 章节实践 | 章节测试 | [03.06_chapter_test.ipynb](./03.06_chapter_test.ipynb) |

## 课程支持的硬件产品与已验证的在线体验环境

| 项目 | 说明 |
|------|------|
| 支持硬件 | CANNLab Atlas A3 实例的 Host CPU；`mpirun -np 2` 已在 A3 Host CPU 实测通过；NPU 不参与计算；本课程是 CPU/MPI 分布式内存并行实验，不是 HCCL/多 NPU 实验 |
| CANNLab 环境 | A3，SoC `Ascend910_9362`，CANN 9.0，Device 0 |
| 镜像模板 | `cann_9.0.0-py3.11-A3-arm-20260829` |
| Notebook 内核 | `Python 3.11.4 (CANN)`，kernelspec 为 `python3` |
| CANNLab 指南 | [CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md) |
| GitCode 在线 Notebook | - |

## 实验环境说明

- 实验需要 C++17、CMake、MPI C++ wrapper 与 `mpirun`；本轮已在 CANNLab A3（Ascend910_9362）Host CPU 上以 `mpirun -np 2` 实测通过；历史实验记录还包含 16 核/16 进程服务器数据，历史实验记录使用的 CANN 版本为 CANN 9.0.0。
- 使用 `$ASCEND_TOOLKIT_HOME`、`$ASCEND_HOME_PATH` 等环境变量定位实际安装，不假定固定路径。
- Notebook 为教学重组内容，本地开发阶段保持未执行状态；实现、构建方式与参考结果来自对应原实验工程的 README、源码、脚本、CSV 与历史运行记录，不将历史日志或静态检查写成当前机器真机 PASS。参考耗时会随硬件、MPI 实现与系统负载变化，学习时应关注正确性与变化趋势。
- 在线体验链接由仓库维护人员配置，当前记为 `-`。

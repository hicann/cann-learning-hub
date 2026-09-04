![并行编程原理与实践（基于国产平台）](./images/readme_cover.png)

---

# 并行编程原理与实践（基于国产平台）

## 课程简介

本课程围绕国产昇腾平台上的并行编程与应用实践展开，依次介绍毕昇编译工具链、OpenMP、MPI、AscendCL、Ascend C、HCCL，以及 CANN 应用调试、调优和集群作业部署。课程包含 10 个实验，共 22 课时，原教学体系编号为 3.1～5.3。

## 适合人群

面向具备 C/C++ 或 Python 基础，希望学习并行编程、昇腾算子调用与 CANN 应用开发的学习者。建议按照下表顺序学习；各实验的具体前置要求见对应章节 README。

## 学习目标

完成课程后，学习者能够：

- 使用毕昇编译器完成基础 Host 工程构建；
- 理解 OpenMP、MPI 与 HCCL 的并行和通信模型；
- 通过 C/C++、Python 接口调用昇腾算子，并完成 Ascend C 算子开发；
- 使用 CANN 工具定位程序问题，分析应用瓶颈；
- 理解 CANN/HPC 应用的构建部署、作业运行与结果管理流程。

## 课程支持的硬件产品

| 项目 | 要求 |
|---|---|
| 已验证硬件 | Atlas A3（SoC Ascend910_9362） |
| CANN 版本 | 9.0.0 |
| Python | 3.11 |
| 多卡要求 | HCCL 多卡实验需要 2 张及以上 NPU；单卡环境可完成 world1 学习路径 |

## 已验证的在线体验环境

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
|---|---|---|---|
| CANNLab 云开发环境 | `cann_9.0.0-py3.11-A3-arm-20260829` | Python 3.11.4（CANN） | 按章节 README 的环境说明运行 |

环境创建方法参见 [CANNLab 环境体验指南](../../../docs/CANNLab_env_experience_guide.md)。

## 课程章节目录

| 顺序 | 原实验编号 | 实验名称 | 课时 | 章节入口 |
|---|---|---|---:|---|
| 01 | 3.1 | 毕昇编译器安装 | 2 | [01_bisheng_toolchain](./01_bisheng_toolchain/) |
| 02 | 3.2 | OpenMP 编程 | 2 | [02_openmp_programming](./02_openmp_programming/) |
| 03 | 3.3 | MPI 编程 | 2 | [03_mpi_programming](./03_mpi_programming/) |
| 04 | 4.1 | ACL C/C++ 算子库调用 | 2 | [04_acl_cpp_operator_calls](./04_acl_cpp_operator_calls/) |
| 05 | 4.2 | ACL Python 算子库调用 | 2 | [05_acl_python_operator_calls](./05_acl_python_operator_calls/) |
| 06 | 4.3 | 算子开发（SpMV + RoPE） | 4 | [06_ascendc_operator_development](./06_ascendc_operator_development/) |
| 07 | 4.4 | HCCL 编程 | 2 | [07_hccl_programming](./07_hccl_programming/) |
| 08 | 5.1 | CANN 编程错误调试 | 2 | [08_cann_debugging_tools](./08_cann_debugging_tools/) |
| 09 | 5.2 | CANN 应用调优（Dis-GMRES + MuduoXinyu） | 2 | [09_cann_application_tuning](./09_cann_application_tuning/) |
| 10 | 5.3 | 集群作业部署与管理（Xyce） | 2 | [10_cluster_job_deployment](./10_cluster_job_deployment/) |

> 实验 4.3 和 5.2 各作为一个实验保留统一入口与统一考核，其内部子路径按章节 README 指引学习。

OpenMP、MPI、Ascend C、GMRES 与 Xyce 共用 `common/sparse` 中的 CSR 数据与 CPU 参考实现，各章仍保留独立的实验入口。

## 作者团队

计卫星、高建花、杨怡雪、傅俊林

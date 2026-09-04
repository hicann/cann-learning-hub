![ACL C/C++ 算子库调用](./images/readme_cover.png)

---

# ACL C/C++ 算子库调用（CANN 模块 4.1）

## 课程简介

本课程对应 CANN 教学体系模块 4.1，面向 C/C++ 开发者学习通过 ACL（AscendCL）调用昇腾算子库。课程从 ACL Runtime 与资源管理入手，分别以 ACLNN GEMM 与 ops-sparse SpMV 为实验对象，完成真实 API 生命周期的调用实践，并对比稠密与稀疏 API 的差异。实验材料基于 acl-c/GEMM-acl 与 SpMV-acl 工程。

## 适合人群与前置要求

面向具备 C/C++、CMake、矩阵乘与 COO/CSR 稀疏矩阵基础的学习者；建议先了解 ACL Runtime 的初始化、设备内存与 Stream 概念。

## 学习目标

- 理解 ACL Runtime 初始化、设备/上下文管理与资源释放流程；
- 按真实 API 生命周期调用 ACLNN GEMM 与 ops-sparse SpMV；
- 掌握稠密与稀疏算子调用 API 的差异与适用场景；
- 能够用构建与调用证据区分 CPU reference 与 ACL 算子执行路径。

## 课程章节目录

章节目录：`本目录`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 4.1 章节介绍 | 课程总览 | [04.01_chapter_intro.ipynb](./04.01_chapter_intro.ipynb) |
| 4.2 ACL Runtime 与资源 | 初始化与资源管理 | [04.02_acl_runtime_and_resources.ipynb](./04.02_acl_runtime_and_resources.ipynb) |
| 4.3 ACLNN GEMM | 稠密算子调用 | [04.03_aclnn_gemm.ipynb](./04.03_aclnn_gemm.ipynb) |
| 4.4 ops-sparse SpMV | 稀疏算子调用 | [04.04_acl_sparse_spmv.ipynb](./04.04_acl_sparse_spmv.ipynb) |
| 4.5 稠密与稀疏 API 对比 | 差异与选型 | [04.05_dense_sparse_api_comparison.ipynb](./04.05_dense_sparse_api_comparison.ipynb) |
| 4.6 章节实践 | 章节测试 | [04.06_chapter_test.ipynb](./04.06_chapter_test.ipynb) |

## 课程支持的硬件产品与已验证的在线体验环境

| 项目 | 说明 |
|------|------|
| 支持硬件 | Atlas A3（SoC Ascend910_9362）；SpMV 需另行安装与 CANN 配套的 ops-sparse |
| CANNLab 环境 | A3，SoC `Ascend910_9362`，CANN 9.0，Device 0 |
| 镜像模板 | `cann_9.0.0-py3.11-A3-arm-20260829` |
| Notebook 内核 | `Python 3.11.4 (CANN)`，kernelspec 为 `python3` |
| CANNLab 指南 | [CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md) |
| GitCode 在线 Notebook | - |

## 实验环境说明

- 实验需要匹配版本的 Ascend Toolkit 与 ACL 库；SpMV ACL 实验还需要 `ops-sparse`。本轮 ACLNN GEMM 与 ops-sparse SpMV 曾在安装配套组件的 CANNLab A3（Ascend910_9362）上通过；当前镜像若未预装 ops-sparse，需先按 `EXPERIMENT_GUIDE.md` 安装。SpMV 调用外部 ops-sparse 算子库，本课程不声称课程自写 kernel，也不声称未经 profiler 证明的 AI Core 子类型。历史实验验证硬件为 Ascend 910B/910B3，历史实验记录使用的 CANN 版本为 CANN 9.0.0。
- 使用 `$ASCEND_TOOLKIT_HOME`、`$ASCEND_HOME_PATH` 等环境变量定位实际安装，不假定固定路径。
- Notebook 为教学重组内容，本地开发阶段保持未执行状态；实现、构建方式与参考结果来自对应原实验工程的 README、源码、脚本、CSV 与历史运行记录，不将历史日志或静态检查写成当前机器真机 PASS。参考耗时会随硬件、CANN 版本与系统负载变化。
- 在线体验链接由仓库维护人员配置，当前记为 `-`。

![集群作业部署与管理：Xyce](./images/readme_cover.png)

---

# 集群作业部署与管理（CANN 模块 5.3）

## 课程简介

本课程对应 CANN 教学体系模块 5.3，学习 CANN/HPC 应用的集群部署与管理。课程以 Ascend-Xyce wrapper/adapter 为实验对象，覆盖 Xyce 应用与 Adapter 结构、依赖与构建部署、benchmark 启动与配置、日志结果管理与故障定位。实验材料基于 Ascend-Xyce wrapper/adapter 与 Ascend-GMRES 工程。

## 适合人群与前置要求

面向具备 C/C++、CMake 与 HPC 应用构建部署基础的学习者，建议先完成 5.2 CANN 应用调优课程；本课程覆盖 Ascend-Xyce wrapper/adapter 的构建部署与作业管理，不涉及完整 netlist 仿真。

## 学习目标

- 理解 Ascend-Xyce wrapper/adapter 的应用接入结构与边界；
- 完成依赖准备与构建部署，识别运行时依赖关系；
- 掌握 benchmark 启动与配置管理方法；
- 能够依据日志与结果进行故障定位，不虚构完整 netlist 仿真或调度器作业。

## 课程章节目录

章节目录：`本目录`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 10.1 章节介绍 | 课程总览 | [10.01_chapter_intro.ipynb](./10.01_chapter_intro.ipynb) |
| 10.2 Xyce 应用与 Adapter | 接入结构与边界 | [10.02_xyce_application_and_adapter.ipynb](./10.02_xyce_application_and_adapter.ipynb) |
| 10.3 依赖与构建部署 | 依赖与构建流程 | [10.03_dependency_and_build_deployment.ipynb](./10.03_dependency_and_build_deployment.ipynb) |
| 10.4 Benchmark 启动与配置 | 启动与参数管理 | [10.04_benchmark_launch_and_configuration.ipynb](./10.04_benchmark_launch_and_configuration.ipynb) |
| 10.5 结果与故障定位 | 日志、结果与排障 | [10.05_logs_results_and_troubleshooting.ipynb](./10.05_logs_results_and_troubleshooting.ipynb) |
| 10.6 章节实践 | 章节测试 | [10.06_chapter_test.ipynb](./10.06_chapter_test.ipynb) |

## 课程支持的硬件产品与已验证的在线体验环境

| 项目 | 说明 |
|------|------|
| 支持硬件 | Atlas A3（SoC Ascend910_9362）；课程 wrapper/adapter 与 Device GMRES 热点已在 A3 通过；完整 upstream Xyce 整链未实现/未验证 |
| CANNLab 环境 | A3，SoC `Ascend910_9362`，CANN 9.0，Device 0 |
| 镜像模板 | `cann_9.0.0-py3.11-A3-arm-20260829` |
| Notebook 内核 | `Python 3.11.4 (CANN)`，kernelspec 为 `python3` |
| CANNLab 指南 | [CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md) |
| GitCode 在线 Notebook | - |

## 实验环境说明

- 历史实验验证硬件为 Ascend 910B/910B3，历史实验记录使用的 CANN 版本为 CANN 9.0.0；部署实验还依赖 Xyce 及其第三方依赖的构建环境。
- 使用 `$ASCEND_TOOLKIT_HOME`、`$ASCEND_HOME_PATH` 等环境变量定位实际安装，不假定固定路径。
- Notebook 为教学重组内容，本地开发阶段保持未执行状态；实现、构建方式与参考结果来自对应原实验工程的 README、源码、脚本、CSV 与历史运行记录，不将历史日志或静态检查写成当前机器真机 PASS。课程教授 Ascend-Xyce wrapper/adapter 的依赖部署与 benchmark 管理，不虚构完整 netlist 仿真或调度器作业。
- 本课程当前只覆盖 Ascend-Xyce wrapper/adapter 的构建、部署与作业管理；本轮课程 wrapper/adapter 与 Device GMRES 热点已在 A3 通过；完整 upstream Xyce netlist、Newton/time-step、Trilinos/Epetra 与 `TYPE=ASCEND` 整链未实现/未验证，不得写成完整 Xyce 已通过。
- 在线体验链接由仓库维护人员配置，当前记为 `-`。

![毕昇编译器安装](./images/readme_cover.png)

---

# 毕昇编译器安装（CANN 模块 3.1）

## 课程简介

本课程对应 CANN 教学体系模块 3.1（毕昇编译器安装），是 CANN 3.1–5.3 十个实验的起步环境实验，为后续 OpenMP、MPI、ACL 与算子开发实验准备环境与构建知识。课程从 CANN 开发环境检查出发，验证云沙箱中的鲲鹏毕昇 Host 编译器，并区分 CANN 异构毕昇编译入口，随后完成 Host C++ 构建与 HPC 工具链依赖检查。

## 适合人群与前置要求

面向具备 C/C++ 或 Python 基础、准备进入 CANN 与高性能计算系列实验的学习者；本课程是工具链与环境准备实验，无 NPU 算子开发前置要求，适合作为 3.1–5.3 十个实验的起点。

## 学习目标

- 检查 CANN 9.0 开发环境，区分 Host 编译链与 Ascend C/AI Core 编译链；
- 区分鲲鹏毕昇 Host 编译器（`clang`/`clang++`）与 CANN 异构毕昇编译器（`bisheng`）；
- 掌握 Host C++ 构建流程与 HPC 工具链依赖的配置方法；
- 能够独立完成后续课程实验所需的环境诊断与构建准备。

## 课程章节目录

章节目录：`本目录`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 1.1 章节介绍 | 课程总览 | [01.01_chapter_intro.ipynb](./01.01_chapter_intro.ipynb) |
| 1.2 CANN 开发环境 | 环境检查与编译链区分 | [01.02_cann_development_environment.ipynb](./01.02_cann_development_environment.ipynb) |
| 1.3 毕昇编译器基础 | 编译器概念与基本用法 | [01.03_bisheng_compiler_basics.ipynb](./01.03_bisheng_compiler_basics.ipynb) |
| 1.4 Host C++ 构建流程 | Host 侧构建实践 | [01.04_host_cpp_build_workflow.ipynb](./01.04_host_cpp_build_workflow.ipynb) |
| 1.5 HPC 工具链依赖 | 依赖识别与配置 | [01.05_hpc_toolchain_dependencies.ipynb](./01.05_hpc_toolchain_dependencies.ipynb) |
| 1.6 章节实践 | 章节测试 | [01.06_chapter_test.ipynb](./01.06_chapter_test.ipynb) |

## 课程支持的硬件产品与已验证的在线体验环境

| 项目 | 说明 |
|------|------|
| 支持硬件 | 工具链检查在 CANNLab Atlas A3（SoC Ascend910_9362）环境进行 |
| CANNLab 环境 | A3，SoC `Ascend910_9362`，CANN 9.0，Device 0 |
| 镜像模板 | `cann_9.0.0-py3.11-A3-arm-20260829` |
| Notebook 内核 | `Python 3.11.4 (CANN)`，kernelspec 为 `python3` |
| CANNLab 指南 | [CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md) |
| GitCode 在线 Notebook | - |

## 实验环境说明

- CANNLab A3 镜像提供的 `bisheng` 是 CANN 异构编译器；鲲鹏毕昇 Host 编译器是独立软件包，安装后通过 `clang`/`clang++` 使用，二者不能互相替代。
- 鲲鹏毕昇 Host 编译器的下载、完整性校验和安装步骤以[官方安装指南](https://www.hikunpeng.com/document/detail/zh/kunpengdevps/compilation/ug-bisheng/kunpengbisheng_06_0005.html)为准。配置完成后执行 `hash -r`，再用 `clang -v` 与 `clang++ -v` 验证。
- 使用 `$ASCEND_TOOLKIT_HOME`、`$ASCEND_HOME_PATH` 等环境变量定位 CANN，不假定固定安装路径。
- 在线体验链接由仓库维护人员配置，当前记为 `-`。

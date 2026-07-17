# 面向高性能计算的数据结构

本课程围绕高性能计算场景中的数据组织、内存访问、并行计算和性能优化方法展开。课程通过可运行的 Ascend C 实验，帮助学习者理解数据分块、片上存储和多核并行对计算性能的影响，并掌握面向昇腾 NPU 的基础性能分析与优化方法。

## 课程适用学习人群

- 具备 C/C++ 基础语法和基本数据结构知识的学习者。
- 具备基础 Linux 命令行和 Jupyter Notebook 使用经验，希望学习 Ascend C 与高性能计算优化的学生或开发者。

## 整体学习目标

- 理解高性能计算中数据结构、数据布局与内存访问模式之间的关系。

## 课程支持的硬件产品

| 硬件产品 | 验证状态 |
| -- | -- |
| Atlas A2 系列产品 | ✅ 已验证 |

已验证软件版本：CANN 9.0.0。

## 已验证的在线体验环境

- gitcode 在线体验 Notebook
- CANNLab 云开发环境
  - NPU 镜像模板：`cann_9.0.0_py3.11-A2-arm`
  - 规格：`1*NPU 910B3 16vCPUs 32GiB`
  - Python 内核：Python 3.11.15

CANNLab 环境创建与使用方法请参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。

## 课程章节目录

### 第二章：并行计算

| Notebook | Link | 状态 |
| -- | -- | -- |
| 02.02 矩阵乘法分块优化实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structure_for_hpc&scanFilePath=contrib/tutorials/data_structure_for_hpc/02_parallel_computing/02.02_extra_ascendc_static_tensor_matmul.ipynb) | ✅ 已发布 |

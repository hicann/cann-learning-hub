![HCCL 编程](./images/readme_cover.png)

---

# HCCL 编程（CANN 模块 4.4）

## 课程简介

本课程对应 CANN 教学体系模块 4.4，学习 HCCL（华为集合通信库）多设备通信编程。课程以分布式 SpMV 为实验对象，从 HCCL rank 与通信域概念出发，梳理 Broadcast/AllGather 数据流，完成 HCCL SpMV 通信实验实现，并进行多 NPU 扩展性分析。实验材料基于 hccl-spmv 工程，是 MPI 分布式编程在昇腾多设备场景的延续。

## 适合人群与前置要求

面向具备 C/C++、MPI 分布式编程与昇腾设备基础的学习者，建议先完成 3.3 MPI 课程；本课程当前以单 rank（world1）通信域为主，多卡场景尚未完成真机验证。

## 学习目标

- 理解 HCCL rank、device、communicator 与集合通信原语；
- 建立分布式 SpMV 的 Broadcast/AllGather 数据流模型；
- 完成 HCCL 通信实验实现并验证正确性；
- 分析多 NPU 扩展性，识别通信开销对性能的影响。

## 课程章节目录

章节目录：`本目录`（含 answer / images / src 子目录）

| 章节 | 说明 | 相对链接 |
|------|------|----------|
| 7.1 章节介绍 | 课程总览 | [07.01_chapter_intro.ipynb](./07.01_chapter_intro.ipynb) |
| 7.2 Rank 与通信域 | 概念与初始化 | [07.02_hccl_rank_and_communicator.ipynb](./07.02_hccl_rank_and_communicator.ipynb) |
| 7.3 分布式 SpMV 数据流 | 通信模式设计 | [07.03_distributed_spmv_dataflow.ipynb](./07.03_distributed_spmv_dataflow.ipynb) |
| 7.4 HCCL 通信实验实现 | 实现与验证 | [07.04_hccl_spmv_implementation.ipynb](./07.04_hccl_spmv_implementation.ipynb) |
| 7.5 多 Rank 扩展性 | 多 NPU 性能分析 | [07.05_multi_npu_scaling_analysis.ipynb](./07.05_multi_npu_scaling_analysis.ipynb) |
| 7.6 章节实践 | 章节测试 | [07.06_chapter_test.ipynb](./07.06_chapter_test.ipynb) |

## 课程支持的硬件产品与已验证的在线体验环境

| 项目 | 说明 |
|------|------|
| 支持硬件 | Atlas A3（SoC Ascend910_9362）；world1（单 rank 通信域）路径已建立；HCCL 2/4/8 多卡未验证，不得写成多卡通过 |
| CANNLab 环境 | A3，SoC `Ascend910_9362`，CANN 9.0，Device 0 |
| 镜像模板 | `cann_9.0.0-py3.11-A3-arm-20260829` |
| Notebook 内核 | `Python 3.11.4 (CANN)`，kernelspec 为 `python3` |
| CANNLab 指南 | [CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md) |
| GitCode 在线 Notebook | - |

## 实验环境说明

- 实验需要匹配版本的 Ascend Toolkit 与 HCCL 库；历史实验验证硬件为 Ascend 910B/910B3，历史实验记录使用的 CANN 版本为 CANN 9.0.0。
- 当前执行边界：world1（单 rank 通信域）路径已建立；HCCL 2/4/8 多卡场景尚未完成真机验证，不得声称多 NPU 扩展性已通过。
- 使用 `$ASCEND_TOOLKIT_HOME`、`$ASCEND_HOME_PATH` 等环境变量定位实际安装，不假定固定路径。
- Notebook 为教学重组内容，本地开发阶段保持未执行状态；实现、构建方式与参考结果来自对应原实验工程的 README、源码、脚本、CSV 与历史运行记录，不将历史日志或静态检查写成当前机器真机 PASS。本课程不解释为完整 AI Core SpMV 性能实验。
- 在线体验链接由仓库维护人员配置，当前记为 `-`。

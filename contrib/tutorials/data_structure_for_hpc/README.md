![面向高性能计算的数据结构](./images/README封面.png)

------

## 课程简介

随着人工智能的普及与发展，高性能计算需求的不断增长，传统CPU串行模式下的数据结构教学已难以适配技术的发展。本课程依托于华为昇腾计算平台与CANN技术体系，构建高性能计算场景下的数据结构模块化实验。实验内容涵盖了并行计算、分布式计算、算子开发等核心内容，每个实验均提供了可运行的Ascend C程序，可结合华为ICT学院与此配套的体系化课程进行深入学习。通过实验课程，指导学习者使用Ascend C编程，面向昇腾NPU完成算子的设计、实现、调试和优化，培养从数据结构、算法设计到性能评价的系统工程能力。

## 适合人群

建议学习者具备C/C++基础和数据结构基础，但不要求事先有Ascend C编程经验。

- 对并行计算、分布式计算或算子开发感兴趣，希望系统学习数据结构与并行算法的学习者；
- 具备数据结构基础，希望进一步了解高性能场景下的数据组织、存储与并行算法实现的学习者；
- 对华为昇腾计算平台与Ascend C算子开发感兴趣的学习者。

## 学习目标

- 理解面向高性能计算的数据结构设计原理
- 掌握并行算法设计与实现的基本方法
- 具备面向华为昇腾平台的算子开发实践能力

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

### 第一章：高性能计算基础

| Notebook | Link | 状态 |
| -- | -- | -- |

### 第二章：并行计算

| Notebook | Link | 状态 |
| -- | -- | -- |
| 02.00 基于平衡二叉树的并行前缀和实现实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structure_for_hpc&scanFilePath=contrib/tutorials/data_structure_for_hpc/02_parallel_computing/02.00_intra_prefix_sum_balanced_tree.ipynb) | ✅ 已发布 |
| 02.01 基于并行广度优先遍历的网络故障诊断实验 | - | 🚧 待内测 |
| 02.02 矩阵乘法分块优化实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structure_for_hpc&scanFilePath=contrib/tutorials/data_structure_for_hpc/02_parallel_computing/02.02_extra_ascendc_static_tensor_matmul.ipynb) | ✅ 已发布 |
| 02.03 稀疏矩阵存储格式与分块计算优化实验 | - | 🚧 待内测 |

### 第三章：分布式计算

| Notebook | Link | 状态 |
| -- | -- | -- |
| 03.01 基于HCCL的分布式字符串词频统计实验 | - | 🚧 待内测 |
| 03.02 基于一致性哈希环的分布式哈希表模拟实验 | - | 🚧 待内测 |
| 03.03 分布式B+树结构动态更新与局部数据迁移实验 | - | 🚧 待内测 |
| 03.04 基于 HCCL 的 CSR 图分区实验 | - | 🚧 待内测 |

### 第四章：算子开发

| Notebook | Link | 状态 |
| -- | -- | -- |
| 04.01 低精度矩阵乘法优化实验 | - | 🚧 待内测 |
| 04.02 卷积分类头前向计算实现与优化 | - | 🚧 待内测 |
| 04.03 RMSNorm向量算子实现与流水优化 | - | 🚧 待内测 |


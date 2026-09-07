[![星闪短距通信系统仿真](./images/readmeimage.png)](https://e.huawei.com/cn/talent/learning/#/zone?customizedZoneId=RP3jalJZGJ3jm-T0hbGiDqzQ8iE)

# Nearlink SDR Simulation Course

基于 TXS-10002-2025 SparkLink SLE 标准的星闪通信链路仿真教学课程。

## 课程概述

本课程通过六个章节的实验，从物理层 GFSK 调制解调入手，逐步深入到 Polar 信道编码、MAC 帧结构、HARQ/AMC/QoS 链路保障机制，最终完成 SleNode 双节点端到端安全通信仿真，构建完整的星闪 SLE 协议栈认知体系。本课程支持的硬件型号为 CPU、NPU 910B3 和 NPU 950，且本课程已经在 gitcode notebook 在线体验环境中验证跑通无报错。

## 前置知识

建议开始实验前具备以下基础：
- Python 基本语法（函数、循环、条件、面向对象）
- 复数的表示与运算、欧拉公式、傅里叶变换的基本概念、概率统计基础概念和分贝的基本概念
- 采样与采样率的概念、调制与解调的原理和基带信号与载波的概念
- 奈奎斯特采用定理、滤波器的基础概念、误码率和信噪比的概念

## 整体学习目标

- 掌握 GFSK 从比特到 IQ 信号的调制解调全流程
- 理解 Polar 码的信道极化原理与 SC 解码算法，能够定量标定编码增益，并分析码率、码长对性能的影响
- 掌握星闪 SLE 三种 MAC 帧类型的结构与适用场景，能够完成帧级端到端仿真并进行 FER 与吞吐量分析
- 理解 HARQ 重传、AMC 自适应 MCS 与 QoS 流控背压三类链路保障机制的原理与行为
- 掌握跳频分集、接入建链与功率自适应等抗干扰与链路管理技术，理解其在衰落信道下的增益
- 能够构建 G/T 双节点端到端通信系统，掌握建链、数据交换、ECDH+AES 安全通信与 MCS 自适应跟踪的完整流程
- 具备从物理层波形、信道编码、MAC 帧交换到系统级联调的完整星闪链路仿真能力

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | CPU、NPU 910B3 和 NPU 950 |
| Python | 3.14 及以上 |
| 依赖库 | numpy、matplotlib、scipy 等 |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| cann-learning-hub 在线体验 notebook | cann_9.0.0_py3.11-A2-arm | Python 3.11.15 | 各 Notebook 表格中的"在线体验"链接可直接打开运行 |

## 推荐学习顺序

按实验 01–02 完成物理层基础：GFSK 调制与 Polar 编码，优先理解调制与信道编码的基本原理和仿真接口。
按实验 03–04 学习 MAC 帧结构与链路可靠性保障（HARQ、AMC、QoS），并与前两章的物理层结果衔接，保持仿真参数口径一致。
按实验 05 学习跳频、接入与功率自适应等高级特性，对照基础链路观察各特性带来的增益。
完成实验 06 双节点端到端仿真：FER 统计、安全链路与 MCS 自适应，串联前几章内容完成整链路验证。
建议每章按 chapter_intro → 理论 → 实验 → chapter_test 的顺序完成，chapter_test 通过后再进入下一章。

## 星闪通信链路仿真（初级）

### 第一章：GFSK 调制解调与 BER 性能分析

| Notebook | Link | 状态 |
| --- | --- | --- |
| 1.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/01_gfsk_basics/01.01_chapter_intro.ipynb) | ✅ 已发布 |
| 1.2 GFSK 调制解调完整流程 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/01_gfsk_basics/01.02_gfsk.ipynb) | ✅ 已发布 |
| 1.3 BER 扫描与参数分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/01_gfsk_basics/01.03_param_sweep.ipynb) | ✅ 已发布 |
| 1.4 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/01_gfsk_basics/01.04_chapter_test.ipynb) | ✅ 已发布 |

## 星闪通信链路仿真（中级）

### 第二章：Polar 编码与性能对比

| Notebook | Link | 状态 |
| --- | --- | --- |
| 2.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/02_polar_coding/02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 2.2 信道编码与 Polar 码原理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/02_polar_coding/02.02_polar_theory.ipynb) | ✅ 已发布 |
| 2.3 无编码 BPSK BER 扫描 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/02_polar_coding/02.03_uncoded_bpsk.ipynb) | ✅ 已发布 |
| 2.4 Polar 编码性能对比与分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/02_polar_coding/02.04_polar_compare.ipynb) | ✅ 已发布 |
| 2.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/02_polar_coding/02.05_chapter_test.ipynb) | ✅ 已发布 |

### 第三章：MAC 帧类型传输

| Notebook | Link | 状态 |
| --- | --- | --- |
| 3.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/03_mac_frames/03.01_chapter_intro.ipynb) | ✅ 已发布 |
| 3.2 MAC 帧结构与适配原理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/03_mac_frames/03.02_mac_theory.ipynb) | ✅ 已发布 |
| 3.3 MAC 帧类型传输演示 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/03_mac_frames/03.03_frame.ipynb) | ✅ 已发布 |
| 3.4 MAC 帧 FER 扫描与吞吐量分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/03_mac_frames/03.04_fer_scan.ipynb) | ✅ 已发布 |
| 3.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/03_mac_frames/03.05_chapter_test.ipynb) | ✅ 已发布 |

## 星闪通信链路仿真（高级）

### 第四章：HARQ 重传、AMC 自适应与流控

| Notebook | Link | 状态 |
| --- | --- | --- |
| 4.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/04_link_reliability/04.01_chapter_intro.ipynb) | ✅ 已发布 |
| 4.2 HARQ、AMC 与流控原理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/04_link_reliability/04.02_reliability_theory.ipynb) | ✅ 已发布 |
| 4.3 HARQ 重传仿真 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/04_link_reliability/04.03_harq.ipynb) | ✅ 已发布 |
| 4.4 AMC 跟踪与 QoS 流控 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/04_link_reliability/04.04_amc_and_qos_flow.ipynb) | ✅ 已发布 |
| 4.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/04_link_reliability/04.05_chapter_test.ipynb) | ✅ 已发布 |

### 第五章：跳频抗干扰、接入与功率控制

| Notebook | Link | 状态 |
| --- | --- | --- |
| 5.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/05_advanced_features/05.01_chapter_intro.ipynb) | ✅ 已发布 |
| 5.2 跳频、接入与功率控制原理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/05_advanced_features/05.02_advanced_theory.ipynb) | ✅ 已发布 |
| 5.3 跳频分集仿真 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/05_advanced_features/05.03_hopping.ipynb) | ✅ 已发布 |
| 5.4 接入建链仿真 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/05_advanced_features/05.04_access.ipynb) | ✅ 已发布 |
| 5.5 功率自适应仿真 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/05_advanced_features/05.05_power_adapt.ipynb) | ✅ 已发布 |
| 5.6 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/05_advanced_features/05.06_chapter_test.ipynb) | ✅ 已发布 |

### 第六章：SleNode 双节点端到端通信

| Notebook | Link | 状态 |
| --- | --- | --- |
| 6.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/06_dual_node/06.01_chapter_intro.ipynb) | ✅ 已发布 |
| 6.2 双节点基础 FER/BER 扫描 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/06_dual_node/06.02_dual_node_fer.ipynb) | ✅ 已发布 |
| 6.3 安全通信仿真 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/06_dual_node/06.03_secure_link.ipynb) | ✅ 已发布 |
| 6.4 MCS 自适应跟踪 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/06_dual_node/06.04_mcs_adapt.ipynb) | ✅ 已发布 |
| 6.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/nearlink_sdr_sim&scanFilePath=contrib/tutorials/nearlink_sdr_sim/06_dual_node/06.05_chapter_test.ipynb) | ✅ 已发布 |
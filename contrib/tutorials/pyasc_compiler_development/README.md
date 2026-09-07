[![依托CANN的编译原理课程创新实践](./images/header.png)](https://e.huawei.com/cn/talent/learning/#/zone?customizedZoneId=Z1b0GdpXDAXhEh9nj6BITx5ThRE/)

------

## 课程简介

编译原理是计算机科学与技术、软件工程等计算机大类专业的核心课程，旨在系统揭示高级编程语言如何被自动转换为机器代码这一“计算世界的神秘面纱”。随着人工智能技术的迅猛发展，以 AI 芯片（NPU）为代表的异构计算架构对编译技术提出了全新要求，如何开发高性能算子、充分释放每一分硬件潜能，已成为决定 AI 应用落地的关键一环。

本课程依托华为昇腾计算平台与 CANN 技术体系，在系统讲授经典编译理论的基础上，融入昇腾 CANN 异构计算架构与昇腾 AI 处理器的编译技术内容。课程采用“理论 + 实践”的教学模式，以 PyAsc 开源编译框架为载体，系统讲解从编译器优化、Python 前端、MLIR 中间表示（ASC-IR）到 Ascend C 目标代码生成的完整编译链路，帮助学生建立从传统编译技术到 AI 编译技术的知识桥梁。

本教程由哈尔滨工业大学陈鄞老师及其团队开发。

## 适合人群

建议学习者具备编译原理基础和 Python/C++ 基础，但不要求事先有 Ascend C 编程经验。

- 计算机科学与技术、软件工程等相关专业的本科生：作为专业核心课程，适合大二下学期至大三年级学生修读；
- 对编程语言底层实现机制、编译器工作原理有浓厚兴趣的开发者；
- 有志于从事芯片软件栈或算子库开发的开发者：可帮助学习者建立面向 AI 芯片的编程与编译知识基础。

## 学习目标

- 系统掌握编译程序的基本结构、工作流程及各阶段的原理与实现技术
- 能够独立设计并实现一个涵盖前端到后端的完整编译器原型
- 理解 AI 编译器架构与昇腾 CANN 异构计算架构的编译优化机制
- 掌握 Ascend C 算子编程语言的基本原理与应用场景
- 培养系统设计与工程实现能力，为从事系统软件、AI 框架或芯片软件栈开发奠定基础

## 课程支持的硬件产品

| 硬件产品 | 验证状态 |
| -- | -- |
| Atlas A2 系列产品 | ✅ 已验证 |
| Atlas A3 系列产品 | 🚧 待验证 |

已验证软件版本：CANN 9.0.0。

## 已验证的在线体验环境

- gitcode 在线体验 Notebook
- CANNLab 云开发环境
  - NPU 镜像模板：`cann_9.0.0_py3.11-A2-arm`
  - 规格：`1*NPU 910B3 16vCPUs 32GiB`
  - Python 内核：Python 3.11.15

CANNLab 环境创建与使用方法请参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。

## 课程章节目录

### 第一章：编译器优化

| Notebook | Link | 状态 |
| -- | -- | -- |
| 01 毕昇编译器优化验证 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/pyasc_compiler_development&scanFilePath=contrib/tutorials/pyasc_compiler_development/01_bisheng_compiler_optimization/01_bisheng_compiler_optimization.ipynb) | 🚧 待内测 |

### 第二章：PyAsc Python 前端

| Notebook | Link | 状态 |
| -- | -- | -- |
| 02 PyAsc 简单算子 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/pyasc_compiler_development&scanFilePath=contrib/tutorials/pyasc_compiler_development/02_pyasc_simple_operator/02_pyasc_simple_operator.ipynb) | 🚧 待内测 |
| 03 Python↔C 前端映射 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/pyasc_compiler_development&scanFilePath=contrib/tutorials/pyasc_compiler_development/03_python_c_mapping/03_python_c_mapping.ipynb) | 🚧 待内测 |

### 第三章：MLIR 与代码生成

| Notebook | Link | 状态 |
| -- | -- | -- |
| 04 MLIR 定义 ASC-IR | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/pyasc_compiler_development&scanFilePath=contrib/tutorials/pyasc_compiler_development/04_mlir_asc_ir/04_mlir_asc_ir.ipynb) | 🚧 待内测 |
| 05 ASC-IR→AscendC 代码生成 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/pyasc_compiler_development&scanFilePath=contrib/tutorials/pyasc_compiler_development/05_ascir_to_ascendc/05_ascir_to_ascendc.ipynb) | 🚧 待内测 |

## 参考资料

- [PyAsc 项目仓库](https://gitcode.com/cann/pyasc)
- [CANN 社区](https://gitcode.com/cann/community)
- [PyAsc 快速入门](https://gitcode.com/cann/pyasc/blob/master/docs/quick_start.md)
- [PyAsc 架构介绍](https://gitcode.com/cann/pyasc/blob/master/docs/architecture_introduction.md)

## 许可证

本教程采用 [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) 协议。

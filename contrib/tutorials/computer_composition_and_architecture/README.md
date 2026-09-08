# 计算机组成原理与体系结构

## 课程简介

“计算机组成原理与体系结构”是计算机类专业一门理论性、工程性、技术性和实践性都很强的核心专业基础课程，在计算机学科系列课程中处于承上启下的作用。课程以计算机内部总体结构为主线，涵盖数据表示表示、运算器、控制器、存储器、输入/输出系统等主要内容。详细讨论计算机组织结构、各主要功能部件的工作原理、设计与实现方法。课程着力加深学生对计算机软、硬件系统的整体化理解，建立硬件/软件协同、通用计算与智能计算相融合的计算机系统思维与设计能力。

## 核心技能

- 建立软硬协同的系统观，并能识别计算机功能部件和计算机系统设计环节的软硬协同因素；
- 建立通用计算与智能计算融合的异构计算系统观，能够基于典型异构计算硬件平台，合理分配计算任务，完成指定的异构协同计算任务；
- 能对CPU性能、高速缓冲存储器、虚拟存储器、异构计算机系统等进行性能建模分析和评价；
- 能根据任务要求进行硬件功能部件及CPU软硬协同设计。包括设计满足特定功能要求的运算器、控制器、存储器等硬件功能件及CPU，具备硬件系统的开发能力。

## 适用对象

可适配计算机类专业计算机组成原理课程使用

## 整体学习目标

以“通算-智算融合”为导向，重构系统性知识体系，通过深度整合从CPU、NPU到异构系统软件栈的全栈知识，并引入华为昇腾芯片指令集、任务调度等真实案例，帮助学生建立软硬件协同、通算智算一体的全局系统观。

## 课程实验环境

课程支持**在线体验环境**和**离线实验环境**二种模式。

### 在线体验环境（CANNLab）

**软硬件配套说明**

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 训练/推理系列产品 |
| CANN 版本 | 8.5.2 及以上 |
| Python | 3.11 |

**具体在线体验环境说明**

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 云开发环境 | cann_8.5.2_py3.11-A2-arm | Python 3.11 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建 CANNLab 环境运行 notebook |

> **注意：** Notebook 用于阅读教程和章节练习；涉及 ATC 编译、ACL/GeSession 执行的动手实践需在配备 Ascend NPU 的服务器或昇腾云环境上运行（「基础概念入门」章节除外，无需 NPU）。

### 离线实验环境（香橙派AIpro 开发板）

**硬件环境：**

| 组件 | 配置说明 |
|------|----------|
| 主控开发板 | 香橙派AIpro（8T），华为昇腾AI处理器（4核64位CPU + AI处理器）|
| 存储 | 板载32MB SPI Flash，Micro SD卡槽，≥ 32GB SD卡 |
| 内存 | 8GB / 16GB LPDDR4X |
| 系统镜像 | openEuler 22.03 或 Ubuntu 22.04（开发板出厂预装）|
| 网络连接 | 以太网或Wi-Fi，确保开发板可访问互联网 |
| PC主机 | x86_64架构，用于SSH远程连接开发板 |

**软件环境：**

| 软件组件 | 版本/说明 |
|----------|-----------|
| 操作系统 | Ubuntu 22.04 或 openEuler 22.03（开发板端）|
| CANN Toolkit | 昇腾CANN社区版 8.0.RC1 或更高版本 |
| ATC工具 | CANN Toolkit内置的模型转换工具 |
| AscendCL（ACL） | CANN运行时API库，提供Python（pyACL）和C++接口 |
| 深度学习框架 | ONNX Runtime / PyTorch ≥ 1.10（用于模型导出）|
| 编程语言 | Python 3.8+ |

> **注意：** 如需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)

**硬件组装步骤：**
1. 将SD卡插入开发板卡槽（SD卡的准备请参看“OrangePi_AI_Pro_昇腾_用户手册”）。
2. 使用USB Type-C连接线为开发板提供电源（5V/4A快充适配器或电源）。
3. 通过Mini HDMI-HDMI转接线连接显示器；若使用远程控制，可省略此步骤，直接通过SSH连接。
4. 连接键盘和鼠标（USB或蓝牙）；有线键鼠插入USB接口，蓝牙外设需在系统桌面环境下完成配对。
5. 通过网络接口连接网线或以图形界面配置Wi-Fi联网；建议优先使用有线网络以确保下载稳定。
6. 开启电源，等待系统启动；Ubuntu桌面版本会自动进入桌面环境，openEuler可能进入命令行界面；通过IP地址建立SSH远程连接。


## 计算机组成原理与体系结构（章节及实验安排）

### 第一章：计算机系统概论

1.1 现代计算机组成及工作原理

1.2 计算机系统的层次结构

1.3 性能评价指标

### 第二章：数据表示

2.1 定点数数据表示

2.2 浮点数据表示

2.3 张量数据表示

2.4 校验码

实验2：[CPU + CANN 异构环境验证](./02_representation_of_data/02.01_cpu_cann_heterogeneous_environment_verification.ipynb)

实验3：[单精度 / 半精度浮点运算正确性对比](./02_representation_of_data/02.02_fp16_vs_fp32.ipynb)

实验4：[CANN 算子开发](./02_representation_of_data/02.03_cann_operator_develop.ipynb)

### 第三章：运算方法与运算器

3.1 定点数加法与减法运算"

3.2 定点乘法运算

3.3 定点除法运算

3.4 浮点运算

3.5 运算器组织与 AI 算子设计"	

实验1：[通用运算器设计](./03_computational_methods_and_units/03.01_mips_alu_design.ipynb)

### 第四章：存储系统

4.1 存储器概述

4.2 主存储器

4.3 高速缓存存储器

4.4 虚拟存储器

实验5：[寄存器堆设计](./04_storage_system/04.01_mips_register_file_design.ipynb)

实验6：[昇腾计算语言 ACL 应用](./04_storage_system/04.02_ascend_acl_application.ipynb)

### 第五章：指令系统

5.1 指令格式

5.2 寻址方式

5.3 指令设计

实验7：[X86 + CANN异构指令系统认知与执行流程验证](./05_instruction_set_architecture/05.01_x86_and_cann_heterogeneous_instruction.ipynb)

### 第六章：中央处理器

6.1 中央处理器的功能

6.2 指令周期与时序

6.3 数据通路

6.4 控制器设计

6.5 单周期CPU设计

6.5 多周期CPU设计

6.6 异构计算的数据通路与控制

实验8：[单/多周期CPU设计](./06_central_processing_unit/06.01_cpu_design.ipynb)

实验9：[CANN 开发环境搭建与模型离线推理体验](./06_central_processing_unit/06.02_cann_development_environment_and_inference.ipynb)

实验10：[CANN 工具进行 AI 模型性能分析](./06_central_processing_unit/06.03_cann_performance_analysis.ipynb)
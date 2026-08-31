![基于昇腾CANN的智能嵌入式技术](./images/readme_cover.png)

------

## 课程简介

本课程以华为昇腾 AI 芯片为核心平台，面向新工科开设的全栈开发课程，聚焦嵌入式系统与人工智能融合，构建从硬件、操作系统、深度学习到部署落地的完整知识体系。课程涵盖智能系统概述、智能硬件系统、智能软件操作系统、深度学习开发、智能系统部署与实现、智能视觉系统开发、智能语音与语言处理、智能机器人系统开发八大模块，以理论结合实践，贯穿智能芯片、软件、计算、互联四大要素，形成完整实践闭环。课程立足国产化技术栈，突出软硬协同、工程实践与前沿技术融合，每个章节均提供可在 Notebook 中直接运行的 Lecture 理论课程与 Lab 实验课程，可结合华为 ICT 学院与此配套的体系化课程进行深入学习。通过本课程，指导学习者使用昇腾 CANN 异构计算架构与 Ascend C 编程，面向昇腾 NPU 完成从模型训练、优化转换到端侧部署的完整开发流程，培养从系统认知、算法设计到工程实践的复合型技术能力。

## 适合人群

建议学习者具备 C/C++ 与 Python 基础，以及深度学习基本概念，但不要求事先有昇腾 CANN 编程经验。可适配计算机、人工智能、信息电子、自动化、智能工程等专业智能嵌入式和智能系统等课程使用。

- 对嵌入式系统与人工智能融合感兴趣，希望系统学习昇腾全栈开发的学习者；
- 具备深度学习基础，希望进一步了解模型轻量化、量化、裁剪、蒸馏及端侧部署的学习者；
- 对华为昇腾计算平台、CANN 异构计算架构与 Ascend C 算子开发感兴趣的学习者；
- 希望掌握智能视觉、智能语音与语言处理、智能机器人系统综合开发能力的学习者。

## 学习目标

- 掌握智能系统整体架构与全流程设计能力
- 熟练进行昇腾硬件操作、接口应用开发与驱动调试
- 能设计、训练深度学习模型并完成轻量化优化（量化、裁剪、知识蒸馏）
- 精通 CANN 异构计算，实现模型在昇腾平台的转换、部署与性能调优
- 具备智能视觉、自然语言、机器人系统的综合开发与集成能力

## 课程支持的硬件产品

| 硬件产品 | 验证状态 |
| --- | --- |
| Atlas A2 系列产品（Ascend 910B3） | ✅ 已验证 |
| 香橙派 AI Pro 开发板（Ascend 310B） | ✅ 已验证 |

已验证软件版本：CANN 9.0.0。

## 已验证的在线体验环境

- gitcode 在线体验 Notebook
- CANNLab 云开发环境
  - NPU 镜像模板：`cann_9.0.0_py3.11-A2-arm`
  - 规格：`1*NPU 910B3 16vCPUs 32GiB`
  - Python 内核：Python 3.11.15

CANNLab 环境创建与使用方法请参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。

## 课程章节目录

### 第一章：智能系统概述

**理论课程（Lecture）**

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture1 智能系统概述 | 智能系统定义、意义、层级体系与应用场景；四大核心组成（芯片、软件、计算、互联）；走近昇腾 910 平台体验；智能系统开发流程；硬件、软件、算法与通信技术展望 | [查看](./01_ai_sys_intro/Lecture1/Lecture1_intelligent_system_overview.ipynb) |

**实验课程（Lab）**

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验1.1 CANNLab 开发环境搭建和测试 | 基于已配置完成的云沙箱基础环境，聚焦昇腾端侧开发关键问题点，直接进入核心实践环节 | 云沙箱 | [查看](./01_ai_sys_intro/Lab1_1/lab1.1_cann_env_setup_test.ipynb) |
| 实验1.2 昇腾香橙派开发板系统制作烧录与基础运行实验 | 香橙派 AI Pro 开发板系统镜像烧录、首次启动、网络配置与基础运行验证 | 开发板 | [查看](./01_ai_sys_intro/Lab1_2/lab1.2_orangepi_burn_and_run.ipynb) |

### 第二章：智能硬件系统

**理论课程（Lecture）**

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture2 智能硬件系统 | SoC 片上系统、CPU、协处理加速单元（NPU/GPU/DSP/FPGA）、存储器单元、常用接口；昇腾硬件系统、达芬奇 AI 处理器架构、昇腾开发应用平台 | [查看](./02_hw_sys/Lecture2/Lecture2_intelligent_hardware_system.ipynb) |

**实验课程（Lab）**

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验2.1 昇腾嵌入式 AI 开发基础探究实验 | 基于云沙箱环境，围绕昇腾嵌入式 AI 开发基础知识与核心能力展开探究 | 云沙箱 | [查看](./02_hw_sys/Lab2_1/lab2.1_ascend_embedded_ai_basic_lab.ipynb) |
| 实验2.2 昇腾香橙派开发板环境测试与部署 | 基于香橙派 AIpro 开发板，完成硬件信息查询、环境验证与程序部署三大核心任务 | 开发板 | [查看](./02_hw_sys/Lab2_2/lab2.2_ascend_orangepi_env_test.ipynb) |

### 第三章：智能软件操作系统

**理论课程（Lecture）**

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture3 智能软件操作系统 | 嵌入式操作系统概念；昇腾软件体系四层分层架构；双轨制操作系统策略；Linux 启动流程与内核定制；驱动开发三种方案；wiringOP 库接口开发；NPU-SMI 管理工具；系统构建方法；应用程序开发流程；Git 版本管理；PyTorch + torch_npu 异构计算体验 | [查看](./03_ai_os/Lecture3/Lecture3_what_is_embedded_os.ipynb) |

**实验课程（Lab）**

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验3.1 昇腾平台 GPIO 驱动与 CANN Runtime 协同控制仿真实验 | 基于仿真环境，通过 libgpiod 库完成 GPIO 外设控制，CANN Runtime 接口调用 AI 算力，实践硬件资源调度与协同运行 | 云沙箱 | [查看](./03_ai_os/Lab3_1/lab3.1_cann_gpio_control_sim.ipynb) |
| 实验3.2 昇腾香橙派 GPIO 驱动与 SPI 回环检测实验 | 基于 Orange Pi AI Pro 开发板，GPIO 驱动程序开发与 SPI 回环检测实践 | 开发板 | [查看](./03_ai_os/Lab3_2/lab3.2_orange_pi_driver.ipynb) |
| 实验3.3 昇腾 CANN 基础操作实验 | CANN、操作系统与驱动程序协同架构；CANN 四层软件栈核心模块；NPU 硬件识别、环境验证、ACL 编程体验；香橙派版本信息查询 | 云沙箱 + 开发板 | [查看](./03_ai_os/Lab3_3/lab3.3_cann_os_driver.ipynb) |

### 第四章：智能系统深度学习开发

**理论课程（Lecture）**

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture4 智能系统深度学习开发 | 深度学习网络架构体系；模型设计与实现；模型训练开发流程 | [查看](./04_dl_dev/Lecture4/Lecture4_deep_learning_development.ipynb) |

**实验课程（Lab）**

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验4.1 常见轻量化深度学习网络实验 | 轻量化网络架构设计，在保持较高精度的同时大幅减少参数量和计算量，边缘端 AI 部署关键技术 | 云沙箱 | [查看](./04_dl_dev/Lab4_1/lab4.1_lightweight_deep_learning_experiment.ipynb) |
| 实验4.2 深度学习网络量化实验 | 三种主流量化方法（动态 PTQ、静态 PTQ、QAT）完整实践，昇腾 910B 硬件适配 | 云沙箱 | [查看](./04_dl_dev/Lab4_2/lab4.2_dl_network_quantization.ipynb) |
| 实验4.3 网络裁剪（Pruning）实验 | L1 非结构化裁剪、微调恢复、QAT 量化感知训练及 INT8 转换，裁剪与量化组合压缩技术 | 云沙箱 | [查看](./04_dl_dev/Lab4_3/lab4.3_dl_network_pruning.ipynb) |
| 实验4.4 网络知识蒸馏（Knowledge Distillation）实验 | 教师网络训练、蒸馏损失设计（硬标签+软标签）、学生网络蒸馏训练与基线对比 | 云沙箱 | [查看](./04_dl_dev/Lab4_4/lab4.4_dl_network_distillation.ipynb) |
| 实验4.5 昇腾香橙派部署深度学习网络实验 | 融合轻量化、量化、裁剪、蒸馏四种技术，在香橙派 AIPro 上完成从训练到推理的全链路实践 | 开发板 | [查看](./04_dl_dev/Lab4_5/lab4.5_dl_network_orangepi.ipynb) |

### 第五章：智能系统部署与实现

**理论课程（Lecture）**

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture5 智能系统部署与实现 | 部署方案概述与部署流程；CANN 异构计算架构与核心组件详解；模型部署全流程案例；深度学习模型优化技术；AscendCL 推理验证实践 | [查看](./05_deploy/Lecture5/lecture5_intelligent_system_deployment.ipynb) |

**实验课程（Lab）**

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验5.1 Ascend C 自定义算子开发循序渐进教程 | 四个递进式实验，引导逐步理解 Ascend C 算子开发核心概念与编程范式 | 云沙箱 | [查看](./05_deploy/Lab5_1/lab5.1_ascendc_CANN_operator.ipynb) |
| 实验5.2 昇腾香橙派基于 Ascend C 的基础算子开发实验 | 基于香橙派昇腾 310B3 开发板，Python + torch_npu 完成四个递进式算子实验的代码研读、上板运行与结果验证 | 开发板 | [查看](./05_deploy/Lab5_2/lab5.2_ascendc_basic_operator_orangepi.ipynb) |
| 实验5.3 ONNX 模型到 OM 模型转换与验证云沙箱实验 | 以 SimpleCNN（MNIST）为载体，完整走通 ONNX → ATC → OM → AscendCL 推理链路，对比多种模型格式部署效果 | 云沙箱 | [查看](./05_deploy/Lab5_3/lab5.3_cann_sandbox_onnx_to_om.ipynb) |
| 实验5.4 香橙派开发板 OM 模型转换与手写数字识别 | 以 MNIST 手写数字识别为载体，在香橙派上完成 ATC 模型转换和 ACL 端侧推理 | 开发板 | [查看](./05_deploy/Lab5_4/lab5.4_orangepi_om_model_conversion.ipynb) |

### 第六章：智能视觉系统开发

**理论课程（Lecture）**

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture6 智能视觉系统开发 | 智能视觉系统概念；视觉系统开发全流程；昇腾视觉开发三大关键技术（DVPP/AIPP/模型部署）；模型部署链路 PT → ONNX → OM；昇腾 NPU 视觉推理实践；四条推理路径与性能对比 | [查看](./06_vision_dev/Lecture6/Lecture6_vision_system_dev.ipynb) |

**实验课程（Lab）**

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验6.1 昇腾 DVPP 和 AIPP 图片及视频处理实验 | DVPP 数字视频预处理与 AIPP AI 预处理两大硬件加速能力，OpenCV（CPU）与 DVPP（NPU）处理路径对比 | 云沙箱 | [查看](./06_vision_dev/Lab6_1/lab6.1_dvpp_aipp_cann.ipynb) |
| 实验6.2 基于 DVPP 与 AIPP 的 YOLO 目标检测与 DeepSort 多目标追踪 | YOLOv8 目标检测模型部署与 DeepSort 多目标追踪，DVPP 硬件视频解码与 AIPP 预处理加速 | 云沙箱 | [查看](./06_vision_dev/Lab6_2/lab6.2_dvpp_aipp_yolo_deepsort_cann.ipynb) |
| 实验6.3 昇腾香橙派的图片和视频 YOLO 目标检测实验 | YOLOv8 端侧部署，图片检测、视频检测、USB 摄像头实时检测 + HDMI 显示三个完整案例 | 开发板 | [查看](./06_vision_dev/Lab6_3/lab6.3_dvpp_aipp_yolo_orangepi.ipynb) |

### 第七章：智能语音与语言处理

**理论课程（Lecture）**

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture7 智能语音与语言处理 | 智能语音系统概述；语音采集与前处理技术；语音识别与合成；自然语言理解（NLU）；大语言模型（LLM）与昇腾部署；昇腾平台语音处理实践；性能调优与参数配置 | [查看](./07_nlp_dev/Lecture7/Lecture7_intelligent_speech_nlp.ipynb) |

**实验课程（Lab）**

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验7.1 嵌入式轻量化大模型训练实验 | 使用 LoRA（低秩适配）方法对 Qwen1.5-0.5B-Chat 进行参数高效微调，涵盖数据集构建、LoRA 配置、模型训练与效果对比 | 云沙箱 | [查看](./07_nlp_dev/Lab7_1/lab7.1_cann_sandbox_lora_train.ipynb) |
| 实验7.2 嵌入式轻量化大模型部署与测试实验 | LoRA 微调模型部署与测试，延续训练实验产物，完成大模型推理验证 | 云沙箱 | [查看](./07_nlp_dev/Lab7_2/lab7.2_cann_sandbox_lora_deploy.ipynb) |
| 实验7.3 Qwen1.5-0.5B 大语言模型在昇腾香橙派部署实验 | Qwen1.5-0.5B-Chat 在香橙派上部署，ONNX 导出 → ATC 编译 OM → AscendCL 推理完整流程 | 开发板 | [查看](./07_nlp_dev/Lab7_3/lab7.3_nlp_deploy_orangepi.ipynb) |

### 第八章：智能机器人系统开发

**理论课程（Lecture）**

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture8 智能机器人系统开发 | 智能机器人系统概述；机器人本体开发（差动驱动运动控制闭环、ROS2 话题机制）；SLAM 开发（多传感器融合、激光 SLAM、Nav2 导航） | [查看](./08_robot_dev/Lecture8/Lecture8_robot_system_overview.ipynb) |

**实验课程（Lab）**

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验8.1 基于深度强化学习的机器人避障开发实验 | 马尔可夫决策过程建模，Dueling DQN 算法实现，昇腾 NPU 上训练与推理 | 云沙箱 | [查看](./08_robot_dev/Lab8_1/lab8.1_drl_obstacle_avoidance.ipynb) |
| 实验8.2 CANN 云平台机器人开发实验案例 | 理论知识数字化，云端仿真感知-建图-规划-控制-学习完整链路，昇腾 NPU 加速对比 | 云沙箱 | [查看](./08_robot_dev/Lab8_2/lab8.2_cloud_robot_demo.ipynb) |
| 实验8.3 昇腾香橙派嵌入式智能机器人实验 | 基于 ROS2 与昇腾平台香橙派开发板，机器人运动控制、SLAM 建图、自主定位与路径导航完整流程 | 开发板 | [查看](./08_robot_dev/Lab8_3/lab8.3_ascend_orangepi_experiment.ipynb) |

## 课程配套课件

本课程配有完整的 PDF 课件，存放于 `slides/` 目录下：

| 课件 | 对应章节 | 链接 |
| --- | --- | --- |
| 第1章-智能系统概述.pdf | 第一章 | [查看](https://www.qmpan.com/f/gRgVS6/Chapter%201%20-%20Intelligent%20Systems%20Overview.pdf) |
| 第2章-智能硬件系统.pdf | 第二章 | [查看](https://www.qmpan.com/f/qgE6t8/Chapter%202%20-%20Intelligent%20Hardware%20Systems.pdf) |
| 第3章-智能软件操作系统.pdf | 第三章 | [查看](https://www.qmpan.com/f/DoyrU9/Chapter%203%20-%20Intelligent%20Software%20Operating%20Systems.pdf) |
| 第4章-智能系统深度学习开发.pdf | 第四章 | [查看](https://www.qmpan.com/f/XBG0iZ/Chapter%204%20-%20Deep%20Learning%20Development%20for%20Intelligent%20Systems.pdf) |
| 第5章 智能系统部署与实现.pdf | 第五章 | [查看](https://www.qmpan.com/f/rqNeSD/Chapter%205%20-%20Intelligent%20System%20Deployment%20and%20Implementation.pdf) |
| 第6章-智能视觉系统开发.pdf | 第六章 | [查看](https://www.qmpan.com/f/7L1Aij/Chapter%206%20-%20Intelligent%20Vision%20System%20Development.pdf) |
| 第7章-智能语音系统开发.pdf | 第七章 | [查看](https://www.qmpan.com/f/e1b0Hn/Chapter%207%20-%20Intelligent%20Speech%20System%20Development.pdf) |
| 第8章-智能机器人系统开发.pdf | 第八章 | [查看](https://www.qmpan.com/f/4nePiE/Chapter%208%20-%20Intelligent%20Robot%20System%20Development.pdf) |

## 课程开发人员

毕盛 副教授、董敏 副教授

## 环境依赖版本要求

| 依赖包 | 推荐版本 | 说明 |
|--------|----------|------|
| torch | 2.4.0 | 与 torch_npu 兼容 |
| torchvision | 0.19.0 | 与 torch 2.4.0 配对，避免 `torchvision::nms` 不存在 |
| transformers | 4.39.3 | 与 Qwen1.5 模型兼容 |
| numpy | <2.0 | NumPy 2.x 与部分 CANN API 不兼容 |
| onnxscript | latest | Lecture4/Lab8.1 ONNX 导出需要 |

> **注意**：torch 2.12.0 与 torchvision 0.22.1 不兼容（`torchvision::nms` 不存在），
> 请使用上述推荐版本组合。

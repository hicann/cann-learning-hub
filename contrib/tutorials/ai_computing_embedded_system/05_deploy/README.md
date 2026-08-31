# 第五章：智能系统部署与实现

本章节讲解智能系统部署方案与流程、CANN 异构计算架构与核心组件、Ascend C 算子开发，并通过实验完成从算子开发到模型转换部署的完整链路。

## 理论课程（Lecture）

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture5 智能系统部署与实现 | 部署方案概述与部署流程；CANN 异构计算架构与核心组件详解；模型部署全流程案例；深度学习模型优化技术；AscendCL 推理验证实践 | [查看](./Lecture5/lecture5_intelligent_system_deployment.ipynb) |

### Lecture 章节结构

- 1. 智能系统部署方案概述
- 2. 智能系统部署流程
- 3. CANN 异构计算架构
- 4. CANN 核心组件详解
- 5. CANN 应用案例：模型部署全流程
- 6. 深度学习模型优化技术
- 7. 动手实践：AscendCL 推理验证

## 实验课程（Lab）

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验5.1 Ascend C 自定义算子开发循序渐进教程 | 四个递进式实验，引导逐步理解 Ascend C 算子开发核心概念与编程范式 | 云沙箱 | [查看](./Lab5_1/lab5.1_ascendc_CANN_operator.ipynb) |
| 实验5.2 昇腾香橙派基于 Ascend C 的基础算子开发实验 | 基于香橙派昇腾 310B3 开发板，Python + torch_npu 完成四个递进式算子实验的代码研读、上板运行与结果验证 | 开发板 | [查看](./Lab5_2/lab5.2_ascendc_basic_operator_orangepi.ipynb) |
| 实验5.3 ONNX 模型到 OM 模型转换与验证云沙箱实验 | 以 SimpleCNN（MNIST）为载体，完整走通 ONNX → ATC → OM → AscendCL 推理链路，对比多种模型格式部署效果 | 云沙箱 | [查看](./Lab5_3/lab5.3_cann_sandbox_onnx_to_om.ipynb) |
| 实验5.4 香橙派开发板 OM 模型转换与手写数字识别 | 以 MNIST 手写数字识别为载体，在香橙派上完成 ATC 模型转换和 ACL 端侧推理 | 开发板 | [查看](./Lab5_4/lab5.4_orangepi_om_model_conversion.ipynb) |

## 配套课件

- [第5章 智能系统部署与实现.pdf](https://www.qmpan.com/f/rqNeSD/Chapter%205%20-%20Intelligent%20System%20Deployment%20and%20Implementation.pdf)

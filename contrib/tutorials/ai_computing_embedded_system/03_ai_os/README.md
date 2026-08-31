# 第三章：智能软件操作系统

本章节拆开昇腾的软件操作系统，讲解嵌入式操作系统概念、昇腾全栈软件体系架构、驱动开发与应用软件开发流程，并通过 PyTorch + torch_npu 感受昇腾异构计算能力。

## 理论课程（Lecture）

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture3 智能软件操作系统 | 嵌入式操作系统概念；昇腾软件体系四层分层架构；双轨制操作系统策略；Linux 启动流程与内核定制；驱动开发三种方案；wiringOP 库接口开发；NPU-SMI 管理工具；系统构建方法；应用程序开发流程；Git 版本管理；PyTorch + torch_npu 异构计算体验 | [查看](./Lecture3/Lecture3_what_is_embedded_os.ipynb) |

### Lecture 章节结构

- 1. 什么是嵌入式操作系统
- 2. 昇腾软件体系架构：分层全栈设计
- 3. 用代码感受昇腾软件环境
- 4. 昇腾双轨制操作系统策略
- 5. 嵌入式 Linux 系统启动流程
- 6. 昇腾定制 Linux 内核
- 7. Linux 操作系统驱动开发：三种方案对比
- 8. 昇腾开发板接口开发：wiringOP 库
- 9. 昇腾 NPU-SMI 系统管理工具
- 10. Linux 系统构建方法
- 11. 应用程序开发流程：从 Hello World 到远程开发
- 12. 代码版本管理：Git
- 13. 用 PyTorch + torch_npu 感受昇腾异构计算

## 实验课程（Lab）

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验3.1 昇腾平台 GPIO 驱动与 CANN Runtime 协同控制仿真实验 | 基于仿真环境，通过 libgpiod 库完成 GPIO 外设控制，CANN Runtime 接口调用 AI 算力，实践硬件资源调度与协同运行 | 云沙箱 | [查看](./Lab3_1/lab3.1_cann_gpio_control_sim.ipynb) |
| 实验3.2 昇腾香橙派 GPIO 驱动与 SPI 回环检测实验 | 基于 Orange Pi AI Pro 开发板，GPIO 驱动程序开发与 SPI 回环检测实践 | 开发板 | [查看](./Lab3_2/lab3.2_orange_pi_driver.ipynb) |
| 实验3.3 昇腾 CANN 基础操作实验 | CANN、操作系统与驱动程序协同架构；CANN 四层软件栈核心模块；NPU 硬件识别、环境验证、ACL 编程体验；香橙派版本信息查询 | 云沙箱 + 开发板 | [查看](./Lab3_3/lab3.3_cann_os_driver.ipynb) |

## 配套课件

- [第3章-智能软件操作系统.pdf](https://www.qmpan.com/f/DoyrU9/Chapter%203%20-%20Intelligent%20Software%20Operating%20Systems.pdf)

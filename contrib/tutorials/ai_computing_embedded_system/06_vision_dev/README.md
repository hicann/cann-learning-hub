# 第六章：智能视觉系统开发

本章节建立智能视觉系统开发的完整认知框架，讲解昇腾视觉开发三大关键技术（DVPP/AIPP/模型部署），并通过实验完成从图片视频处理到 YOLO 目标检测与多目标追踪的端侧部署。

## 理论课程（Lecture）

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture6 智能视觉系统开发 | 智能视觉系统概念；视觉系统开发全流程；昇腾视觉开发三大关键技术（DVPP/AIPP/模型部署）；模型部署链路 PT → ONNX → OM；昇腾 NPU 视觉推理实践；四条推理路径与性能对比 | [查看](./Lecture6/Lecture6_vision_system_dev.ipynb) |

### Lecture 章节结构

- 1. 什么是智能视觉系统
- 2. 视觉系统开发全流程
- 3. 昇腾视觉开发三大关键技术
- 4. 模型部署链路：PT → ONNX → OM
- 5. 动手实践：在昇腾 NPU 上体验视觉推理
- 6. 四条推理路径与性能对比

## 实验课程（Lab）

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验6.1 昇腾 DVPP 和 AIPP 图片及视频处理实验 | DVPP 数字视频预处理与 AIPP AI 预处理两大硬件加速能力，OpenCV（CPU）与 DVPP（NPU）处理路径对比 | 云沙箱 | [查看](./Lab6_1/lab6.1_dvpp_aipp_cann.ipynb) |
| 实验6.2 基于 DVPP 与 AIPP 的 YOLO 目标检测与 DeepSort 多目标追踪 | YOLOv8 目标检测模型部署与 DeepSort 多目标追踪，DVPP 硬件视频解码与 AIPP 预处理加速 | 云沙箱 | [查看](./Lab6_2/lab6.2_dvpp_aipp_yolo_deepsort_cann.ipynb) |
| 实验6.3 昇腾香橙派的图片和视频 YOLO 目标检测实验 | YOLOv8 端侧部署，图片检测、视频检测、USB 摄像头实时检测 + HDMI 显示三个完整案例 | 开发板 | [查看](./Lab6_3/lab6.3_dvpp_aipp_yolo_orangepi.ipynb) |

## 配套课件

- [第6章-智能视觉系统开发.pdf](https://www.qmpan.com/f/7L1Aij/Chapter%206%20-%20Intelligent%20Vision%20System%20Development.pdf)

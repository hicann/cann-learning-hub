# 第四章：智能系统深度学习开发

本章节讲解深度学习网络架构体系、模型设计与实现、模型训练开发流程，并通过系列实验掌握轻量化网络、量化、裁剪、知识蒸馏等模型压缩与端侧部署技术。

## 理论课程（Lecture）

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture4 智能系统深度学习开发 | 深度学习网络架构体系；模型设计与实现；模型训练开发流程 | [查看](./Lecture4/Lecture4_deep_learning_development.ipynb) |

### Lecture 章节结构

- 4.1 深度学习网络架构体系
- 4.2 深度学习模型设计与实现
- 4.3 深度学习模型训练开发

## 实验课程（Lab）

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验4.1 常见轻量化深度学习网络实验 | 轻量化网络架构设计，在保持较高精度的同时大幅减少参数量和计算量，边缘端 AI 部署关键技术 | 云沙箱 | [查看](./Lab4_1/lab4.1_lightweight_deep_learning_experiment.ipynb) |
| 实验4.2 深度学习网络量化实验 | 三种主流量化方法（动态 PTQ、静态 PTQ、QAT）完整实践，昇腾 910B 硬件适配 | 云沙箱 | [查看](./Lab4_2/lab4.2_dl_network_quantization.ipynb) |
| 实验4.3 网络裁剪（Pruning）实验 | L1 非结构化裁剪、微调恢复、QAT 量化感知训练及 INT8 转换，裁剪与量化组合压缩技术 | 云沙箱 | [查看](./Lab4_3/lab4.3_dl_network_pruning.ipynb) |
| 实验4.4 网络知识蒸馏（Knowledge Distillation）实验 | 教师网络训练、蒸馏损失设计（硬标签+软标签）、学生网络蒸馏训练与基线对比 | 云沙箱 | [查看](./Lab4_4/lab4.4_dl_network_distillation.ipynb) |
| 实验4.5 昇腾香橙派部署深度学习网络实验 | 融合轻量化、量化、裁剪、蒸馏四种技术，在香橙派 AIPro 上完成从训练到推理的全链路实践 | 开发板 | [查看](./Lab4_5/lab4.5_dl_network_orangepi.ipynb) |

## 配套课件

- [第4章-智能系统深度学习开发.pdf](https://www.qmpan.com/f/XBG0iZ/Chapter%204%20-%20Deep%20Learning%20Development%20for%20Intelligent%20Systems.pdf)

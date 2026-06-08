# 基于 PyPTO 实现《动手学深度学习》
## 课程作者与联系方式
- 作者：@Feixinzhx、@dmh_hmd、@laoE、@Approach1999 等。
- 邮箱：<paintstar000@outlook.com>

---

## 课程简介

本教程基于 [《动手学深度学习》](https://zh.d2l.ai/)（d2l.ai），使用 [PyPTO](https://gitcode.com/cann/pypto.git) 框架在昇腾 NPU 上实现深度学习中的核心概念与算法。通过将 d2l.ai 中的经典模型和算子从零开始用 PyPTO 实现，帮助开发者掌握基于 Tile 编程模型的高性能算子开发能力。

---

## 目录结构

```
d2l_ai_pypto/
├── 0x_<chapter_name>/               # 章节名称
│   ├── answers/                     # 练习题参考答案（可选）
│   ├── data/                        # 数据存放目录（可选）
│   ├── images/                      # 章节图片资源（可选）
│   ├── src/                         # 章节内容源码（可选）
│   ├── 0x.01_chapter_intro.ipynb    # 章节介绍
│   ├── 0x.02_<section_name>         # 小节内容（notebook/md）
│   ├── ...
│   └── README.md                    # 章节概述
├── ...                    
└── README.md                        # 课程介绍（本文件）                
```
---

## 运行环境

| 项目 | 版本 |
|------|----------|
| 硬件 | 昇腾 910C / 910B |
| pypto | 0.2.0 |
| cann | 9.0.0+|
| python | 3.11.4 |

推荐使用 CANNLab `cann_9.0.0 py3.11-A3-arm` 环境。

---

## 已发布课程

| 文档 | Link | 状态 |
|--|--|--|
| 01.01 使用 CANNLab 快速进行 pypto 开发 | [在线阅读](./01_pypto_quick_start/01.01_CANNLab_pypto_quick_start.md) | ✅ 已发布 |
| 01.02 在 CANNLab 上进行常用环境持久化配置 | [在线阅读](./01_pypto_quick_start/01.02_CANNLab_proj_env_config.md) | ✅ 已发布 |
| 01.03 使用 notebook 进行 pypto 开发 | [在线阅读](./01_pypto_quick_start/01.03_notebook_quick_start.md)  | ✅ 已发布 |
| 02.01 数据操作 | [在线阅读](./02_pypto_preliminaries/02.01_data_manipulation.ipynb) | ✅ 已发布 |
| 02.02 数据预处理 | [在线阅读](./02_pypto_preliminaries/02.02_data_preprocessing.ipynb) | ✅ 已发布 |
| 02.04 微积分 | [在线阅读](./02_pypto_preliminaries/02.04_calculus.ipynb) | ✅ 已发布 |
| 02.05 自动微分 | [在线阅读](./02_pypto_preliminaries/02.05_autograd.ipynb) | ✅ 已发布 |
| 02.06 概率 | [在线阅读](./02_pypto_preliminaries/02.06_probability.ipynb) | ✅ 已发布 |
| 03.01 线性回归（简介） | [在线阅读](./03_pypto_linear_networks/03.01_linear_regression.ipynb) | ✅ 已发布 |
| 03.02 线性回归（从零开始） | [在线阅读](./03_pypto_linear_networks/03.02_linear_regression_scratch.ipynb) | ✅ 已发布 |
| 03.03 线性回归（简洁实现） | [在线阅读](./03_pypto_linear_networks/03.03_linear_regression_concise.ipynb) | ✅ 已发布 |
| 03.04 softmax 回归（简介） | - | 🚧 编写中 |
| 03.05 图像分类数据集 | - | 🚧 编写中 |
| 03.06 Softmax 回归（从零开始） | - | 🚧 编写中 |
| 03.07 Softmax 回归（简洁实现） | - | 🚧 编写中 |

---

## 课程目录

| 章节 | 标题 | 状态 |
|--|--|--|
| 第1章 | pypto 快速开始 |  ✅ 已发布 |
| 第2章 | 预备知识（数据操作、线性代数、自动微分、概率） | 🚧 编写中 |
| 第3章 | 线性神经网络（线性回归、softmax 回归） | 🚧 编写中 |
| 第4章 | 多层感知机（MLP、激活函数、正则化） | 🚧 规划中 |
| 第5章 | 深度学习计算（层与块、参数管理、自定义层） | 🚧 规划中 |
| 第6章 | 卷积神经网络（卷积、填充、步幅、池化、LeNet） | 🚧 规划中 |
| 第7章 | 现代卷积神经网络（AlexNet、VGG、ResNet、DenseNet） | 🚧 规划中 |
| 第8章 | 循环神经网络（RNN、语言模型） | 🚧 规划中 |
| 第9章 | 现代循环神经网络（GRU、LSTM、Seq2Seq） | 🚧 规划中 |
| 第10章 | 注意力机制（多头注意力、Transformer） | 🚧 规划中 |
| 第11章 | 优化算法（SGD、动量法、Adam） | 🚧 规划中 |
| 第12章 | 计算机视觉（图像增广、目标检测、语义分割） | 🚧 规划中 |
| 第13章 | 自然语言处理（词嵌入、BERT、情感分析） | 🚧 规划中 |

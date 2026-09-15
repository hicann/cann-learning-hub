# 基于 PyPTO 实现《动手学深度学习》

## 课程作者与联系方式
- 作者：@Feixinzhx、@linking_sky、@dmh_hmd、@laoE、@Approach1999 等。
- 邮箱：[paintstar000@outlook.com](mailto:paintstar000@outlook.com)

---

## 课程简介

本教程基于 [《动手学深度学习》](https://zh.d2l.ai/)（d2l.ai），使用 [PyPTO](https://gitcode.com/cann/pypto.git) 框架在昇腾 NPU 上实现深度学习中的核心概念与算法。通过将 d2l.ai 中的经典模型和算子从零开始用 PyPTO 实现，帮助开发者掌握基于 Tile 编程模型的高性能算子开发能力。

---

## 整体学习目标

完成本教程后，你将能够：
- 熟练使用 CANNLab 开发环境并完成 PyPTO 开发与持久化配置
- 理解基于 Tile 的 PTO（Parallel Tensor/Tile Operation）编程范式
- 掌握使用 PyPTO 编写线性回归、MLP、CNN、RNN、Transformer 等经典深度学习模型核心算子与网络
- 具备在昇腾 NPU 上进行高性能融合算子开发与优化的能力

---

## 配置说明

| 项目       | 要求 |
|----------| --- |
| 推荐硬件     | Atlas A3 训练/推理系列产品（昇腾 910C）、Atlas A2 训练/推理系列产品（昇腾 910B） |
| pypto 版本 | 0.2.1 及以上 |
| CANN 版本  | 9.1.0 及以上 |
| Python   | 3.11.4 |

---

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 镜像模板                   | Python 内核 | 说明 |
| --- |----------------------------| --- | --- |
| CANNLab 云开发环境 | `cann_9.0.0 py3.11-A3-arm` | Python 3.11.4 | 推荐环境。参考 [CANNLab 环境体验指南](../../../docs/CANNLab_env_experience_guide.md) 进行环境配置与开发 |

> **注：** 目前 CANNLab 暂不支持直接创建 `cann_9.1.0` 镜像，需先选择创建 `cann_9.0.0` 环境随后在“我的环境”中进行 **CANN 包升级**（详细流程参考 [01.01_CANNLab快速开始](./01_pypto_quick_start/01.01_CANNLab_pypto_quick_start.md)）。`CANN 9.1.0` 已自带 `PyPTO 0.2.1`，升级后重启即可使用。  
> **本地体验：** 如在本地物理机环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/cann/download)。

---

## 目录结构

```text
d2l_ai_pypto/
├── 0x_<chapter_name>/               # 章节名称
│   ├── answers/                     # 练习题参考答案
│   ├── data/                        # 数据存放目录
│   ├── images/                      # 章节图片资源
│   ├── src/                         # 章节内容源码
│   ├── 0x.01_chapter_intro.ipynb    # 章节介绍
│   ├── 0x.02_<section_name>         # 小节内容（notebook/md）
│   ├── ...
│   └── README.md                    # 章节概述
├── ...                    
└── README.md                        # 课程介绍（本文件）                
```

---

## 课程目录

| 章节 | 标题 | Link | 状态 |
|---|---|---|---|
| 第1章 | pypto 快速开始 | [在线阅读](./01_pypto_quick_start/README.md) | ✅ 已发布 |
| 第2章 | 预备知识（数据操作、线性代数、自动微分、概率） | [在线阅读](./02_pypto_preliminaries/README.md) | ✅ 已发布 |
| 第3章 | 线性神经网络（线性回归、softmax 回归） | [在线阅读](./03_pypto_linear_networks/README.md) | ✅ 已发布 |
| 第4章 | 多层感知机（MLP、激活函数、正则化） | [在线阅读](./04_pypto_multilayer_perceptrons/README.md) | ✅ 已发布 |
| 第5章 | 深度学习计算（层与块、参数管理、自定义层） | [在线阅读](./05_pypto_deep_learning_computation/README.md) | ✅ 已发布 |
| 第6章 | 卷积神经网络（卷积、填充、步幅、池化、LeNet） | [在线阅读](./06_pypto_convolutional_networks/README.md) | ✅ 已发布 |
| 第7章 | 现代卷积神经网络（AlexNet、VGG、ResNet、DenseNet） | [在线阅读](./07_pypto_modern_convolutional_networks/README.md) | ✅ 已发布 |
| 第8章 | 循环神经网络（RNN、语言模型） | [在线阅读](./08_pypto_recurrent_neural_networks/README.md) | ✅ 已发布 |
| 第9章 | 现代循环神经网络（GRU、LSTM、Seq2Seq） | [在线阅读](./09_pypto_modern_recurrent_neural_networks/README.md) | ✅ 已发布 |
| 第10章 | 注意力机制（多头注意力、Transformer） | [在线阅读](./10_pypto_attention_mechanisms/README.md) | ✅ 已发布 |
| 第11章 | 优化算法（SGD、动量法、Adam） | [在线阅读](./11_pypto_optimization/README.md) | ✅ 已发布 |
| 第12章 | 计算机视觉（图像增广、目标检测、语义分割） | [在线阅读](./12_pypto_computer_vision/README.md) | ✅ 已发布 |
| 第13章 | 自然语言处理：预训练（词嵌入、BERT） | [在线阅读](./13_pypto_natural_language_processing_pretraining/README.md) | ✅ 已发布 |
| 第14章 | 自然语言处理：应用（情感分析、自然语言推断） | [在线阅读](./14_pypto_natural_language_processing_applications/README.md) | ✅ 已发布 |

---

## 反馈

如果您在学习过程中发现任何问题或有改进建议，欢迎通过 [paintstar000@outlook.com](mailto:paintstar000@outlook.com) 邮件联系或在 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) 仓库提交 Issue。

# 基于 PyPTO 实现《动手学深度学习》
## 课程作者与联系方式
- 作者：@Feixinzhx、@linking_sky、@dmh_hmd、@laoE、@Approach1999 等。
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

| 项目 | 版本             |
|------|----------------|
| 硬件 | 昇腾 910C / 910B |
| pypto | 0.2.1          |
| cann | 9.1.0+         |
| python | 3.11.4         | 

推荐使用 CANNLab `cann_9.1.0 py3.11-A3-arm` 环境。


---

## 课程目录

| 章节   | 标题                                    | Link | 状态 |
|------|---------------------------------------|--|--|
| 第1章  | pypto 快速开始                            | [在线阅读](./01_pypto_quick_start/README.md) | ✅ 已发布 |
| 第2章  | 预备知识（数据操作、线性代数、自动微分、概率）               | [在线阅读](./02_pypto_preliminaries/README.md) | ✅ 已发布 |
| 第3章  | 线性神经网络（线性回归、softmax 回归）               | [在线阅读](./03_pypto_linear_networks/README.md) | ✅ 已发布 |
| 第4章  | 多层感知机（MLP、激活函数、正则化）                   | [在线阅读](./04_pypto_multilayer_perceptrons/README.md) | ✅ 已发布 |
| 第5章  | 深度学习计算（层与块、参数管理、自定义层）                 | [在线阅读](./05_pypto_deep_learning_computation/README.md) | ✅ 已发布 |
| 第6章  | 卷积神经网络（卷积、填充、步幅、池化、LeNet）             | [在线阅读](./06_pypto_convolutional_networks/README.md) | ✅ 已发布 |
| 第7章  | 现代卷积神经网络（AlexNet、VGG、ResNet、DenseNet） | - | 🚧 编写中 |
| 第8章  | 循环神经网络（RNN、语言模型）                      | [在线阅读](./08_pypto_recurrent_neural_networks/README.md) | ✅ 已发布 |
| 第9章  | 现代循环神经网络（GRU、LSTM、Seq2Seq）            | [在线阅读](./09_pypto_modern_recurrent_neural_networks/README.md) | ✅ 已发布 |
| 第10章 | 注意力机制（多头注意力、Transformer）              | [在线阅读](./10_pypto_attention_mechanisms/README.md) | ✅ 已发布 |
| 第11章 | 优化算法（SGD、动量法、Adam）                    | [在线阅读](./11_pypto_optimization/README.md) | ✅ 已发布 |
| 第12章 | 计算机视觉（图像增广、目标检测、语义分割）                 | - | 🚧 编写中 |
| 第13章 | 自然语言处理：预训练（词嵌入、BERT）                  | - | 🚧 编写中 |
| 第14章 | 自然语言处理：应用（情感分析、自然语言推断）                | - | 🚧 编写中 |

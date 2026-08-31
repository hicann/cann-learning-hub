# 第七章：智能语音与语言处理

本章节系统介绍智能语音系统的核心技术体系，讲解大语言模型（LLM）与昇腾部署技术，并在昇腾 NPU 平台上实践 LoRA 微调训练与端侧部署。

## 理论课程（Lecture）

| Lecture | 内容概要 | 链接 |
| --- | --- | --- |
| Lecture7 智能语音与语言处理 | 智能语音系统概述；语音采集与前处理技术；语音识别与合成；自然语言理解（NLU）；大语言模型（LLM）与昇腾部署；昇腾平台语音处理实践；性能调优与参数配置 | [查看](./Lecture7/Lecture7_intelligent_speech_nlp.ipynb) |

### Lecture 章节结构

- 1. 智能语音系统概述
- 2. 语音采集与前处理技术
- 3. 语音识别与合成
- 4. 自然语言理解（NLU）
- 5. 大语言模型（LLM）与昇腾部署
- 6. 昇腾平台语音处理实践
- 7. 性能调优与参数配置

## 实验课程（Lab）

| Lab | 内容概要 | 实验环境 | 链接 |
| --- | --- | --- | --- |
| 实验7.1 嵌入式轻量化大模型训练实验 | 使用 LoRA（低秩适配）方法对 Qwen1.5-0.5B-Chat 进行参数高效微调，涵盖数据集构建、LoRA 配置、模型训练与效果对比 | 云沙箱 | [查看](./Lab7_1/lab7.1_cann_sandbox_lora_train.ipynb) |
| 实验7.2 嵌入式轻量化大模型部署与测试实验 | LoRA 微调模型部署与测试，延续训练实验产物，完成大模型推理验证 | 云沙箱 | [查看](./Lab7_2/lab7.2_cann_sandbox_lora_deploy.ipynb) |
| 实验7.3 Qwen1.5-0.5B 大语言模型在昇腾香橙派部署实验 | Qwen1.5-0.5B-Chat 在香橙派上部署，ONNX 导出 → ATC 编译 OM → AscendCL 推理完整流程 | 开发板 | [查看](./Lab7_3/lab7.3_nlp_deploy_orangepi.ipynb) |

## 配套课件

- [第7章-智能语音系统开发.pdf](https://www.qmpan.com/f/e1b0Hn/Chapter%207%20-%20Intelligent%20Speech%20System%20Development.pdf)

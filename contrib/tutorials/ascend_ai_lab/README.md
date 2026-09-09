![image.png](https://raw.gitcode.com/user-images/assets/10360319/3f467630-8dd6-427d-9662-2d5f76455daf/image.png 'image.png')

# Ascend AI Lab：昇腾 AI 实战课程

## 课程整体简介

本课程面向昇腾 AI 开发者，通过五个实战章节，依次掌握轻量级图像分类模型训练、YOLO 目标检测训练调优、Atlas 200I DK A2 端侧推理部署与自定义后处理算子集成、大模型参数高效微调与算子优化方法：

- 第 02 章：MobileNetV3 图像分类实战（深度可分离卷积、SE、h-swish，Tiny ImageNet 训练与 NPU 迁移）
- 第 03 章：YOLO 单卡训练与性能调优实战（PASCAL VOC 数据准备、XML 转 YOLO txt、单卡 Ascend NPU 训练、AMP/Warmup 调优、MSPROF 瓶颈分析，多卡 DDP 作为扩展内容）
- 第 04 章：YOLO 端侧推理与自定义算子集成实战（YOLOv5s ONNX 转 OM、PyACL 加载 OM 推理、CPU NMS baseline、Ascend C 自定义 NMS 算子 YoloNmsCustom、ACLNN 集成 NPU 后处理、MindStudio Profiling）
- 第 05 章：DeepSeek-LLM-7B LoRA 微调实战（心理健康多轮对话，LoRA + SDPA/FlashAttention）
- 第 06 章：算子优化实验（Conv+BN 算子融合，优化前/后性能对比）

## 适用学习人群

- 已掌握 Python 与 PyTorch 基础（张量、nn.Module、自动求导、训练循环、DataLoader）
- 具备 CNN / Transformer 基本概念，了解目标检测与 YOLO 基本流程
- 理解候选框、置信度、类别分数、IoU、NMS 等目标检测后处理概念
- 了解 Linux Shell、Jupyter Notebook 和基础工程目录组织方式
- 希望在昇腾 Ascend NPU 上完成模型训练、端侧推理、自定义算子开发、微调和性能优化的开发者

## 支持的硬件产品

- Ascend 910 / Ascend 910B / Ascend 910B3 / Ascend 910B4
- Atlas 200I DK A2 开发者套件
- Ascend 310B4

## 已验证的在线体验环境

| 环境 | 状态 | 说明 |
| --- | --- | --- |
| CANNLab 云开发环境 | ✅ | 见 [CANNLab 环境体验指南](./CANNLab_env_experience_guide.md)；用于第 02、03、05、06 章实验；第 03 章已验证 YOLO 单卡 NPU 训练与 MSPROF 分析；NPU 镜像模板名称：cann_9.0.0 py3.11-A2-arm；Python 内核版本：3.11.4 |
| Atlas 200I DK A2 开发板环境 | ✅ | 用于第 04 章端侧推理与自定义算子集成；已验证 Ascend 310B4、CANN Toolkit 8.0.RC1、PyACL、ATC、msopgen、opc、ACLNN、msprof/MindStudio Profiling 流程 |

## 课程章节目录

### 第 02 章：MobileNetV3 图像分类实战

| Notebook | Link | 状态 |
| --- | --- | --- |
| 2.1 章节概述 | [02.01_chapter_intro.ipynb](./02_mobilenetv3_image_classification/02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 2.2 环境配置 | [02.02_environment_setup.ipynb](./02_mobilenetv3_image_classification/02.02_environment_setup.ipynb) | ✅ 已发布 |
| 2.3 数据集探索 | [02.03_dataset_exploration.ipynb](./02_mobilenetv3_image_classification/02.03_dataset_exploration.ipynb) | ✅ 已发布 |
| 2.4 网络结构 | [02.04_model_structure.ipynb](./02_mobilenetv3_image_classification/02.04_model_structure.ipynb) | ✅ 已发布 |
| 2.5 训练流程 | [02.05_training_pipeline.ipynb](./02_mobilenetv3_image_classification/02.05_training_pipeline.ipynb) | ✅ 已发布 |
| 2.6 训练实战 | [02.06_training_practice.ipynb](./02_mobilenetv3_image_classification/02.06_training_practice.ipynb) | ✅ 已发布 |
| 2.7 评估与推理 | [02.07_evaluation_and_inference.ipynb](./02_mobilenetv3_image_classification/02.07_evaluation_and_inference.ipynb) | ✅ 已发布 |
| 2.8 NPU 迁移 | [02.08_npu_migration.ipynb](./02_mobilenetv3_image_classification/02.08_npu_migration.ipynb) | ✅ 已发布 |
| 2.9 章节实践 | [02.09_chapter_test.ipynb](./02_mobilenetv3_image_classification/02.09_chapter_test.ipynb) | ✅ 已发布 |

### 第 03 章：YOLO 单卡训练与性能调优

| Notebook | Link | 状态 |
| --- | --- | --- |
| 3.1 实验概览与环境准备 | [03.01_chapter_intro.ipynb](./03_YOLO_Training_Ascend/03.01_chapter_intro.ipynb) | ✅ 已发布 |
| 3.2 PASCAL VOC 数据集准备与 YOLO 配置 | [03.02_dataset_preparation_and_yolo_config.ipynb](./03_YOLO_Training_Ascend/03.02_dataset_preparation_and_yolo_config.ipynb) | ✅ 已发布 |
| 3.3 单卡 NPU 启动与 DDP 扩展说明 | [03.03_single_npu_launch_and_ddp_extension.ipynb](./03_YOLO_Training_Ascend/03.03_single_npu_launch_and_ddp_extension.ipynb) | ✅ 已发布 |
| 3.4 单卡训练代码结构解析 | [03.04_training_code_structure.ipynb](./03_YOLO_Training_Ascend/03.04_training_code_structure.ipynb) | ✅ 已发布 |
| 3.5 AMP 混合精度与 Warmup 调优 | [03.05_amp_warmup_tuning.ipynb](./03_YOLO_Training_Ascend/03.05_amp_warmup_tuning.ipynb) | ✅ 已发布 |
| 3.6 单卡 NPU 训练实战 | [03.06_single_npu_training_practice.ipynb](./03_YOLO_Training_Ascend/03.06_single_npu_training_practice.ipynb) | ✅ 已发布 |
| 3.7 MSPROF 性能 Profiling 与瓶颈定位 | [03.07_msprof_profiling_and_chapter_test.ipynb](./03_YOLO_Training_Ascend/03.07_msprof_profiling_and_chapter_test.ipynb) | ✅ 已发布 |

### 第 04 章：YOLO 端侧推理与自定义算子集成

| Notebook | Link | 状态 |
| --- | --- | --- |
| 4.1 开发板介绍与使用教程 | [04.01_chapter_intro.ipynb](./04_YOLO_Edge_TBE_Ascend/04.01_chapter_intro.ipynb) | ✅ 已发布 |
| 4.2 YOLO 模型准备与 OM 转换 | [04.02_yolo_model_preparation_and_om_conversion.ipynb](./04_YOLO_Edge_TBE_Ascend/04.02_yolo_model_preparation_and_om_conversion.ipynb) | ✅ 已发布 |
| 4.3 PyACL 加载 OM 模型端侧推理 | [04.03_pyacl_om_inference.ipynb](./04_YOLO_Edge_TBE_Ascend/04.03_pyacl_om_inference.ipynb) | ✅ 已发布 |
| 4.4 YOLO 后处理与 CPU NMS 基线 | [04.04_yolo_postprocess_cpu_nms_baseline.ipynb](./04_YOLO_Edge_TBE_Ascend/04.04_yolo_postprocess_cpu_nms_baseline.ipynb) | ✅ 已发布 |
| 4.5 YOLO NMS 自定义后处理算子开发 | [04.05_yolo_nms_custom_operator_development.ipynb](./04_YOLO_Edge_TBE_Ascend/04.05_yolo_nms_custom_operator_development.ipynb) | ✅ 已发布 |
| 4.6 NPU 后处理集成与异构卸载 | [04.06_npu_postprocess_integration_offload.ipynb](./04_YOLO_Edge_TBE_Ascend/04.06_npu_postprocess_integration_offload.ipynb) | ✅ 已发布 |
| 4.7 延迟对比与 MindStudio Profiling | [04.07_latency_comparison_and_profiling.ipynb](./04_YOLO_Edge_TBE_Ascend/04.07_latency_comparison_and_profiling.ipynb) | ✅ 已发布 |
| 4.8 部署验收与常见问题 | [04.08_deployment_validation_and_chapter_test.ipynb](./04_YOLO_Edge_TBE_Ascend/04.08_deployment_validation_and_chapter_test.ipynb) | ✅ 已发布 |

### 第 05 章：DeepSeek-LLM-7B LoRA 微调实战

| Notebook | Link | 状态 |
| --- | --- | --- |
| 5.1 章节概述 | [05.01_chapter_intro.ipynb](./05_deepseek_lora_finetune/05.01_chapter_intro.ipynb) | ✅ 已发布 |
| 5.2 环境配置与模型加载 | [05.02_environment_and_model_loading.ipynb](./05_deepseek_lora_finetune/05.02_environment_and_model_loading.ipynb) | ✅ 已发布 |
| 5.3 数据探索与预处理 | [05.03_data_exploration.ipynb](./05_deepseek_lora_finetune/05.03_data_exploration.ipynb) | ✅ 已发布 |
| 5.4 LoRA 微调训练 | [05.04_lora_training.ipynb](./05_deepseek_lora_finetune/05.04_lora_training.ipynb) | ✅ 已发布 |
| 5.5 模型评测与对话测试 | [05.05_evaluation_and_dialogue_test.ipynb](./05_deepseek_lora_finetune/05.05_evaluation_and_dialogue_test.ipynb) | ✅ 已发布 |
| 5.6 长对话测试 | [05.06_long_dialogue_test.ipynb](./05_deepseek_lora_finetune/05.06_long_dialogue_test.ipynb) | ✅ 已发布 |
| 5.7 章节实践 | [05.07_chapter_test.ipynb](./05_deepseek_lora_finetune/05.07_chapter_test.ipynb) | ✅ 已发布 |

### 第 06 章：算子优化实验

| Notebook | Link | 状态 |
| --- | --- | --- |
| 6.1 章节概述 | [06.01_chapter_intro.ipynb](./06_operator_optimization/06.01_chapter_intro.ipynb) | ✅ 已发布 |
| 6.2 实验总览 | [06.02_experiment_overview.ipynb](./06_operator_optimization/06.02_experiment_overview.ipynb) | ✅ 已发布 |
| 6.3 算子剖析 | [06.03_operator_analysis.ipynb](./06_operator_optimization/06.03_operator_analysis.ipynb) | ✅ 已发布 |
| 6.4 优化前基线 | [06.04_baseline_benchmark.ipynb](./06_operator_optimization/06.04_baseline_benchmark.ipynb) | ✅ 已发布 |
| 6.5 Conv+BN 融合 | [06.05_conv_bn_fusion.ipynb](./06_operator_optimization/06.05_conv_bn_fusion.ipynb) | ✅ 已发布 |
| 6.6 前后对比与结论 | [06.06_comparison_and_conclusion.ipynb](./06_operator_optimization/06.06_comparison_and_conclusion.ipynb) | ✅ 已发布 |
| 6.7 章节实践 | [06.07_chapter_test.ipynb](./06_operator_optimization/06.07_chapter_test.ipynb) | ✅ 已发布 |

## 课程内容

| 序号 | 主题 | 主要内容 | 课件 |
| --- | --- | --- | --- |
| 02 | MobileNetV3 图像分类 | 网络原理、训练流程、评估迁移与课后习题 | [02_mobilenetv3_image_classification.pptx](./slides/02_mobilenetv3_image_classification.pptx) |
| 03 | YOLO 单卡训练与性能调优 | PASCAL VOC 原始数据结构说明、XML 转 YOLO txt、YOLO 训练配置、单卡 Ascend NPU 启动、训练代码结构解析、AMP 混合精度、Warmup、Batch Size 与 DataLoader workers 调优、checkpoint 与断点恢复、MSPROF 采集训练瓶颈，多卡 DDP 作为扩展说明 | 授课 PPT 随课程材料提供 |
| 04 | YOLO 端侧推理与自定义算子集成 | YOLOv5s ONNX 到 OM 转换、PyACL 加载 OM 完成端侧推理、使用 bus.jpg 验证推理链路、建立 CPU NMS baseline、开发并安装 Ascend C 自定义 NMS 算子 YoloNmsCustom、通过 ACLNN runner 将后处理卸载至 NPU、对齐 CPU/NPU NMS 输出、使用 msprof/MindStudio Profiling 分析算子耗时 | 授课 PPT 随课程材料提供 |
| 05 | DeepSeek LoRA 微调 | LoRA 原理、SFTTrainer 训练、对话测试与课后习题 | [05_deepseek_lora.pptx](./slides/05_deepseek_lora.pptx) |
| 06 | 算子优化实验 | Conv+BN 融合原理、实验步骤、前后对比与课后习题 | [06_operator_optimization.pptx](./slides/06_operator_optimization.pptx) |

## 说明

- 章节 `answer`、`images`、`src` 目录分别存放参考答案、配图和工程源码。
- Notebook 中的命令默认以当前章节目录为工作目录执行；如果运行环境路径不同，请优先检查 CANN 环境变量、数据路径、模型路径和自定义 OPP 安装位置。

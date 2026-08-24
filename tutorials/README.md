# CANN 教程中心
本目录汇总了昇腾 CANN（Compute Architecture for Neural Networks）各核心领域的官方教程，覆盖大模型推理、算子开发、高性能编程等关键方向，助力开发者快速掌握昇腾 NPU 开发能力。

## 教程列表
| 教程名称 | 教程描述 | 访问链接 | 状态 |
|----------|----------|----------|------|
| 大模型推理系列课程 | 从大语言模型基础出发，介绍 CANN 推理仓、推理优化、量化与 Profiling，并提供 Qwen3-8B 单卡实践 | [llm_inference](./llm_inference) | ✅ 已发布 |
| Ascend C 算子开发系列教程 | 面向昇腾 NPU 的高性能算子开发全流程教程，包含 Tiling 模板化编程、算子调试、性能优化等核心内容 | [ascendc_operator_development](./ascendc_operator_development) | ✅ 已发布 |
| Ascend C 算子开发系列教程（Kernel 直调版） | 面向昇腾 NPU 的 Ascend C 算子开发教程，包含算子核函数、Tiling 计算、矩阵算子、CV 融合算子、调试调优等核心内容 | [ascendc_operator_development_light](./ascendc_operator_development_light) | ✅ 已发布 |
| MC2融合算子开发教程 | MC2融合算子开发教程，详细介绍 MC2 融合算子的概念、开发流程和实践 | [MC2_fused_operator_development](./MC2_fused_operator_development) | ✅ 已发布 |
| Conv 算子开发实战教程 | 面向昇腾 NPU 的 Conv 算子开发实战教程，覆盖卷积算子开发核心概念与实践 | [conv_operator_development](./conv_operator_development) | ✅ 已发布 |
| HIXL 应用开发系列教程 | 基于昇腾单边通信库的应用开发教程，包含核心API介绍、传输模式选择、问题定位、性能分析等核心内容 | [hixl_development](./hixl_development) | ✅ 已发布 |
| HCCL 集合通信系列教程 | 面向昇腾 NPU 的 HCCL 集合通信开发教程，涵盖集合通信基础概念、常用通信算子、开发流程及实践 | [hccl_development](./hccl_development) | ✅ 已发布 |
| 大模型 SFT 训练系列教程 | 面向大模型监督微调（SFT）的训练实践教程，涵盖数据处理、模型训练及完整 SFT 训练流程 | [sft_training_pipeline](./sft_training_pipeline) | ✅ 已发布 |
| AutoFusion 自动融合开发系列教程 | 面向 CANN 图模式的 AutoFusion 自动融合开发教程，涵盖自动融合基础概念、融合原理及开发实践 | [autofusion_development](./autofusion_development) | ✅ 已发布 |
| GE 图引擎开发系列教程 | 面向 GE 图引擎的开发教程，涵盖基础概念、图构建与编译、模型执行与优化、扩展开发及问题定位 | [ge_development](./ge_development) | ✅ 已发布 |

## 教程使用说明
1. 已发布教程：点击链接可直接进入对应目录，包含教程文档、示例代码、实操步骤等完整内容；
2. 开发中教程：将持续补充后续算子开发教程；
3. 规划中教程：将持续更新开发进度，优先覆盖 PyPTO 核心语法、Tile 分片实战、算子性能调优等内容；
4. 环境要求：各教程的 CANN 版本要求以对应教程 README 为准，最低支持 8.5.0+，如离线体验需自行安装配套版本。

## 反馈与建议
若在教程学习过程中遇到问题，可通过以下方式反馈：
- 提交 Issue 至本仓库；
- 联系昇腾开发者社区技术支持。

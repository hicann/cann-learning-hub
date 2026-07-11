# 大模型推理系列课程

本课程面向具备基础编程能力的高校学生，从大语言模型的基本工作过程出发，逐步介绍昇腾 CANN 大模型推理的部署、性能分析和优化方法。课程由五个主题课件和一套 Qwen3-8B 单卡实践组成，理论内容与工程实践按学习顺序组织。

## 课程内容

| 序号 | 主题 | 主要内容 | 课件 |
|---|---|---|---|
| 01 | 大语言模型基础 | 文本表示、生成流程、Transformer、Attention、MLP、MoE、KV Cache 与多模态扩展 | [01_llm_fundamentals.pdf](./slides/01_llm_fundamentals.pdf) |
| 02 | CANN 开源推理仓库 | cann-recipes-infer 的定位、目录结构、模型覆盖、配置入口与社区参与方式 | [02_cann_inference_repository_overview.pdf](./slides/02_cann_inference_repository_overview.pdf) |
| 03 | 大模型推理优化基础 | Prefill 与 Decode、推理指标、Eager 与图模式，以及单卡常用优化方法 | [03_llm_inference_optimization_fundamentals.pdf](./slides/03_llm_inference_optimization_fundamentals.pdf) |
| 04 | 大模型量化基础 | 量化的基本原理、常见精度格式、校准方法与效果评估 | [04_llm_quantization_fundamentals.pdf](./slides/04_llm_quantization_fundamentals.pdf) |
| 05 | Profiling 与性能瓶颈定位 | Baseline 建立、日志解读、Profiler 采集，以及 op_statistic、kernel_details 和 trace 分析 | [05_profiling_and_performance_bottleneck_analysis.pdf](./slides/05_profiling_and_performance_bottleneck_analysis.pdf) |

## Qwen3-8B 实践

配套实践以 Qwen3-8B 单卡推理为主线，使用 cann-recipes-infer 的 YAML 配置和推理入口完成部署、Profiling、优化与量化验证。详细环境要求和运行说明见 [qwen3_8b/README.md](./qwen3_8b/README.md)。

| 序号 | 实践内容 | 在线体验 |
|---|---|---|
| 01 | 课程与实践流程介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/01_chapter_intro.ipynb) |
| 02 | Baseline 推理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/02_baseline_inference.ipynb) |
| 03 | Profiling 分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/03_profiling_analysis.ipynb) |
| 04 | Dense RMSNorm NPU 融合优化验证 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/04_npu_optimization.ipynb) |
| 05 | Qwen3-8B 量化 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/05_quantization_qwen3_8b.ipynb) |
| 06 | 自定义量化 MatMul 算子开发与模型接入 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/llm_inference/qwen3_8b&scanFilePath=tutorials/llm_inference/qwen3_8b/06_custom_matmul_operator_development_and_integration_with_qwen3_8b.ipynb) |

## 目录结构

```text
llm_inference/
├── README.md
├── slides/       # 按课程顺序编号的 PDF 课件
└── qwen3_8b/     # Notebook、实践代码与运行说明
```

建议先按编号阅读课件，再依次运行 Qwen3-8B Notebook。实践所需的 CANN、Python 依赖、模型权重和硬件条件以 Qwen3-8B 实践说明为准。

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

<table>
<tr><th>序号</th><th>实践内容</th><th>在线体验</th></tr>
<tr><td>01</td><td>课程与实践流程介绍</td><td rowspan="6">在CANNLab中运行（<a href="../../docs/CANNLab_env_experience_guide.md">CANNLab运行教程指导书</a>）</td></tr>
<tr><td>02</td><td>Baseline 推理</td></tr>
<tr><td>03</td><td>Profiling 分析</td></tr>
<tr><td>04</td><td>Dense RMSNorm NPU 融合优化验证</td></tr>
<tr><td>05</td><td>Qwen3-8B 量化</td></tr>
<tr><td>06</td><td>自定义量化 MatMul 算子开发与模型接入</td></tr>
</table>

## 目录结构

```text
llm_inference/
├── README.md
├── slides/       # 按课程顺序编号的 PDF 课件
└── qwen3_8b/     # Notebook、实践代码与运行说明
```

建议先按编号阅读课件，再依次运行 Qwen3-8B Notebook。实践所需的 CANN、Python 依赖、模型权重和硬件条件以 Qwen3-8B 实践说明为准。

# cann-learning-hub

> cann-learning-hub 是CANN （Compute Architecture for Neural Networks）生态的官方开源学习中心仓库，聚焦 NPU 加速计算开发能力培养，汇聚从入门到进阶的全栈学习资源。仓库涵盖 CANN 全栈加速计算的系列示例与最佳实践教程，支持以 Notebook 方式在线 / 离线交互式运行，帮助开发者零门槛上手。我们致力于打造动态、全面的 CANN 知识平台，系统化整理入门指南、高级优化教程、精选算子与模型示例及经过验证的最佳实践方案。通过持续迭代更新，助力开发者快速掌握 CANN 开发技能，高效释放昇腾 NPU 算力，加速 AI 应用的开发与创新。欢迎广大开发者贡献案例、教程、文档及各类学习资源，共建开放共享的 CANN 开发者生态。

本仓已集成代码仓库智能体，点击 [![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/cann-learning-hub) 徽章，开启在线智能代码学习与知识问答体验！

---

## 🧭 快速导航

| 你是 | 直达链接 |
| :--- | :--- |
| 🎓 **入门新手** | → [新手入门学习路径](#beginner) |
| 💻 **资深开发者** | → [全栈课程体系地图](#fullstack) |
| 🏫 **高校教师** | → [高校教学方案专区](#university) |
| 🏆 **竞赛备赛** | → [赛事备考专区](#competition) |
| 🏋️ **想做模型微调** | → [SwanLab 共建案例](#swanlab) |
| 📝 **想看真实客户实践案例** | → [技术博客](#blogs) |

---

## 📋 运行方式说明

本仓教程支持两种运行方式：

| 运行方式 | 说明 |
| :--- | :--- |
| **在线体验** | 点击即可直接打开在线运行环境运行，无需额外配置 |
| **[open in CANNLab](https://gitcode.com/org/cann/cannlab)** | 需先进入 CANNLab 创建环境，clone 本仓后从目录树中找到对应教程打开运行，注意选择内核（如 Python 3.11.4） |

---

## 📋 前置基础（已具备可跳过）：
> - **编程语言**：Python/C/C++
> - **计算机系统**：计算机组成原理/操作系统基础（进程/线程、内存管理、环境配置等）
> - **AI 与数学**：深度学习基础（神经网络）+ 线性代数与微积分（矩阵运算、梯度等）

<a id="beginner"></a>
## 🎓 新手入门学习路径

> 所有方向共享「阶段一：认识平台」公共基础，完成后按兴趣选择一条主线深入。没有唯一正确路径——选你最关心的场景出发即可。

**🚀 一键起步**：从 [人工智能基础](./quick_start/cann_basics/01_ai_basics.ipynb) 开始，可点击在线体验直接运行，边学边练。

### 阶段一：认识平台（约 2h）【公共基础】

**目标**：理解 CANN 是什么，验证 NPU 环境可用

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 1 | [人工智能基础](./quick_start/cann_basics/01_ai_basics.ipynb) | AI 发展历程、算子概念（名称/类型/Tensor/shape/format/Axis） | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/01_ai_basics.ipynb) |
| 2 | [什么是 NPU](./quick_start/cann_basics/02_what_is_npu.ipynb) | 昇腾 NPU 硬件架构：DaVinci 核心、AI Core / Vector / Cube 计算单元 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/02_what_is_npu.ipynb) |
| 3 | [什么是 CANN](./quick_start/cann_basics/03_what_is_cann.ipynb) | CANN 异构计算架构与软件栈：分层架构、Ascend C、torch_npu 适配 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/03_what_is_cann.ipynb) |
| 4 | [Hello World：NPU 加法](./quick_start/cann_basics/04_hello_world_npu.ipynb) | 基于 torch_npu 跑通第一个运算，验证环境可用 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/04_hello_world_npu.ipynb) |
| 5 | [MNIST 手写数字识别（可选）](./contrib/tutorials/swanlab_examples/mnist/mnist.ipynb) | 你的第一个模型训练：数据加载→模型构建→训练→评估，配套 SwanLab 可视化，可直观感受昇腾NPU模型训练 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

## 🧭 选择你的主线

| 主线 | 适合人群 | 总时长 | 核心产出 | 难度 |
| :--- | :--- | :---: | :--- | :---: |
| [🧠 大模型推理](#mainline-inference) | 想快速跑通大模型、做推理优化 | ~10h | 端到端推理调优能力 | ★★☆ |
| [🏋️ 大模型训练](#mainline-training) | 想做模型训练与微调 | ~16h | SFT/RL 训练与微调能力 | ★★★ |
| [⚙️ 算子开发](#mainline-operator) | 想掌握底层编程、参与竞赛 | ~20h | 自定义算子开发与接入 | ★★★ |
| [📊 推荐系统](#mainline-recsys) | 想做推荐/召回业务落地 | ~8h | CTR 排序 + 召回全链路 | ★★☆ |

---

<a id="mainline-inference"></a>
### 主线一：🧠 大模型推理（约 10h ~ 32h）

> 以 Qwen3-8B 端到端推理案例为主线，自然贯通"跑通推理 → 发现瓶颈 → 性能调优 → 量化与算子开发"。

<details>
<summary><b>阶段二：跑通推理（约 3-4h）</b></summary>

**目标**：在 NPU 上跑通 Qwen3-8B 推理，获得"真实大模型跑起来了"的成就感，配套课件是理论基础，建议先学习课件，再进行实践。

> 📖 **配套课件**：[大语言模型基础](./tutorials/llm_inference/slides/01_llm_fundamentals.pdf) ｜ [CANN 推理仓库](./tutorials/llm_inference/slides/02_cann_inference_repository_overview.pdf)


| 序号 | 课程（实践） | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 5 | [章节介绍](./tutorials/llm_inference/qwen3_8b/01_chapter_intro.ipynb) | 全流程概览，知道接下来要做什么 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 6 | [Qwen3-8B 推理](./tutorials/llm_inference/qwen3_8b/02_baseline_inference.ipynb) | 跑通 Qwen3-8B BF16 推理，感知大模型推理 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

</details>

<details>
<summary><b>阶段三：调优实战（约 4h）</b></summary>

**目标**：走完"发现瓶颈 → 性能优化"工程闭环，理解为什么需要算子级优化，配套课件是理论基础，建议先学习课件，再进行实践。

> 📖 **配套课件**：[推理优化基础](./tutorials/llm_inference/slides/03_llm_inference_optimization_fundamentals.pdf) ｜ [Profiling 与瓶颈定位](./tutorials/llm_inference/slides/05_profiling_and_performance_bottleneck_analysis.pdf)

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 7 | [Profiling 分析](./tutorials/llm_inference/qwen3_8b/03_profiling_analysis.ipynb) | 对 Baseline 做 Profiling，定位 RMSNorm 等小算子链路瓶颈 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 8 | [Dense RMSNorm NPU 融合优化](./tutorials/llm_inference/qwen3_8b/04_npu_optimization.ipynb) | 切换融合开关，A/B 对比验证算子融合的性能收益 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

> 💡 至此完成推理主线核心（约 10h），已具备端到端 BF16 推理调优能力。以下为可选拓展。

</details>

<details>
<summary><b>阶段四：量化与算子开发（可选，约 24h）</b></summary>

**目标**：从量化推理中发现算子瓶颈，系统学习算子开发后自研算子并接入真实模型，配套课件是理论基础，建议先学习课件，再进行实践。

> 📖 **配套课件**：[量化基础](./tutorials/llm_inference/slides/04_llm_quantization_fundamentals.pdf)

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 9 | [量化 Qwen3-8B 模型](./tutorials/llm_inference/qwen3_8b/05_quantization_qwen3_8b.ipynb) | AMCT 工具导出 W8A8 权重 → 量化推理 → Profiling 定位 `QuantBatchMatmulV3` 瓶颈 → 推导 `QmmCustom` 原型 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 10 | [Ascend C 算子开发系列（Kernel 直调版）](./tutorials/ascendc_operator_development_light) | 系统学习 Tiling 设计、Kernel 实现、编译与调试调优全流程 | [在线体验](./tutorials/ascendc_operator_development_light) |
| 11 | [自定义量化算子开发并接入 Qwen3-8B](./tutorials/llm_inference/qwen3_8b/06_custom_matmul_operator_development_and_integration_with_qwen3_8b.ipynb) | 用步骤 ⑩ 的技能实现步骤 ⑨ 的原型 → 编译 → 替换瓶颈算子 → 验证收益 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |


> 💡 阶段四叙事：**⑨ 量化给动机**（为什么需要自定义算子）→ **⑩ Ascend C 给能力**（怎么开发算子）→ **⑪ 接入给闭环**（学以致用，替换验证）

</details>

---

<a id="mainline-training"></a>
### 主线二：🏋️ 大模型训练

> 从微调实战到 SFT/RL 训练全流程，系统掌握大模型训练与性能优化能力。

<details>
<summary><b>阶段二：微调实战案例（SwanLab 共建）🚧</b></summary>

**目标**：通过端到端微调案例，配套 SwanLab 可视化能力，快速感知训练过程，可选择一个或多个感兴趣的案例进行学习。

| 序号 | 课程 | 课程内容 | 状态 |
| :---: | :--- | :--- | :---: |
| 5 | [医学模型微调](./contrib/tutorials/swanlab_examples/qwen3_medical_sft) | Qwen3 医学领域 SFT + SwanLab 可视化 | ✅ 已上线 |
| 6 | [ms-swift 框架微调](https://docs.swanlab.cn/course/llm_train_course/03-sft/8.other_frameworks/ms-swift.html) | ms-swift 框架微调 + SwanLab 可视化 | 🚧 建设中 |
| 7 | [Qwen3-smVL 多模态微调](https://docs.swanlab.cn/course/llm_train_course/06-multillm/2.qwen3_smolvlm_muxi/) | 多模态拼接微调 + SwanLab 可视化 | 🚧 建设中 |
| 8 | [CosyVoice 语音微调](https://docs.swanlab.cn/course/llm_train_course/07-audio/1.cosyvoice-sft/) | 语音模型微调 + SwanLab 可视化 | 🚧 建设中 |

> 💡 案例正在从 SwanLab 平台迁移至 `contrib/tutorials/swanlab_examples/`，完成后将在 CANNLab 环境提供在线体验。详见 [SwanLab 共建案例](#swanlab)。

</details>

<details>
<summary><b>阶段三：SFT/RL 初阶课程</b></summary>

**目标**：掌握 SFT 与 RL 训练基线，具备端到端训练跑通与问题定位能力，配套课件是理论基础，请先学习配套课件再进行实践。

> 📖 **配套课件**：[SFT 训练流程](./tutorials/sft_training_pipeline)，[RL 训练流程](./tutorials/rl_training_pipeline)

| 序号 | 课程（实践） | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 9 | [SFT 训练系列（初阶）](./tutorials/sft_training_pipeline) | 环境搭建 Torchtitan / Qwen3-1.7B 基线跑通 + Profiling / 应用融合算子 / 性能对比 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 10 | [RL 训练系列（初阶）](./tutorials/rl_training_pipeline)  | Qwen3-1.7B 基线跑通 / 训推分离架构 / Wordle 评分 / vLLM-Ascend 适配 / 问题定位 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

</details>

<details>
<summary><b>阶段四：SFT/RL 中阶课程 🚧</b></summary>

**目标**：掌握长序列训练与 RL 后端开发能力

| 序号 | 课程 | 课程内容 | 状态 |
| :---: | :--- | :--- | :---: |
| 11 | SFT 训练系列（中阶） | varlen 注意力与上下文并行（CP）融合适配，消除 padding 冗余，提升长序列训练效率 | 🚧 建设中 |
| 12 | RL 训练系列（中阶）| 基于 torchtitan-npu 的 RL 后端开发，打通 vLLM rollout→Wordle reward→Actor 更新完整链路，完成多后端一致性对比 | 🚧 建设中 |

</details>

<details>
<summary><b>阶段五：SFT/RL 高阶课程 🚧</b></summary>

**目标**：掌握训练性能优化与显存调优能力

| 序号 | 课程 | 课程内容 | 状态 |
| :---: | :--- | :--- | :---: |
| 13 | SFT 训练系列（高阶） | 计算图静态化与 AutoFuse 融合优化，selective AC 显存调优，提升 CP 并行端到端训练吞吐 | 🚧 建设中 |
| 14 | RL 训练系列（高阶） | 性能基线搭建与瓶颈定位，FSDP2/TP/CP/PP 并行调优，RL 训练流水优化，A/B 验证训练稳定性 | 🚧 建设中 |

</details>

---

<a id="mainline-operator"></a>
### 主线三：⚙️ 算子开发（约 20h）

> 从"10 分钟体验算子"到系统掌握 Ascend C 编程范式，具备自定义算子开发与工程化能力。

<details>
<summary><b>阶段二：体验算子（约 2h）</b></summary>

**目标**：快速感知算子开发与调用全流程

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 5 | [10 分钟体验自定义算子](./quick_start/first_custom_operator/first_custom_operator.ipynb) | 第一个自定义算子开发，感知 Ascend C 编写与编译全流程 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/first_custom_operator&scanFilePath=quick_start/first_custom_operator/first_custom_operator.ipynb) |
| 6 | [10 分钟体验算子 API 调用](./quick_start/first_operator_api_call/first_operator_api_call.ipynb) | 第一个算子调用，感知算子调用与价值 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/first_operator_api_call&scanFilePath=quick_start/first_operator_api_call/first_operator_api_call.ipynb) |

</details>

<details>
<summary><b>阶段三：系统学习（约 12h）</b></summary>

**目标**：掌握 Tiling 设计、Kernel 实现、编译与调试调优全流程

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 7 | [Ascend C 算子开发系列（Kernel 直调版）](./tutorials/ascendc_operator_development_light) | 算子基础概念、编程范式、Vector/Cube/融合算子开发与调试调优 | [在线体验](./tutorials/ascendc_operator_development_light) |

</details>

<details>
<summary><b>阶段四：算子开发实战</b></summary>

**目标**：掌握标准算子工程开发流程，具备社区贡献能力

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 8 | [Ascend C 算子开发系列（算子工程版）](./tutorials/ascendc_operator_development) | 工程化开发流程、开源社区贡献规范与玩法 | [在线体验](./tutorials/ascendc_operator_development) |
| 9 | Vector 算子开发| Vector 算子开发实战 | 🚧 建设中 |
| 10 | Conv 算子开发实战| 卷积算子开发核心概念与实践 | 🚧 建设中 |
| 11 | [MC2 融合算子实战](./tutorials/MC2_fused_operator_development) | Matmul/Conv/通算融合等典型算子实战 | [在线体验](./tutorials/MC2_fused_operator_development) |

> 💡 **进阶练习**：[CANNJudge 算子题库](https://cannjudge.cn) 在线刷题 → [CANN 大赛专区](https://competition.gitcode.com/competition?type=cann) 参赛验证

</details>

---

<a id="mainline-recsys"></a>
### 主线四：📊 推荐系统（约 8h）

> 基于 Torch-RecHub 跑通推荐系统全链路：排序 → 召回 → 多任务 → 工程化。

<details>
<summary><b>阶段二：跑通 CTR（约 2h）</b></summary>

**目标**：跑通 CTR 训练链路，获得端到端业务体感

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 5 | [QuickStart：CTR 预测（DeepFM）](./contrib/tutorials/torch-rechub/00_QuickStart_CTR_DeepFM.ipynb) | DataFrame → Feature → DeepFM → CTRTrainer → AUC | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/00_QuickStart_CTR_DeepFM.ipynb) |

</details>

<details>
<summary><b>阶段三：进阶建模（约 4h）</b></summary>

**目标**：掌握排序、召回、多任务三大推荐建模范式

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 6 | [序列兴趣建模：DIN](./contrib/tutorials/torch-rechub/01_Ranking_DIN.ipynb) | 历史行为序列、SequenceFeature 与 DIN attention | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/01_Ranking_DIN.ipynb) |
| 7 | [匹配/召回：DSSM + Annoy](./contrib/tutorials/torch-rechub/02_Matching_DSSM.ipynb) | 双塔召回与向量 Top-K 检索 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/02_Matching_DSSM.ipynb) |
| 8 | [多任务学习：MMOE](./contrib/tutorials/torch-rechub/03_MultiTask_MMOE.ipynb) | 多目标建模、expert、gate 与 tower | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/03_MultiTask_MMOE.ipynb) |

</details>

<details>
<summary><b>阶段四：工程化（约 2h）</b></summary>

**目标**：掌握实验跟踪与模型导出部署

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 9 | [实验跟踪：model_logger](./contrib/tutorials/torch-rechub/04_Experiment_Tracking_Light.ipynb) | WandB / SwanLab / TensorBoardX 轻量实验跟踪 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/04_Experiment_Tracking_Light.ipynb) |
| 10 | [模型导出与推理验证：ONNX](./contrib/tutorials/torch-rechub/05_Model_Export_and_Serving.ipynb) | ONNX 导出、ONNXRuntime 推理验证和量化入口 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/05_Model_Export_and_Serving.ipynb) |

</details>

---

> 💡 **配套资源**：[CANNLab 实验环境指南](./docs/CANNLab_env_experience_guide.md) ｜ [社区讨论](https://gitcode.com/cann/cann-learning-hub/discussions) ｜ [常见问题](https://gitcode.com/cann/cann-learning-hub/issues)

---

<a id="fullstack"></a>
## 💻 全栈课程体系地图

> 按 **技术领域 × 难度等级** 双维度搭建完整课程体系，高手可直接跳过入门内容，精准定位高阶课程。

<table width="100%">
<tr><th width="16%">技术领域</th><th width="28%">初级课程</th><th width="28%">中级课程</th><th width="28%">高级课程</th></tr>
<tr>
<td><b>⚙️ 算子开发</b></td>
<td><a href="./tutorials/ascendc_operator_development_light">Kernel 直调版</a> · d2l PyPTO 实战 🚧 · PyASC 入门 🚧<br/><sub>算子基础概念、编程范式、PyPTO Tensor 编程、PyASC 函数开发</sub></td>
<td><a href="./tutorials/ascendc_operator_development">算子工程版</a> · Conv 算子实战 🚧 · Vector 算子实战 🚧<br/><sub>Vector/Cube/融合算子开发、工程化流程、卷积与矢量算子实战</sub></td>
<td><a href="./tutorials/MC2_fused_operator_development">MC2 融合算子实战</a> · AscendC V2 高阶 🚧 · MoE/FA 算子实战 🚧<br/><sub>Matmul/Conv/通算融合、aclnn/aclGraph 工程化</sub></td>
</tr>
<tr>
<td><b>🧠 大模型推理</b></td>
<td><a href="./tutorials/llm_inference">推理系列（初级）</a> · <a href="./quick_start/first_llm_inference/01_qwen3_npu_inference_baseline.ipynb">极简参考实现</a><br/><sub>推理部署、Qwen3-8B 端到端实战</sub></td>
<td><a href="./tutorials/llm_inference">推理系列（中级）</a><br/><sub>推理优化/量化基础/算子融合/图模式/多卡并行/KV Cache/Profiling</sub></td>
<td><a href="./reference_practice/model_inference_optimization/sana_video">Sana-Video 最佳实践</a> · 推理系列（高级）🚧<br/><sub>多流/控核/MTP/PD分离/KV Cache Offload/4bit量化</sub></td>
</tr>
<tr>
<td><b>🏋️ 大模型训练</b></td>
<td><a href="./tutorials/sft_training_pipeline">SFT 训练（初阶）</a> · <a href="./tutorials/rl_training_pipeline">RL 训练（初阶）</a> · <a href="#swanlab">SwanLab 微调案例</a><br/><sub>SFT/RL 基线跑通、医学/多模态/语音微调（可视化）</sub></td>
<td>SFT 训练（中阶） 🚧 · RL 训练（中阶） 🚧<br/><sub>varlen+CP 融合、RL 后端开发、SwanLab 共建最佳实践</sub></td>
<td>SFT/RL 训练（高阶）🚧<br/><sub>计算图优化、显存调优、端到端吞吐</sub></td>
</tr>
<tr>
<td><b>📊 推荐系统</b></td>
<td><a href="./contrib/tutorials/torch-rechub/00_QuickStart_CTR_DeepFM.ipynb">CTR 预测（DeepFM）</a> · <a href="./contrib/tutorials/torch-rechub/01_Ranking_DIN.ipynb">DIN</a> · <a href="./contrib/tutorials/torch-rechub/02_Matching_DSSM.ipynb">DSSM</a><br/><sub>排序、召回、多任务、实验跟踪、模型导出</sub></td>
<td>🚧 建设中<br/><sub>时延优化/动态图性能优化/多实例控核</sub></td>
<td>🚧 建设中<br/><sub>多卡训练调优/自定义算子开发/自定义Pass开发</sub></td>
</tr>
<tr>
<td><b>🔗 图框架</b></td>
<td><a href="./tutorials/ge_development">GE 图引擎</a> · <a href="./tutorials/TorchAir_development">TorchAir</a> · <a href="./tutorials/autofusion_development">AutoFusion</a><br/><sub>图编译执行、图模式优化、自动融合</sub></td>
<td>🚧 建设中<br/><sub>图构建/编译配置/自定义算子入图/自定义融合pass/静态&动态shape</sub></td>
<td>🚧 建设中<br/><sub>自动融合原理/融合策略求解/SuperKernel等</sub></td>
</tr>
<tr>
<td><b>📡 通信</b></td>
<td><a href="./tutorials/hixl_development">HiXL 单边通信</a> · <a href="./tutorials/hccl_development">集合通信</a><br/><sub>单边通信基础、集合通信基础</sub></td>
<td>🚧 建设中<br/><sub>HiXL传输模式、集合通信开发</sub></td>
<td>🚧 建设中<br/><sub>PD分离与KV Cache传输、多卡推理/训练通信算子开发</sub></td>
</tr>
<tr>
<td><b>📱 应用开发</b></td>
<td>应用开发 🚧<br/><sub>ATC离线模型编译&推理、GE/ATC/模型转换全流程</sub></td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><b>🤖 CANNBot</b></td>
<td><a href="https://gitcode.com/cann/cannbot-skills">CANNBot</a> · <a href="./contrib/tutorials/data_structure_for_hpc">HPC 数据结构</a><br/><sub>开发辅助、高性能计算基础</sub></td>
<td><a href="./tutorials/CANNBot/README.md">CANNBot 系列课程</a><br/><sub>用 CANNBot 生成与优化 Ascend 算子</sub></td>
<td>🚧 建设中</td>
</tr>
</table>

---

<a id="university"></a>
## 🏫 高校教学方案专区（建设中）

> 直接输出可落地的成套教学解决方案，匹配高校不同开课场景，老师可直接参考、二次改造。

| 方案 | 适用场景 | 学时 | 内容覆盖 | 配套资源 |
| :--- | :--- | :---: | :--- | :--- |
| **本科常态化必修课** | 大三计算机/AI 专业必修课，对标浙大《人工智能芯片与系统》类课程 | 48 | 异构计算基础、Ascend C 算子开发、模型迁移与部署、综合实训 | 完整课程大纲、课时分配、12 个配套实验、课件模板、题库、教学 PPT |
| **短期实训营** | 高校暑期实训、企业新人培训、1-2 周集中实训 | 20 | 环境快速上手、基础算子开发、小项目实操 | 实训大纲、实操手册、结营考核题目 |
| **研究生进阶专题** | 研究生高阶课程、科研方向入门 | 32 | 分布式训练、算子深度优化、大模型训练适配 | 专题大纲、科研案例、参考论文、前沿实验 |

> 🏫 **已合作高校**：北京邮电大学、哈尔滨工业大学、上海交通大学等高校已基于本仓内容开设相关课程。

---

<a id="competition"></a>
## 🏆 赛事备考专区

> 按赛事分类聚合所有相关内容，直接匹配赛事考点，选手无需从通用课程中自行筛选。

| 赛事 | 赛事说明 | 备赛资源 | 赛事权益 |
| :--- | :--- | :--- | :--- |
| [**CANNJudge 算子题库**](https://cannjudge.cn) | 开放题库，Ascend C 算子编程在线刷题，实时评测；含历届算子赛真题 | [算子开发练习](https://cannjudge.cn) → 在线刷题<br/>[Ascend C 算子开发系列](./tutorials/ascendc_operator_development_light) → 前置课程<br/>[MC2 融合算子实战](./tutorials/MC2_fused_operator_development) → 高阶练习 | 实时评测排名等 |
| [**CANN 大赛专区**](https://competition.gitcode.com/competition?type=cann) | 算子天梯赛 / 校园赛等各种大赛 | [大赛报名入口](https://competition.gitcode.com/competition?type=cann) → 报名与赛程<br/>[skills 目录](./skills) → 竞赛提交技能与算子工程生成 | 获奖对应、人才库推荐等 |

<details>
<summary>查看备赛建议路径</summary>

1. **前置学习**：完成 [新手入门学习路径](#beginner) 阶段一 + 阶段二
2. **专项练习**：在 [CANNJudge](https://cannjudge.cn) 按考点分类刷题
3. **高阶提升**：学习 [Ascend C 算子开发系列（算子工程版）](./tutorials/ascendc_operator_development) 
4. **赛前集训**：关注大赛专区公告，参加赛前培训直播
5. **参赛提交**：使用 [skills/cannjudge-submit](./skills) 快速提交

</details>

---

<a id="certification"></a>
## 📜 等级认证学习路径（建设中）

> 对应 AI Infra 工程师初/中/高三级认证体系。通识课程 + 场景方向（推理 / 训练 / 推荐 / 应用开发 / 算子 5 选 1），明确"考什么、学什么"。

---

<a id="swanlab"></a>
## 🤝 SwanLab 共建案例

> 与 [SwanLab](https://swanlab.cn) 合作共建的大模型训练实战案例，正在从 SwanLab 平台迁移至本仓 `contrib/tutorials/swanlab_examples/` 目录下。

**适合人群**：想在 NPU 上做模型微调的开发者 · 关注训练过程可视化的用户 · 医学/语音/多模态领域开发者

**特色**：每个案例均配套 SwanLab 训练可视化（loss/metrics 曲线、实验对比），迁移完成后可在 CANNLab 环境直接运行。

| 类别 | 案例 | 简介 | 原始文档 | 状态 |
| :--- | :--- | :--- | :--- | :---: |
| 深度学习入门 | MNIST 手写数字识别 | NPU 上 CNN 手写数字识别入门 | [查看](./contrib/tutorials/swanlab_examples/mnist) | ✅ 已上线 |
| 监督微调 | 数学解题模型微调 | Qwen2.5-0.5B LoRA 数学解题微调实战 | [查看](./contrib/tutorials/swanlab_examples/math_solver_qwen2.5_lora) | ✅ 已上线 |
| 监督微调 | 医学模型微调 | Qwen3 医学领域 SFT 实战 | [查看](./contrib/tutorials/swanlab_examples/qwen3_medical_sft) · [原文](https://docs.swanlab.cn/course/llm_train_course/03-sft/4.qwen3-medical-finetune/) | ✅ 已上线 |
| 监督微调 | 其他框架微调——ms-swift | 使用 ms-swift 框架进行微调 | [文档](https://docs.swanlab.cn/course/llm_train_course/03-sft/8.other_frameworks/ms-swift.html) | 🚧 建设中 |
| 多模态 | Qwen3-smVL 模型拼接微调 | 多模态模型拼接微调实战 | [文档](https://docs.swanlab.cn/course/llm_train_course/06-multillm/2.qwen3_smolvlm_muxi/) | 🚧 建设中 |
| 音频 | CosyVoice 微调派蒙语音 | 语音模型微调实战 | [文档](https://docs.swanlab.cn/course/llm_train_course/07-audio/1.cosyvoice-sft/) | 🚧 建设中 |

> 💡 案例持续迁移中，完成后将在 CANNLab 环境提供在线体验。

---

## 🔗 配套生态与社区资源

<table style="table-layout:fixed;width:100%">
<tr><th width="8%"></th><th width="25%">平台</th><th width="67%">说明</th></tr>
<tr><td>📖 <b>学</b></td><td><b>cann-learning-hub</b>（本仓）</td><td>系列教程 + 快速上手 + 参考实践 + 技术博客</td></tr>
<tr><td>🏋️ <b>练</b></td><td><b><a href="https://cannjudge.cn">CANNJudge</a></b></td><td>开放题库，Ascend C 算子编程在线刷题，实时评测</td></tr>
<tr><td>🔬 <b>练</b></td><td><b>CANNLab</b></td><td>任意 CANN 代码仓右上角一键启动，获得 NPU 环境用于练习 &amp; 调测（初始 100 小时，积分可兑换时长）</td></tr>
<tr><td>🏆 <b>赛</b></td><td><b><a href="https://competition.gitcode.com/competition?type=cann">CANN 大赛专区</a></b></td><td>官方 / 社区大赛报名入口，与全国开发者同台竞技</td></tr>
</table>

> 💡 **推荐路径**：选方向 → 看教程 → 去 CANNJudge 刷题 / CANNLab 实验 → 参赛验证 → 贡献你的实践（PR 到 contrib/）

<details>
<summary>查看目录结构</summary>

```
├── quick_start                        # 快速入门
│   ├── cann_basics                    # CANN 基础知识
│   ├── first_custom_operator          # 第一个自定义算子
│   ├── first_operator_api_call        # 第一个算子 API 调用
│   └── first_llm_inference            # 第一个大模型推理和优化
├── tutorials                              # 开发教程
│   ├── ascendc_operator_development       # Ascend C 算子开发
│   ├── ascendc_operator_development_light # Ascend C 算子开发（Kernel 直调版）
│   ├── conv_operator_development          # Conv 算子开发实战
│   ├── MC2_fused_operator_development     # MC2 融合算子开发实战
│   ├── llm_inference                      # 大模型推理系列课程
│   ├── sft_training_pipeline              # 大模型 SFT 训练系列课程
│   ├── rl_training_pipeline               # 大模型 RL 训练系列课程
│   ├── ge_development                     # GE 图引擎开发系列教程
│   ├── TorchAir_development               # TorchAir图模式优化系列教程
│   ├── autofusion_development             # AutoFusion 自动融合开发系列教程
│   ├── hccl_development                   # HCCL 集合通信系列课程
│   ├── hixl_development                   # HiXL 单边通信应用开发
│   ├── CANNBot                            # CANNBot 算子生成相关课程
│   └── ...                                # 待扩展（PyPTO / TileLang 等）
├── reference_practice                 # 参考实践
│   ├── model_inference_optimization   # 模型推理优化
│   │   └── sana_video                # Sana-Video 推理优化
│   └── pytorch_online_inference_operator_optimize  # PyTorch 在线推理算子优化
├── blogs                              # 技术博客
│   ├── operator                       # 算子
│   ├── inference                      # 推理
│   └── training                       # 训练
├── contrib                            # 社区贡献
│   └── tutorials                      # 外部贡献教程
│       ├── torch-rechub               # Torch-RecHub 推荐系统实战教程
│       ├── swanlab_examples           # SwanLab 共建微调案例（MNIST/医学/数学等）
│       ├── swan_llm_course            # SwanLab 共建：LLM 微调实战
│       └── data_structure_for_hpc     # HPC 数据结构
├── skills                             # CANNBot 技能
│   ├── ascendc-ops-project           # 自定义算子工程生成
│   └── cannjudge-submit              # CANNJudge 竞赛提交
├── docs                               # 文档与指南
│   ├── CANNLab_env_experience_guide.md    # CANNLab 环境体验指南
│   ├── CANNLab_course_development_guide.md  # 基于 CANNLab 环境开发与提交课程指南
│   └── course_submission_criteria.md  # 新课程上库与上线验收标准
└── README.md
```

</details>

<a id="blogs"></a>
<details>
<summary>查看技术博客（大部分为真实客户实践案例）</summary>

> CANN 在实际业务场景中的最新技术实践与成果。

<table style="table-layout:fixed;width:100%">
<tr><th width="35%">博客</th><th width="55%">简介</th><th width="10%">时间</th></tr>
<tr><th colspan="3" align="left">算子</th></tr>
<tr><td><a href="./blogs/operator/hccl_custom_operator_aicpu_p2p">AICPU 点对点通信算子开发</a></td><td>基于 AICPU+TS 实现 HCCL 自定义 Send/Recv 算子</td><td>2026.2</td></tr>
<tr><td><a href="./blogs/operator/aicpu_tiling_sink">AICPU Tiling 下沉编程</a></td><td>Tiling 计算下沉到 AICPU，减少 Host 与 Device 交互</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/operator/ascendc_rtc_compilation">Ascend C RTC 即时编译</a></td><td>运行时按 shape 即时编译，兼顾性能与迭代灵活性</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/operator/deepxtrace_moe_slow_card_detection">DeepXTrace 快慢卡在线检测</a></td><td>MOE 推理集群轻量级快慢卡诊断，分钟级定位</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/operator/hccl_reducescatter_high_precision_redevelopment">HCCL ReduceScatter 精度优化</a></td><td>开源 ReduceScatter 精度增强改造</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/operator/transformer_experimental_mix_operator">MIX 算子开发贡献</a></td><td>矩阵化重构 RoPE，落地首个开源 MIX 算子</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/operator/cross_entropy_zloss_fusion">CrossEntropyLoss 与 Zloss 融合</a></td><td>损失函数融合，MoE 场景端到端 5.2% 效率提升</td><td>2025.11</td></tr>
<tr><td><a href="./blogs/operator/kernel_direct_call_programming">算子 Kernel 直调编程</a></td><td>异构混合编程，简化编译部署，降低开发门槛</td><td>2025.11</td></tr>
<tr><td><a href="./blogs/operator/tilingkey_template_programming">TilingKey 模板化编程</a></td><td>统一多场景算子管理，减少 icache miss</td><td>2025.11</td></tr>
<tr><td><a href="./blogs/operator/ascend_c_mmad_selection_guide">Ascend C 矩阵乘接口选型指南</a></td><td>矩阵乘 API 接口对比与选型建议</td><td>2025.10</td></tr>
<tr><td><a href="./blogs/operator/ms_sanitizer">msSanitizer 异常检测工具</a></td><td>单算子开发异常检测，定位内存访问、数据竞争与同步问题</td><td>2025.10</td></tr>
<tr><td><a href="./blogs/operator/nddma_introduction">NDDMA 多维数据搬运</a></td><td>多维 DMA 搬运与 Padding、Transpose、Broadcast、Slice 变换</td><td>2025.10</td></tr>
<tr><td><a href="./blogs/operator/regbase_vec_add">Regbase 编程范式</a></td><td>从向量加法理解寄存器级编程与底层性能优化</td><td>2025.10</td></tr>
<tr><th colspan="3" align="left">推理</th></tr>
<tr><td><a href="./blogs/inference/overlap_scheduling_throughput_optimization">Overlap Scheduling 吞吐优化</a></td><td>CPU 与 NPU 执行重叠，TPS 提升约 70%</td><td>2026.3</td></tr>
<tr><td><a href="./blogs/inference/npugraph_ex_third_party_framework_integration">npugraph_ex 第三方框架集成</a></td><td>图编译与编译缓存能力接入，降低冷启动耗时</td><td>2026.2</td></tr>
<tr><td><a href="./blogs/inference/deepseek_r1_superpod_inference_optimization">Deepseek-R1 SuperPoD 推理优化</a></td><td>全栈协同，TTFT&lt;2s、TPOT&lt;50ms，608 QPM</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/inference/hixl_mooncake_vllm_kv_cache_pooling">HIXL、Mooncake 与 vLLM KV Cache 池化</a></td><td>KV Cache 池化 + D2D/H2H 传输，降低 TTFT</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/inference/hixl_rl_tail_latency_optimization">HIXL RL 长尾时延优化</a></td><td>PD 分离与高效传输，缓解千卡集群长尾</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/inference/longcat_flash_superpod_inference_optimization">LongCat-Flash SuperPod 推理优化</a></td><td>多流并发 + 控核 + SuperKernel，TPOT 10ms</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/inference/npugraph_ex_aclgraph_graph_mode">npugraph_ex 图模式优化</a></td><td>aclGraph 图捕获与重放，减少 Host 下发</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/inference/torch_npu_ipc">torch_npu IPC 特性</a></td><td>跨进程共享设备内存，节省显存</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/inference/torchair_fx_pass_multi_stream">TorchAir 自定义 FX Pass</a></td><td>多流并行自动图变换，减少适配代码</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/inference/sglang_mooncake_hixl_pd_separation_d2d">SGLang、Mooncake 与 HIXL PD 分离</a></td><td>加速 PD 分离 D2D 特性落地</td><td>2025.11</td></tr>
<tr><td><a href="./blogs/inference/superkernel_inference_acceleration">SuperKernel 技术综述</a></td><td>整网编译为大算子，性能再提升 10%-20%</td><td>2025.11</td></tr>
<tr><td><a href="./blogs/inference/vllm_ascend_inference_optimization">vLLM-Ascend 推理优化</a></td><td>PagedAttention + 昇腾适配，提升吞吐</td><td>2025.11</td></tr>
<tr><th colspan="3" align="left">训练</th></tr>
<tr><td><a href="./blogs/training/areal_async_rl_training">AReaL 全异步 RL 训练</a></td><td>全异步 RL + Single Controller，解耦式 Agentic RL</td><td>2026.3</td></tr>
<tr><td><a href="./blogs/training/flashrecovery_training_fault_recovery">FlashRecovery 训练故障恢复</a></td><td>降低检查点 I/O 与回滚重算损失</td><td>2025.12</td></tr>
<tr><td><a href="./blogs/training/sam_speculative_decoding_rl_training">SAM 投机解码 RL 训练</a></td><td>无辅助模型 SAM 投机解码，超 35% 长尾加速</td><td>2025.12</td></tr>
</table>

</details>

---

## 📝 更新日志

<details>
<summary>查看版本配套信息</summary>

本项目源码会基于CANN软件的非beta版本进行全量验证，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明，已验证情况如下表所示。

| 已验证支持CANN版本 | 验证日期 |
|------------------------|-----------|
| 9.0.0 | 2026.06.30 |

</details>

## 🔥 Latest News

- [2026/08] 新增[CANNBot 系列课程](./tutorials/CANNBot/README.md)，围绕 CANNBot 算子开发工具，系统介绍如何使用 CANNBot 生成与优化 Ascend 算子，覆盖 Ascend C / PyPTO / TileLang-Ascend 等开发路径及算子测试全流程。
- [2026/08] 新增[TorchAir图模式优化系列教程](./tutorials/TorchAir_development)，涵盖 TorchAir 基础概念、PyTorch 模型图转换、昇腾 NPU 图模式执行与性能优化实践，帮助开发者掌握基于 TorchAir 的模型开发与优化流程。
- [2026/08] 新增[GE 图引擎开发系列教程](./tutorials/ge_development)，涵盖 GE 基础概念、图构建与编译、模型执行与优化、扩展开发及问题定位。
- [2026/08] 新增[AutoFusion 自动融合开发系列教程](./tutorials/autofusion_development)，涵盖 AutoFusion 基础概念、自动融合原理与开发实践。
- [2026/08] 新增[大模型训练系列课程](./tutorials/sft_training_pipeline)，涵盖大模型监督微调（SFT）核心概念、数据处理、模型训练与完整训练流程实践。
- [2026/08] 新增[HCCL 集合通信开发系列课程](./tutorials/hccl_development)，涵盖 HCCL 基础概念、集合通信原理、开发流程与通信实践，帮助开发者掌握昇腾多卡集合通信开发。
- [2026/07] 新增[Ascend C 算子开发系列（Kernel 直调版）教程](./tutorials/ascendc_operator_development_light)，覆盖算子开发基础概念、Ascend C 编程范式、Vector/Cube/融合算子开发与性能优化实战。
- [2026/07] 新增[大模型推理系列课程](./tutorials/llm_inference)，涵盖大语言模型基础、CANN 推理仓库、推理优化、量化与 Profiling，并提供 Qwen3-8B 单卡实践。
- [2026/07] 新增[Conv 算子开发实战教程](./tutorials/conv_operator_development)，覆盖卷积算子开发核心概念与实践。
- [2026/06] 在线体验适配 CANN 9.0.0。
- [2026/05] 新增[MC2 融合算子开发系列教程](./tutorials/MC2_fused_operator_development)，讲解 MC2 融合算子核心概念与开发方法。
- [2026/05] 新增[HIXL应用开发系列教程](./tutorials/hixl_development)，讲解昇腾单边通信库核心概念与开发方法。
- [2026/04] 新增[skills](./skills) 目录，包含 CANNJudge 竞赛 skill 及自定义算子工程算子生成 skill。
- [2026/03] 新增技术博客内容（blogs 目录）。
- [2026/03] cann-learning-hub项目首次上线。

---

## 💬 相关信息

<details>
<summary>查看相关信息与联系方式</summary>

- [贡献指南](CONTRIBUTION.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/doc)

**联系方式：**

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/cann-learning-hub/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/cann-learning-hub/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/cann-learning-hub/wiki)获取技术文章。

</details>

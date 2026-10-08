# cann-learning-hub

> cann-learning-hub 是 CANN（Compute Architecture for Neural Networks）生态的官方开源学习中心仓库，聚焦 NPU 加速计算开发能力培养，汇聚从入门到进阶的全栈学习资源。仓库涵盖 CANN 全栈加速计算的系列示例与最佳实践教程，支持以 Notebook 方式在线 / 离线交互式运行，帮助开发者零门槛上手。我们致力于打造动态、全面的 CANN 知识平台，系统化整理入门指南、高级优化教程、精选算子与模型示例及经过验证的最佳实践方案。通过持续迭代更新，助力开发者快速掌握 CANN 开发技能，高效释放昇腾 NPU 算力，加速 AI 应用的开发与创新。欢迎广大开发者贡献案例、教程、文档及各类学习资源，共建开放共享的 CANN 开发者生态。

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/cann-learning-hub) 点击上方徽章，开启在线智能代码学习与知识问答体验！

---
## 🚀 三种环境体验方式介绍

> 本仓大部分教程以 Notebook 形式提供，包含文本讲解与可执行代码，可交互式运行与修改。以下三种方式均可运行 Notebook 中的可执行代码，具体适用场景见各教程 README 说明。

**方式一：在线体验（推荐，零配置）**

打开 [人工智能基础](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/01_ai_basics.ipynb)（第一课），点击「在线体验」即可在浏览器中运行你的第一个 Notebook，无需任何环境配置。

> 支持在线体验的课程 README 中已为每个 Notebook 配置对应的在线体验链接，可直接点击运行。

**方式二：CANNLab 云环境**

进入 [CANNLab](https://gitcode.com/org/cann/cannlab) 创建 NPU 云环境（初始 100 小时，积分可兑换时长），clone 本仓后从目录树打开任意教程运行，注意选择课程支持的镜像版本和Python内核。

> 具体使用方式参考 [CANNLab 云开发平台体验指南](./docs/CANNLab_env_experience_guide.md)。

**方式三：本地部署**

```bash
git clone https://gitcode.com/cann/cann-learning-hub.git
cd cann-learning-hub
```

| 依赖 | 要求 |
|:---|:---|
| 硬件 | 昇腾 NPU（Atlas A2/A3 系列等；部分认知类课程无需 NPU） |
| CANN | 9.0.0 及以上，[下载安装](https://www.hiascend.com/cann/download) |
| Python | 3.11 |

> 各教程的具体依赖见其目录下的 `requirements.txt`；环境配置详见 [CANNLab 环境体验指南](./docs/CANNLab_env_experience_guide.md)。

---

## 🎯 角色导航

> 找到你的角色，直达对应内容 👇

| 角色 | 直达 |
| :--- | :--- |
| 🌱 **新手开发者** | [从这里开始](#cann-basics) — CANN 基础第一课，零门槛入门 |
| 🧠 **大模型开发者** | [大模型训练](#mainline-training) · [大模型推理](#mainline-inference) · [SwanLab 微调案例](#swanlab) |
| ⚙️ **算子开发工程师** | [算子开发](#mainline-operator) — Ascend C 编程与自定义算子实战 |
| 📱 **应用开发工程师** | [应用开发（离线推理）](#mainline-appdev) — ATC 编译、ACL 推理与性能调优 |
| 📊 **推荐算法工程师** | [推荐系统](#mainline-recsys) — CTR 排序 + 召回全链路 |
| 🏫 **高校师生** | [高校教学方案专区](#university)  |
| 🏆 **竞赛选手** | [赛事备考专区](#competition) — CANNJudge 题库、大赛报名、备赛路径 |
| 🤝 **社区开发者** |看[技术博客](#技术博客)· [科研案例](#科研案例) 实践， [参与贡献](#参与贡献) — 贡献教程与案例 |

---

## 📋 前置基础（按需补，不用先全学完）

入门阶段，你只需 **Python + NumPy** 就能开始第一课。其他基础可以边学边补，学到卡住时回来查即可。

| 基础 | 什么时候需要 |
| :--- | :--- |
| **Python** | 全部课程，第一课就用到 |
| **NumPy / PyTorch** | 第一课起，张量与矩阵运算 |
| **C / C++** | 进入算子开发主线时 |

<details>
<summary><b>查看完整前置基础清单</b>（8 项 + 推荐资源 + 支撑课程）</summary>

| 基础 | 需要掌握的内容 | 推荐资源 | 支撑课程 |
| :--- | :--- | :--- | :--- |
| **Python** | 变量、函数、类、模块，能读懂并运行脚本 | [廖雪峰 Python 教程](https://liaoxuefeng.com/books/python/introduction/index.html) ｜ B站搜索「黑马程序员 Python」 | 全部课程 |
| **NumPy / PyTorch** | 会创建张量、做矩阵运算，了解 `Tensor` 基本操作 | [NumPy 官方快速入门](https://numpy.org/doc/stable/user/quickstart.html) ｜ [PyTorch 官方教程](https://pytorch.org/tutorials/) ｜ [李沐《动手学深度学习》](https://zh.d2l.ai/) | 01 人工智能基础、04 NPU 实践、推理/训练/推荐主线 |
| **C / C++** | 指针、函数、编译链接，能读懂 C++ 源码 | [菜鸟教程 C++](https://www.runoob.com/cplusplus/cpp-tutorial.html) ｜ B站搜索「黑马程序员 C++」 | Ascend C 算子开发主线 |
| **Linux 基础** | 常用命令、环境变量，能在服务器上运行程序 | [韩顺平《一周学会 Linux》](https://www.bilibili.com/video/BV1Sv411r7vd) ｜ [尚硅谷 Linux 教程](https://www.bilibili.com/video/BV1dW411M7xL) | 全部实践课程 |
| **数学基础** | 矩阵运算、梯度概念 | [宋浩《线性代数》](https://www.bilibili.com/video/BV1d7wAzsE8V) ｜ 备选：[MIT 18.06 线性代数（英文）](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | 深度学习、模型训练 |
| **深度学习概念** | 神经网络、反向传播，知道模型怎么训练和推理 | [吴恩达《深度学习专项》](https://www.bilibili.com/video/BV1FT4y1E74V) ｜ [李沐《动手学深度学习》](https://zh.d2l.ai/) | 01 人工智能基础、推理/训练主线 |
| **Git** | clone / commit / push / pull，会提交 PR | [廖雪峰 Git 教程](https://liaoxuefeng.com/books/git/introduction/index.html) ｜ [Pro Git 中文版](https://git-scm.com/book/zh/v2) | 全部课程、社区贡献 |
| **社区贡献** | 了解社区行为准则、CLA 签署、Issue / PR 提交流程 | [CANN 社区贡献指南](https://gitcode.com/cann/community) | 全部课程、社区共建 |

> 💡 入门阶段只需 Python + NumPy 即可开始第一课；进入算子开发主线时再补 C/C++，进入大模型主线时再补深度学习与 PyTorch。

</details>

---

<a id="cann-basics"></a>
## 🚀 从这里开始

### [CANN 基础（第一课）](./quick_start/cann_basics)

点击「在线体验」，在浏览器中运行你的第一个 Notebook，无需任何环境配置。

这是所有学习路径的**公共基础**（阶段一：CANN 基础，约 2 小时），也是后续五条学习主线的共同起点：

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 1 | [人工智能基础](./quick_start/cann_basics/01_ai_basics.ipynb) | AI 发展历程、算子概念（名称/类型/Tensor/shape） | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/01_ai_basics.ipynb) |
| 2 | [什么是 NPU](./quick_start/cann_basics/02_what_is_npu.ipynb) | 昇腾 NPU 硬件架构：DaVinci 核心、AI Core / Vector / Cube 计算单元 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/02_what_is_npu.ipynb) |
| 3 | [什么是 CANN](./quick_start/cann_basics/03_what_is_cann.ipynb) | CANN 异构计算架构与软件栈：分层架构、Ascend C、torch_npu 适配 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/03_what_is_cann.ipynb) |
| 4 | [NPU 实践](./quick_start/cann_basics/04_npu_practice.ipynb) | Hello NPU、张量运算、矩阵乘法性能对比、Batch 运算、图像高斯模糊卷积 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/cann_basics&scanFilePath=quick_start/cann_basics/04_npu_practice.ipynb) |
| 5 | [MNIST 手写数字识别（可选）](./contrib/tutorials/swanlab_examples/mnist/mnist.ipynb) | 你的第一个模型训练：数据加载 → 模型构建 → 训练 → 评估，配套 SwanLab 可视化 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

> 💡 想用完整 NPU 环境做进阶练习？进入 [CANNLab](https://gitcode.com/org/cann/cannlab) 创建云环境（初始 100 小时，积分可兑换时长）。

---

## 🧭 学完基础，选择一条主线

完成公共基础（阶段一）后，按你关心的场景选一个方向深入。五条主线互不冲突，可以都学，也可以只学一条。

| 主线 | 适合人群 | 总时长 | 核心产出 | 难度 |
| :--- | :--- | :---: | :--- | :---: |
| [🏋️ 大模型训练](#mainline-training) | 想做模型训练与微调 | ~16h | SFT/RL 训练与微调能力 | ★★★ |
| [🧠 大模型推理](#mainline-inference) | 想快速跑通大模型、做推理优化 | ~10h | 端到端推理调优能力 | ★★☆ |
| [⚙️ 算子开发](#mainline-operator) | 想掌握底层编程、做算子级优化 | ~20h | 自定义算子开发与接入 | ★★★ |
| [📱 应用开发（离线推理）](#mainline-appdev) | 想做 ACL 离线推理与应用部署 | ~6h | ATC 编译 + ACL 推理 + 调优 | ★★☆ |
| [📊 推荐系统](#mainline-recsys) | 想做推荐/召回业务落地 | ~8h | CTR 排序 + 召回全链路 | ★★☆ |

---

<a id="mainline-training"></a>
### 主线一：🏋️ 大模型训练

> 从微调实战到 SFT/RL 训练全流程，系统掌握大模型训练与性能优化能力。

**阶段一：[CANN 基础](#cann-basics)（公共基础，约 2h）** ✅ 已在此完成 → 直接进入本主线后续阶段

**阶段二：微调实战案例（SwanLab 共建）🚧**

| 序号 | 课程 | 课程内容 | 状态 |
| :---: | :--- | :--- | :---: |
| 1 | [医学模型微调](./contrib/tutorials/swanlab_examples/qwen3_medical_sft) | Qwen3 医学领域 SFT + SwanLab 可视化 | ✅ 已上线 |
| 2 | [ms-swift 框架微调](https://docs.swanlab.cn/course/llm_train_course/03-sft/8.other_frameworks/ms-swift.html) | ms-swift 框架微调 + SwanLab 可视化 | 🚧 建设中 |
| 3 | [Qwen3-smolVLM 多模态微调](https://docs.swanlab.cn/course/llm_train_course/06-multillm/2.qwen3_smolvlm_muxi/) | 多模态拼接微调 + SwanLab 可视化 | 🚧 建设中 |
| 4 | [CosyVoice 语音微调](https://docs.swanlab.cn/course/llm_train_course/07-audio/1.cosyvoice-sft/) | 语音模型微调 + SwanLab 可视化 | 🚧 建设中 |

> 💡 案例正在从 SwanLab 平台迁移至 `contrib/tutorials/swanlab_examples/`，完成后将在 CANNLab 环境提供在线体验。


<details>
<summary><b>阶段三：SFT/RL 初阶课程</b></summary>

> 📖 **配套课件**：[SFT 训练流程](./tutorials/sft_training_pipeline/slides)，[RL 训练流程](./tutorials/rl_training_pipeline/slides)

| 序号 | 课程（实践） | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 5 | [SFT 训练系列（初阶）](./tutorials/sft_training_pipeline) | Wordle 任务与 SFT 原理 / TorchTitan-FSDP 框架与环境配置 / Qwen3-1.7B 基线训练与推理评测 / 融合算子接入与 Profiling 分析 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 6 | [RL 训练系列（初阶）](./tutorials/rl_training_pipeline) | verl+vLLM-Ascend 环境与框架概览 / RL+GRPO 核心概念与 KL 稳定性 / Wordle Agent Loop 与奖励函数 / 训练指标监控与崩塌调优 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

</details>

<details>
<summary><b>阶段四：SFT/RL 中阶课程</b></summary>

| 序号 | 课程 | 课程内容 | 状态 |
| :---: | :--- | :--- | :---: |
| 7 | [SFT 训练系列（中阶）](./tutorials/sft_training_pipeline) | Attention 算子优化（SDPA→VarLen）/ Sequence Packing 与 TND 变长注意力 / FSDP·TP·CP 分布式通信与体积分析 / VarLen+CP 协同与 TP2/FSDP2 对比 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 8 | [RL 训练系列（中阶）](./tutorials/rl_training_pipeline) | FSDP→TorchTitan-NPU FSDP2 后端切换 / FSDP2·offload·TND 变长注意力核心特性 / Wordle 三步训练实践 / 后端切换总结与排查 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

</details>

<details>
<summary><b>阶段五：SFT/RL 高阶课程 🚧</b></summary>

| 序号 | 课程 | 课程内容 | 状态 |
| :---: | :--- | :--- | :---: |
| 9 | SFT 训练系列（高阶） | 计算图静态化与 AutoFuse 融合优化，selective AC 显存调优 | 🚧 建设中 |
| 10 | RL 训练系列（高阶） | 性能基线搭建与瓶颈定位，FSDP2/TP/CP/PP 并行调优 | 🚧 建设中 |

</details>


---

<a id="mainline-inference"></a>
### 主线二：🧠 大模型推理

> 以 Qwen3 端到端推理案例为主线，理论课件 + 双模型实践双轨并行，自然贯通「跑通推理 → 发现瓶颈 → 融合优化 → 量化与算子 → 图模式调优」。

**阶段一：[CANN 基础](#cann-basics)（公共基础，约 2h）** ✅ 已在此完成 → 直接进入本主线后续阶段

**阶段二：初阶 — 跑通推理（约 3-4h）**

**目标**：在 NPU 上跑通大模型推理，建立 Baseline 并完成首次 Profiling。

> 📖 **配套课件**：[大语言模型基础](./tutorials/llm_inference/slides/intermediate/01_llm_fundamentals.pdf) ｜ [CANN 开源推理仓库](./tutorials/llm_inference/slides/intermediate/02_cann_inference_repository_overview.pdf) ｜ [大模型推理优化基础](./tutorials/llm_inference/slides/intermediate/03_llm_inference_optimization_fundamentals.pdf)


| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 1 | [章节介绍](./tutorials/llm_inference/qwen3_8b/01_chapter_intro.ipynb) | 全流程概览，知道接下来要做什么 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 2 | [Baseline 推理](./tutorials/llm_inference/qwen3_8b/02_baseline_inference.ipynb) | 跑通 Qwen3-8B BF16 推理，感知大模型推理 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 3 | [Profiling 分析](./tutorials/llm_inference/qwen3_8b/03_profiling_analysis.ipynb) | 对 Baseline 做 Profiling，定位 RMSNorm 等小算子链路瓶颈 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |


<details>
<summary><b>阶段三：中阶 — 融合优化与量化（约 4-6h）</b></summary>

**目标**：走完「融合优化 → 量化 → 自定义算子接入」工程闭环。

> 📖 **配套课件**：[大模型量化基础](./tutorials/llm_inference/slides/intermediate/04_llm_quantization_fundamentals.pdf) ｜ [Profiling 与性能瓶颈定位](./tutorials/llm_inference/slides/intermediate/05_profiling_and_performance_bottleneck_analysis.pdf)

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 4 | [Dense RMSNorm NPU 融合优化](./tutorials/llm_inference/qwen3_8b/04_npu_optimization.ipynb) | 切换融合开关，A/B 对比验证算子融合的性能收益 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 5 | [量化 Qwen3-8B 模型](./tutorials/llm_inference/qwen3_8b/05_quantization_qwen3_8b.ipynb) | AMCT 工具导出 W8A8 权重 → 量化推理 → Profiling 定位 `QuantBatchMatmulV3` 瓶颈 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 6 | [自定义量化算子开发并接入 Qwen3-8B](./tutorials/llm_inference/qwen3_8b/06_custom_matmul_operator_development_and_integration_with_qwen3_8b.ipynb) | 用 Ascend C 实现 `QmmCustom` → 编译 → 替换瓶颈算子 → 验证收益 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

> 💡 步骤 ⑥ 需要自定义算子开发能力，前置学习 [主线三：Ascend C 算子开发系列](#mainline-operator)。至此完成推理主线核心，已具备端到端 BF16 推理调优与量化算子接入能力。

</details>

<details>
<summary><b>阶段四：高阶 — 图模式调优与高级理论（可选）</b></summary>

**目标**：掌握图执行模式调优与权重预取，系统理解大模型推理高级优化技术。

> 📖 **配套课件（高级）**：
> [01 推理优化基础](./tutorials/llm_inference/slides/advance/01_fundamentals_of_llm_inference_optimization.pptx) ｜ [02 并行策略设计](./tutorials/llm_inference/slides/advance/02_parallel_strategy_design.pptx) ｜ [03 MTP/多流/预取](./tutorials/llm_inference/slides/advance/03_core_inference_optimization_mtp_multi_stream_prefetch.pptx) ｜ [04 量化优化与实践](./tutorials/llm_inference/slides/advance/04_llm_quantization_optimization_and_practice.pptx) ｜ [05 主流框架 CANN 优化](./tutorials/llm_inference/slides/advance/05_cann_optimization_for_mainstream_inference_frameworks.pptx) ｜ [06 多模态生成推理](./tutorials/llm_inference/slides/advance/06_multimodal_generation_inference_optimization.pptx) ｜ [07 模型 Agent 基础](./tutorials/llm_inference/slides/advance/07_model_agent_fundamentals_from_skills_to_inference_optimization.pptx)

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 7 | [图模式优化：GE Graph vs NPU Graph EX](./tutorials/llm_inference/qwen3_8b/07_npu_graph_optimization.ipynb) | 对比 eager、ge_graph 与 npugraph_ex 三种执行模式，给出图后端选择建议 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 8 | [GE Graph o_proj 权重预取](./tutorials/llm_inference/qwen3_8b/08_ge_o_proj_prefetch.ipynb) | 固定 GE 图模式，对比 o_proj 权重预取关闭/16MiB/32MiB 三组性能收益 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |

> 💡 更多模型与复杂部署优化实践，见 [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 完整仓库。

</details>

---

<a id="mainline-operator"></a>
### 主线三：⚙️ 算子开发

> 从「10 分钟体验算子」到系统掌握 Ascend C 编程范式，具备自定义算子开发与工程化能力。

**阶段一：[CANN 基础](#cann-basics)（公共基础，约 2h）** ✅ 已在此完成 → 直接进入本主线后续阶段

**阶段二：体验算子（约 2h）**

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 1 | [10 分钟体验自定义算子](./quick_start/first_custom_operator/first_custom_operator.ipynb) | 第一个自定义算子开发，感知 Ascend C 编写与编译全流程 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/first_custom_operator&scanFilePath=quick_start/first_custom_operator/first_custom_operator.ipynb) |
| 2 | [10 分钟体验算子 API 调用](./quick_start/first_operator_api_call/first_operator_api_call.ipynb) | 第一个算子调用，感知算子调用与价值 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=quick_start/first_operator_api_call&scanFilePath=quick_start/first_operator_api_call/first_operator_api_call.ipynb) |


<details>
<summary><b>阶段三：系统学习（约 12h）</b></summary>

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 3 | [Ascend C 算子开发系列（Kernel 直调版）](./tutorials/ascendc_operator_development_light) | 算子基础概念、编程范式、Vector/Cube/融合算子开发与调试调优 | [在线体验](./tutorials/ascendc_operator_development_light) |

</details>

<details>
<summary><b>阶段四：算子开发实战</b></summary>

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 4 | [Ascend C 算子开发系列（算子工程版）](./tutorials/ascendc_operator_development) | 工程化开发流程、开源社区贡献规范与玩法 | [在线体验](./tutorials/ascendc_operator_development) |
| 5 | Vector 算子开发 | Vector 算子开发实战 | 🚧 建设中 |
| 6 | Conv 算子开发实战 | 卷积算子开发核心概念与实践 | 🚧 建设中 |
| 7 | [MC2 融合算子实战](./tutorials/MC2_fused_operator_development) | Matmul/Conv/通算融合等典型算子实战 | [在线体验](./tutorials/MC2_fused_operator_development) |

> 💡 **进阶练习**：[CANNJudge 算子题库](https://cannjudge.cn) 在线刷题 → [CANN 大赛专区](https://competition.gitcode.com/competition?type=cann) 参赛验证

</details>

<details>
<summary><b>阶段五：算子性能优化（进阶）</b></summary>

**目标**：通过典型算子实践，掌握从性能基线、瓶颈分析到分步优化与精度校验的调优流程。

| 类别 | 序号 | 课程 | 课程内容 | 样例链接 |
| :--- | :---: | :--- | :--- | :--- |
| Vector 类 | 8 | Vector 算子优化 | 逐元素、归约、归一化与路由算子的 RegBase 改写、融合、多核并行及访存优化 | [GELU](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/gelu_eltwise_regbase_story) / [Softmax](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/softmax_regbase_story) / [RmsNormQuant](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/rms_norm_quant_story) / [KvRmsNormRoPE](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/kv_rms_norm_rope_cache_story) / [MoeInitRouting](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/moe_init_routing_story) |
| Cube 类 | 9 | 矩阵乘算子优化 | 矩阵乘与分组矩阵乘的 Tiling、数据搬运、尾轮负载均衡及低精度量化 | [MatMul](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/matmul_story) / [Grouped MatMul](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/grouped_matmul_story) |
| CV 融合类 | 10 | Cube/Vector 融合优化 | 极简 FA 的 CV 核融合优化实践、全量化 FA 实现、注意力残差的模板选择与两阶段流水，以及 KDA 的 Chunkwise 并行 | [FlashAttnLite](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/flash_attn_lite_story) / [FIA](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/full_quant_fused_infer_attention_score_story) / [AttnRes](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/block_attn_res_story) / [Kimi Delta Attention Lite](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/kimi_delta_attn_lite_story) |
| 通算融合类 | 11 | 通信与计算融合优化 | MoE Token 分发与合并中的通信、计算融合及性能优化 | [MoE Dispatch & Combine](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/moe_dispatch_and_combine_story) |
| 专题类 | 12 | SIMD VF、SIMT 与 Scalar 优化专题 | SIMD VF 广播、归约与数据变换；SIMT 不规则访存与并行计数；ScalarBound 诊断及标量开销优化 | [SIMD VF](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/simd_vf_story) / [VF 数据变换](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/vf_data_transform_story) / [SIMT Scatter](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/simt_scatter_story) / [SIMT Histogram](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/simt_histogram_story) / [Scalar](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/scalar_story) |

> 💡 **实践入口**：[cann-samples](https://gitcode.com/cann/cann-samples) 是 CANN 算子实战样例与性能调优知识库，提供算子实现、优化讲解及编译运行与精度校验示例，可从上表或[性能优化样例总览](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance)选择学习内容。
>
> **学习方式**：
>
> 1. 克隆仓库，按仓库首页的环境部署指南配置 CANN 与依赖。
> 2. 按所选样例 README 在本地编译执行，完成精度校验并记录性能基线。
> 3. 结合优化文档阅读源码；有分步版本的样例可逐版对比，理解各项优化的作用。
> 4. 尝试修改 Tiling、数据搬运或流水安排，重新编译、校验精度并对比性能。
>
> 适用芯片、CANN 版本和运行命令以对应样例说明为准。

</details>

---

<a id="mainline-appdev"></a>
### 主线四：📱 应用开发（离线推理）

> 以 YOLOv13 目标检测为例，走通 ONNX 导出 → ATC 离线编译 → ACL 推理 → 性能调优全流程，掌握昇腾应用开发核心能力。

**阶段一：[CANN 基础](#cann-basics)（公共基础，约 2h）** ✅ 已在此完成 → 直接进入本主线后续阶段

**阶段二：YOLOv13 离线推理实战（约 4h）**

**目标**：在昇腾 NPU 上完成 YOLOv13 端到端离线推理，掌握 ATC 模型编译与 ACL 推理应用开发。

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 1 | [章节介绍](./reference_practice/yolov13_offline_inference/01_chapter_intro.ipynb) | 学习目标、流程和目录说明 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 2 | [环境准备与模型转换](./reference_practice/yolov13_offline_inference/02_model_prepare.ipynb) | 准备 YOLOv13、导出 ONNX 并使用 ATC 生成 OM | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 3 | [ACL 离线推理](./reference_practice/yolov13_offline_inference/03_acl_offline_inference.ipynb) | 编译并运行 ACL C++ 推理程序 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 4 | [结果处理与性能调优](./reference_practice/yolov13_offline_inference/04_result_and_tuning.ipynb) | 后处理、可视化和多流参数调优 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |
| 5 | [章节练习](./reference_practice/yolov13_offline_inference/05_chapter_practice.ipynb) | 巩固模型转换、推理和调优知识 | [open in CANNLab](https://gitcode.com/org/cann/cannlab) |


> 💡 应用开发系列更多课程（GE/ATC/模型转换全流程）持续建设中。

---

<a id="mainline-recsys"></a>
### 主线五：📊 推荐系统

> 基于 Torch-RecHub 跑通推荐系统全链路：排序 → 召回 → 多任务 → 工程化 → NPU 推理优化。

**阶段一：[CANN 基础](#cann-basics)（公共基础，约 2h）** ✅ 已在此完成 → 直接进入本主线后续阶段

**阶段二：跑通 CTR（约 2h）**

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 1 | [QuickStart：CTR 预测（DeepFM）](./contrib/tutorials/torch-rechub/00_QuickStart_CTR_DeepFM.ipynb) | DataFrame → Feature → DeepFM → CTRTrainer → AUC | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/00_QuickStart_CTR_DeepFM.ipynb) |


<details>
<summary><b>阶段三：进阶建模（约 4h）</b></summary>

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 2 | [序列兴趣建模：DIN](./contrib/tutorials/torch-rechub/01_Ranking_DIN.ipynb) | 历史行为序列、SequenceFeature 与 DIN attention | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/01_Ranking_DIN.ipynb) |
| 3 | [匹配/召回：DSSM + Annoy](./contrib/tutorials/torch-rechub/02_Matching_DSSM.ipynb) | 双塔召回与向量 Top-K 检索 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/02_Matching_DSSM.ipynb) |
| 4 | [多任务学习：MMOE](./contrib/tutorials/torch-rechub/03_MultiTask_MMOE.ipynb) | 多目标建模、expert、gate 与 tower | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/03_MultiTask_MMOE.ipynb) |

</details>

<details>
<summary><b>阶段四：工程化（约 2h）</b></summary>

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 5 | [实验跟踪：WandB / SwanLab / TensorBoardX](./contrib/tutorials/torch-rechub/04_Experiment_Tracking_Light.ipynb) | WandB / SwanLab / TensorBoardX 轻量实验跟踪 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/04_Experiment_Tracking_Light.ipynb) |
| 6 | [模型导出与推理验证：ONNX](./contrib/tutorials/torch-rechub/05_Model_Export_and_Serving.ipynb) | ONNX 导出、ONNXRuntime 推理验证和量化入口 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/05_Model_Export_and_Serving.ipynb) |

</details>

<details>
<summary><b>阶段五：NPU 推理优化（约 1h）</b></summary>

| 序号 | 课程 | 课程内容 | 运行方式 |
| :---: | :--- | :--- | :--- |
| 7 | [DIN 多进程控核推理](./contrib/tutorials/torch-rechub/10_NPU_DIN_Inference_MultiInstance_CoreControl.ipynb) | 多进程（spawn Pool）运行 DIN 推理，通过 TorchAir `ge.aicoreNum` 整图控核，含 1000 请求基准与 latency 统计 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/torch-rechub&scanFilePath=contrib/tutorials/torch-rechub/10_NPU_DIN_Inference_MultiInstance_CoreControl.ipynb) |

</details>

---

> 💡 **推荐学习路径**：选方向 → 看教程 → CANNJudge 刷题 / CANNLab 实验 → 参赛验证 → PR 到 contrib/

---

## 📖 完整课程体系（查阅用）

> 以下为课程体系的完整索引，适合有明确学习目标、需要定位具体课程的开发者。初次学习请从上方「从这里开始」入手。

### 全栈课程体系地图

按 **技术领域 × 难度等级** 双维度组织，高手可直接跳过入门内容，精准定位高阶课程。

<details open>
<summary><b>折叠全栈课程体系地图</b>（11 个技术领域 × 3 级难度）</summary>

| 技术领域 | 初级课程 | 中级课程 | 高级课程 |
| :--- | :--- | :--- | :--- |
| **⚙️ 算子开发 · Ascend C** | [Kernel 直调版](./tutorials/ascendc_operator_development_light) | — | — |
| **⚙️ 算子开发 · PyASC** | [PyASC 初级（概述/核函数基础）](./tutorials/pyasc_operator_development) | [PyASC 中级（Vector/Matmul）](./tutorials/pyasc_operator_development) | [PyASC 高级（融合/调优）](./tutorials/pyasc_operator_development) |
| **⚙️ 算子开发 · PyPTO** | [PyPTO 初级（概述/基础/实践）](./tutorials/pypto_development) | [PyPTO 中级（中高级算子实践）](./tutorials/pypto_development) | — |
| **⚙️ 算子开发实战** | — | Vector 🚧 · matmul 🚧 · [Conv 实战](./tutorials/conv_operator_development) | [MC2 融合算子](./tutorials/MC2_fused_operator_development) · MoE 🚧 · FA 🚧 |
| **🧠 大模型推理** | [推理理论课件初级](./tutorials/llm_inference/slides/intermediate) · [Qwen3-8B 实践初级](./tutorials/llm_inference/qwen3_8b) | [推理理论课件中级](./tutorials/llm_inference/slides/intermediate) · [Qwen3-8B 实践中级](./tutorials/llm_inference/qwen3_8b) | [推理理论课件高级](./tutorials/llm_inference/slides/advance) · [Qwen3-8B 实践高级](./tutorials/llm_inference/qwen3_8b) · [Sana-Video](./reference_practice/model_inference_optimization/sana_video) · [PyTorch 算子优化](./reference_practice/pytorch_online_inference_operator_optimize) |
| **🏋️ 大模型训练** | [SFT 初阶](./tutorials/sft_training_pipeline) · [RL 初阶](./tutorials/rl_training_pipeline) · [SwanLab 微调](#swanlab) | [SFT 中阶](./tutorials/sft_training_pipeline) · [RL 中阶](./tutorials/rl_training_pipeline) | SFT/RL 高阶 🚧 |
| **📊 推荐系统** | [CTR 预测](./contrib/tutorials/torch-rechub/00_QuickStart_CTR_DeepFM.ipynb) · [DIN](./contrib/tutorials/torch-rechub/01_Ranking_DIN.ipynb) · [DSSM](./contrib/tutorials/torch-rechub/02_Matching_DSSM.ipynb) · [MMOE](./contrib/tutorials/torch-rechub/03_MultiTask_MMOE.ipynb) · [实验跟踪](./contrib/tutorials/torch-rechub/04_Experiment_Tracking_Light.ipynb) · [模型导出](./contrib/tutorials/torch-rechub/05_Model_Export_and_Serving.ipynb) | [DIN 多进程控核推理](./contrib/tutorials/torch-rechub/10_NPU_DIN_Inference_MultiInstance_CoreControl.ipynb) | — |
| **🔗 图框架** | [GE 图引擎](./tutorials/ge_development) · [TorchAir](./tutorials/TorchAir_development) · [AutoFusion](./tutorials/autofusion_development) | [GE 图编译与优化](./tutorials/ge_development) · [TorchAir 进阶](./tutorials/TorchAir_development/README.md) · [AutoFusion 融合原理](./tutorials/autofusion_development/02_autofusion_principles) | [GE 实践与问题定位](./tutorials/ge_development/05_practice_and_troubleshooting) · [TorchAir 实践](./tutorials/TorchAir_development/04_advanced_features) · [AutoFusion 实战](./tutorials/autofusion_development/03_practice_and_debugging) |
| **📡 通信** | [HiXL 单边通信](./tutorials/hixl_development) · [集合通信](./tutorials/hccl_development) | [HiXL 传输优化](./tutorials/hixl_development) · [HCCL 算子开发](./tutorials/hccl_development) | [HCCL AIV 引擎算子开发](./tutorials/hccl_development/slides/07_hccl_aiv_engine.pdf) · [AI 大集群问题定位](./tutorials/hccl_development/slides/08_troubleshooting_hccl_for_ai_clusters.pdf) |
| **📱 应用开发** | [YOLOv13 离线推理](./reference_practice/yolov13_offline_inference) | — | — |
| **🤖 CANNBot** | [CANNBot](https://gitcode.com/cann/cannbot-skills) · [Notebook 实操篇](./tutorials/CANNBot/notebooks/README.md) · [CANNBot 基础](./tutorials/CANNBot/README.md#初级课程cannbot基础) | [CANNBot 算子开发进阶](./tutorials/CANNBot/README.md#中级课程cannbot算子开发进阶) | — |

</details>

---

<a id="university"></a>
## 🏫 高校教学方案专区（供高校开课参考，建设中）

> 以**原子课程为基础积木**（技术栈四层 × 难度三级，135 门），支撑**启航营**、**传统学科课程**、**CANN 新增课程**、**CANN 教材**四大课程形态，覆盖从短期集训到高校学分课、从入门到专家的完整学习路径。详见 [CANN 课程体系总体规划](./standard_course/README.md)。

<details>
<summary><b> CANN 课程总览 </b></summary>

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        原子课程（00）—— 体系基础                            │
│              135 门积木（技术栈四层 × 难度三级）                            │
│  ┌────────────┬────────────┬────────────┬────────────────────────────┐  │
│  │  L1 应用层  │  L2 框架层  │  L3 加速库  │  L4 编程·编译·NPU 架构     │  │
│  │   (44门)    │   (29门)    │   (13门)    │        (49门)              │  │
│  └────────────┴────────────┴────────────┴────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 │ 支撑（节选/教学化/拼装/系统化）
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼──────────┐ ┌───────▼──────────┐ ┌────────▼───────────┐
│   启航营（01）       │ │ 传统学科课程（02）  │ │  CANN 新增课程（03）  │
│   原子课程节选        │ │  原子课程教学化      │ │   原子课程拼装        │
│ ┌────────────────┐  │ │ ┌────────────────┐ │ │ ┌────────────────┐  │
│ │ 两天营 ≈14h     │  │ │ │ 计组外挂 6–8h   │ │ │ │ AI计算架构 64h  │  │
│ │ 两周营 ≈60h     │  │ │ │ 体系结构三幕剧   │ │ │ │ 数据智能 32h    │  │
│ │ 结业：add_rms_  │  │ │ │ 12 学科清单      │ │ │ │ 加速计算 32-48h │  │
│ │   norm→双算子   │  │ │ └────────────────┘ │ │ │ 数据科学 24-32h │  │
│ └────────────────┘  │ └────────────────────┘ │ │ 深度学习 32-48h │  │
└─────────────────────┘                          │ │ 生成式 AI 32h   │  │
                                                  │ └────────────────┘  │
                                                  └─────────────────────┘
                                 │
                                 │ 系统化
                  ┌──────────────▼──────────────┐
                  │       教材区（04）             │
                  │     原子课程系统化             │
                  │ ┌──────────────────────────┐ │
                  │ │ 教材一：Ascend C 并行程序设计│ │
                  │ │  （↔课程三/启航营算子部分） │ │
                  │ ├──────────────────────────┤ │
                  │ │ 教材二：人工智能芯片与系统   │ │
                  │ │  （↔体系结构三幕剧/课程一）  │ │
                  │ └──────────────────────────┘ │
                  └─────────────────────────────┘
```
</details>

### CANN 课程架构总览

| 分区 | 定位 | 与原子课程关系 | 规模 | 详细大纲 |
| :--- | :--- | :--- | :--- | :--- |
| **00 原子课程** | **基础积木**：技术栈四层 × 难度三级的原子粒度课程 | 体系基础 | 135 门 | [原子课程总纲](./standard_course/00_atomic_courses/README.md) |
| **01 启航营** | 短期集训：两天/两周两营递进，快速入门与实战 | 原子课程**节选** | 两天营 ≈14h / 两周营 ≈60h | [启航营课程总览](./standard_course/01_bootcamp/README.md) |
| **02 传统学科课程** | 学科融入：CANN 知识融入大学传统学分课程，换芯不换课 | 原子课程**教学化** | 计组 + 体系结构 + 12 学科 | [传统学科课程目录](./standard_course/02_traditional_courses/README.md) |
| **03 新增课程** | 高校学分课：六门系统课程 | 原子课程**拼装** | 已排课 2 + 规划 4 | [CANN 新增课程汇总](./standard_course/03_new_courses/README.md) |
| **04 教材区** | 经典书籍：双教材支撑课程体系 | 原子课程**系统化** | 教材一 + 教材二 | [CANN 教材区概览](./standard_course/04_new_books/README.md) |

<details>
<summary><b>00 原子课程——体系基础（135 门）</b></summary>

> 按技术栈四层（应用层/AI 框架层/加速库/编程编译 NPU 架构）× 难度三级（初级/中级/高级）组织，每门原子课可单独支撑一节课（45–90 分钟），是启航营、传统学科课程、新增课程、教材的共同内容来源。

| 技术栈层 | 初级 | 中级 | 高级 | 小计 | 涵盖方向 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **L1 应用层** | 8 | 17 | 19 | **44** | 深度学习 / 计算机视觉 / 生成式 AI / 大语言模型 / 科学计算 / 推荐系统 |
| **L2 框架层** | 7 | 11 | 11 | **29** | 框架基础 / 训练技术 / 推理技术 / 图框架 / 进阶主题 |
| **L3 加速库** | 3 | 8 | 2 | **13** | Aclnn / GE / 通信库 / C++ 并行算法 / Python 加速库 |
| **L4 编程·编译·NPU 架构** | 5 | 30 | 14 | **49** | Ascend C / PyPTO / TileLang / PyAsc / CANNBot / NPU 架构 |
| **合计** | **23** | **66** | **46** | **135** | — |

> 📖 详细大纲：[原子课程总纲](./standard_course/00_atomic_courses/README.md)

</details>

<details>
<summary><b>01 启航营——短期集训（两天营 + 两周营）</b></summary>

> 面向 CANN 算子开发入门者的集中式训练营，理论与实践 1:1 配对，以真实算子开发为主线，结业即可具备 NPU 算子开发与集成能力。两营递进，结业算子同源。

| 维度 | 两天营 | 两周营 |
| :--- | :--- | :--- |
| **总学时** | 2 天 ≈ 14h（理论 7 + 实践 7） | 10 个工作日 ≈ 60h（理论 20 + 实践 40） |
| **结业算子** | `add_rms_norm`（矢量单算子） | `add_rms_norm` + `quant_matmul`（双算子）+ PyTorch 集成 |
| **讲数** | 7 讲 | 13 讲 |
| **课程大纲** | [两天营大纲](./standard_course/01_bootcamp/syllabus_two_days_bootcamp.md) | [两周营大纲](./standard_course/01_bootcamp/syllabus_two_weeks_bootcamp.md) |
| **讲义目录** | [two_days_course/](./standard_course/01_bootcamp/two_days_course/) | [two_weeks_course/](./standard_course/01_bootcamp/two_weeks_course/) |

</details>

<details>

<summary><b>02 传统学科课程——换芯不换课</b></summary>

> CANN/昇腾知识融入大学传统学分课程，核心原则：**换芯不换课**——传统课程大纲主体不动，CANN 以「单元包 / 对照案例 / 实验替换」外挂式融入，教师零改造或低改造即可采用。

| 传统课程 | 融入模式 | 交付形态 | 文件 |
| :--- | :--- | :--- | :--- |
| **计算机组成原理**（本科大二必修） | M2/M5 轻量融入 | 外挂式 6–8 学时 | [computer_organization.md](./standard_course/02_traditional_courses/computer_organization.md) |
| **计算机体系结构**（本科高年级/研究生） | M2/M5 主讲融入 | "AI 时代的体系结构挑战"三幕剧（3 次主讲课 × 135 分钟） | [computer_architecture.md](./standard_course/02_traditional_courses/computer_architecture.md) |
| 其他 12 学科 15 门课（并行计算/DL 系统/图像处理/编译/OS/嵌入式/数值/计算科学/数据科学/HPC 等） | M2–M4 | 方向性规划 | [融入计划](./standard_course/02_traditional_courses/traditional_course_integration_plan.md) |

> 📖 详细大纲：[传统学科课程目录](./standard_course/02_traditional_courses/README.md)

</details>

<details>
<summary><b>03 新增课程大纲（六门高校学分课）</b></summary>

> 由原子课程拼装而成，覆盖从 AI 计算架构到行业应用的完整学习路径。每门课程单独成文，关注前后章节连续性；除大作业外每章节配双平台练习。

| # | 课程 | 学时 | 状态 | 课程大纲 |
| :---: | :--- | :--- | :---: | :--- |
| 一 | AI 计算与神经网络计算架构实践 | 64（16 周） | ✅ 已排课 | [课程大纲](./standard_course/03_new_courses/ai_computing_and_neural_network_architecture_practice/syllabus.md) |
| 二 | 昇腾 AI 与复杂系统数据智能 | ≈32（16 周） | ✅ 已排课 | [课程大纲](./standard_course/03_new_courses/ascend_ai_and_complex_system_data_intelligence/syllabus.md) |
| 三 | 加速计算 | 32–48 | 🆕 规划 | [课程大纲](./standard_course/03_new_courses/accelerated_computing.md) |
| 四 | 数据科学 | 24–32 | 🆕 规划 | [课程大纲](./standard_course/03_new_courses/data_science.md) |
| 五 | 深度学习 | 32–48 | 🆕 规划 | [课程大纲](./standard_course/03_new_courses/deep_learning.md) |
| 六 | 生成式 AI | 32 | 🆕 规划 | [课程大纲](./standard_course/03_new_courses/generative_ai.md) |

</details>

<details>
<summary><b>04 教材区——双教材支撑</b></summary>

> 双教材与课程形成「教材↔课程」双载体形态，将原子课程内容按教材体例系统化组织。

| 教材 | 定位 | 状态 | 对应课程 |
| :--- | :--- | :---: | :--- |
| **《Ascend C 并行程序设计》** | 对标 PMPP——昇腾算子开发权威教材 | ✅ 已完成初稿 | 课程三（加速计算）、启航营算子部分 |
| **《人工智能芯片与系统》** | 对标 AI Accelerators——NPU 架构与系统教材 | 🆕 规划 | 传统学科体系结构三幕剧、课程一架构部分 |

> 📖 详细大纲：[教材区概览](./standard_course/04_new_books/README.md)

</details>

---

<a id="course-collaboration"></a>
## 🤝 课程共建专区

<details>
<summary><b>展开查看课程共建专区</b>（13 门高校共建课程）</summary>

> 基于 CANN 的计算机系统核心课程系列，由高校教师与 CANN 社区共建。每门课程在本仓 `contrib/tutorials/` 下提供实践 Notebook，配套理论课程可[点击查看](https://mp.weixin.qq.com/mp/appmsgalbum?action=getalbum&__biz=MjM5OTkzMjM1Mw==&scene=2&album_id=4693832807398162433&count=3#wechat_redirect)。

| 课程大类 | 贡献学校 | 课程 | 状态 |
| :--- | :--- | :--- | :---: |
| 计算机组成原理与体系结构 | 华中科技大学 | [计算机组成原理与体系结构](./contrib/tutorials/computer_composition_and_architecture) | ✅ 已发布 |
| 面向昇腾智算平台的智能计算系统 | 东南大学 | [面向国产智算平台的智能计算系统（Qwen 算子开发）](./contrib/tutorials/qwen_ops) | ✅ 已发布 |
| 基于昇腾处理器的深度学习实验 | 哈尔滨工业大学（威海） | [基于昇腾处理器的深度学习实验](./contrib/tutorials/ascend_ai_lab) | ✅ 已发布 |
| 《人工智能安全》项目化实践案例 | 南京信息工程大学 | [《人工智能安全》项目化实践案例](./contrib/tutorials/ai_security_nuist) | ✅ 已发布 |
| 昇腾 AI 端云协同与多模态综合实验 | 上海交通大学 | [昇腾 AI 端云协同与多模态综合实验](./contrib/tutorials/ascend_multimodal_practice) | ✅ 已发布 |
| 智能计算系统 | 北京航空航天大学 | [智能计算系统（机器学习系统）](./contrib/tutorials/machine_learning_system) | ✅ 已发布 |
| 智能计算系统 | 华南理工大学 | [嵌入式智能计算课程](./contrib/tutorials/ai_computing_embedded_system) | ✅ 已发布 |
| 人工智能开源通识 | 华东师范大学 | [算子开发与应用](./contrib/tutorials/operator_development_and_application) | ✅ 已发布 |
| 并行计算 | 北京师范大学 | [并行编程原理与实践](./contrib/tutorials/parallel_programming_principles_and_practice) | ✅ 已发布 |
| 并行计算 | 深圳职业技术大学 | [并行计算：基于鲲鹏与昇腾的实践](./contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend) | ✅ 已迁移 |
| 编译原理 | 哈尔滨工业大学 | [PyAsc 编译原理课程](./contrib/tutorials/pyasc_compiler_development) | ✅ 已发布 |
| 数据结构与并行算法 | 中国科学技术大学 | [数据结构与算法设计（计算实践）](./contrib/tutorials/data_structures_compute) | ✅ 已发布 |
| 数据结构与并行算法 | 北京林业大学 | [高性能计算数据结构课程](./contrib/tutorials/data_structure_for_hpc) | ✅ 已发布 |

</details>

---

<a id="competition"></a>
<details>
<summary><b>🏆 赛事备考专区</b></summary>

> 按赛事分类聚合所有相关内容，直接匹配赛事考点，选手无需从通用课程中自行筛选。

| 赛事 | 赛事说明 | 备赛资源 | 赛事权益 |
| :--- | :--- | :--- | :--- |
| [**CANNJudge 算子题库**](https://cannjudge.cn) | 开放题库，Ascend C 算子编程在线刷题，实时评测；含历届算子赛真题 | [算子开发练习](https://cannjudge.cn) → 在线刷题<br/>[Ascend C 算子开发系列](./tutorials/ascendc_operator_development_light) → 前置课程<br/>[MC2 融合算子实战](./tutorials/MC2_fused_operator_development) → 高阶练习 | 实时评测与排名等 |
| [**CANN 大赛专区**](https://competition.gitcode.com/competition?type=cann) | 算子天梯赛 / 校园赛等各种大赛 | [大赛报名入口](https://competition.gitcode.com/competition?type=cann) → 报名与赛程<br/>[skills 目录](./skills) → 竞赛提交技能与算子工程生成 | 获奖权益、人才库推荐等 |

**备赛建议路径：**
1. **前置学习**：完成阶段一（公共基础）+ 阶段二
2. **专项练习**：在 [CANNJudge](https://cannjudge.cn) 按考点分类刷题
3. **高阶提升**：学习 [Ascend C 算子开发系列（算子工程版）](./tutorials/ascendc_operator_development)
4. **赛前集训**：关注大赛专区公告，参加赛前培训直播
5. **参赛提交**：使用 [skills/cannjudge-submit](./skills) 快速提交

</details>

---

<a id="certification"></a>
<details>
<summary><b>📜 等级认证学习路径（建设中）</b></summary>

> 对应 AI Infra 工程师初/中/高三级认证体系。通识课程 + 场景方向（推理 / 训练 / 推荐 / 应用开发 / 算子 5 选 1），明确"考什么、学什么"。

</details>

---

<a id="swanlab"></a>
<details>
<summary><b>🤝 SwanLab 共建案例</b></summary>

> 与 [SwanLab](https://swanlab.cn) 合作共建的大模型训练实战案例，正在从 SwanLab 平台迁移至本仓 `contrib/tutorials/swanlab_examples/` 目录下。

**适合人群**：想在 NPU 上做模型微调的开发者 · 关注训练过程可视化的用户 · 医学/语音/多模态领域开发者

**特色**：每个案例均配套 SwanLab 训练可视化（loss/metrics 曲线、实验对比），迁移完成后可在 CANNLab 环境直接运行。

| 类别 | 案例 | 简介 | 原始文档 | 状态 |
| :--- | :--- | :--- | :--- | :---: |
| 深度学习入门 | MNIST 手写数字识别 | NPU 上 CNN 手写数字识别入门 | [查看](./contrib/tutorials/swanlab_examples/mnist) | ✅ 已上线 |
| 监督微调 | 数学解题模型微调 | Qwen2.5-0.5B LoRA 数学解题微调实战 | [查看](./contrib/tutorials/swanlab_examples/math_solver_qwen2.5_lora) | ✅ 已上线 |
| 监督微调 | 医学模型微调 | Qwen3 医学领域 SFT 实战 | [查看](./contrib/tutorials/swanlab_examples/qwen3_medical_sft) · [原文](https://docs.swanlab.cn/course/llm_train_course/03-sft/4.qwen3-medical-finetune/) | ✅ 已上线 |
| 监督微调 | 其他框架微调——ms-swift | 使用 ms-swift 框架进行微调 | [文档](https://docs.swanlab.cn/course/llm_train_course/03-sft/8.other_frameworks/ms-swift.html) | 🚧 建设中 |
| 多模态 | Qwen3-smolVLM 模型拼接微调 | 多模态模型拼接微调实战 | [文档](https://docs.swanlab.cn/course/llm_train_course/06-multillm/2.qwen3_smolvlm_muxi/) | 🚧 建设中 |
| 音频 | CosyVoice 微调派蒙语音 | 语音模型微调实战 | [文档](https://docs.swanlab.cn/course/llm_train_course/07-audio/1.cosyvoice-sft/) | 🚧 建设中 |

> 💡 案例持续迁移中，完成后将在 CANNLab 环境提供在线体验。

</details>

---

## 🔗 配套生态

| | 平台 | 说明 |
| :--- | :--- | :--- |
| 📖 **学** | **cann-learning-hub**（本仓） | 系列教程 + 快速上手 + 参考实践 + 技术博客 |
| 🏋️ **练** | **[CANNJudge](https://cannjudge.cn)** | 开放题库，Ascend C 算子编程在线刷题，实时评测 |
| 🔬 **练** | **[CANNLab](https://gitcode.com/org/cann/cannlab)** | 任意 CANN 代码仓一键启动 NPU 环境（初始 100 小时，积分可兑换时长） |
| 🏆 **赛** | **[CANN 大赛专区](https://competition.gitcode.com/competition?type=cann)** | 官方/社区大赛报名入口 |

---

## 📝 技术博客

> CANN 在实际业务场景中的最新技术实践与成果。大部分为真实客户实践案例，持续更新中。

**近期精选：**

| 博客 | 简介 | 时间 |
| :--- | :--- | :---: |
| [Overlap Scheduling 吞吐优化](./blogs/inference/overlap_scheduling_throughput_optimization) | CPU 与 NPU 执行重叠，TPS 提升约 70% | 2026.3 |
| [AReaL 全异步 RL 训练](./blogs/training/areal_async_rl_training) | 全异步 RL + Single Controller，解耦式 Agentic RL | 2026.3 |
| [AICPU 点对点通信算子开发](./blogs/operator/hccl_custom_operator_aicpu_p2p) | 基于 AICPU+TS 实现 HCCL 自定义 Send/Recv 算子 | 2026.2 |
| [npugraph_ex 第三方框架集成](./blogs/inference/npugraph_ex_third_party_framework_integration) | 图编译与编译缓存能力接入，降低冷启动耗时 | 2026.2 |

<details>
<summary><b>查看全部技术博客</b>（算子 23 / 推理 23 / 训练 3，共 49 篇）</summary>

<table style="table-layout:fixed;width:100%">
<tr><th width="35%">博客</th><th width="55%">简介</th><th width="10%">时间</th></tr>
<tr><th colspan="3" align="left">算子</th></tr>
<tr><td><a href="./blogs/operator/ascend950_aiv_urma_shmem_communication">昇腾950 AIV 直驱 URMA 的 SHMEM 跨 PE 通信实践</a></td><td>AIV 在 Device 侧构造 WQE，借助 URMA 与 SHMEM 完成跨 PE 远端写和通知，减少 Host 往返</td><td>2026.7</td></tr>
<tr><td><a href="./blogs/operator/multi_vendor_operator_parallel_compilation">多 Vendor 自定义算子并行编译实践</a></td><td>统一调度多个独立 Vendor 子工程，并行完成算子编译、打包和产物隔离</td><td>2026.7</td></tr>
<tr><td><a href="./blogs/operator/tilelang_xllm_qwen35_operator_adaptation">TileLang 与 xLLM 驱动 Qwen3.5 昇腾算子适配</a></td><td>以 TileLang Python DSL 为开发入口，结合 xLLM 完成 Qwen3.5 算子适配</td><td>2026.7</td></tr>
<tr><td><a href="./blogs/operator/dsa_operator_open_source_contribution">DSA 算子开源共建之路</a></td><td>围绕 DSA 长序列训练算子 FP32 支持，从需求到反向贡献 ops-transformer 社区</td><td>2026.6</td></tr>
<tr><td><a href="./blogs/operator/ascend950_rmsnormquant_optimization">Ascend 950 RmsNormQuant 算子分阶段优化</a></td><td>Gamma 预加载、多核并行、寄存器数据流、Double Buffer 与二分累加优化</td><td>2026.5</td></tr>
<tr><td><a href="./blogs/operator/moe_dispatch_combine_optimization">MoE Dispatch 与 Combine 算子优化</a></td><td>跨 Rank 通信去重、本地加权合并、AIV 直驱 RDMA 与多流并行</td><td>2026.5</td></tr>
<tr><td><a href="./blogs/operator/mx_quantized_matmul_optimization">MX 量化矩阵乘性能优化</a></td><td>结合 SWAT、尾轮负载均衡和 UnitFlag 提升流水并行效率</td><td>2026.5</td></tr>
<tr><td><a href="./blogs/operator/scalar_npu_operator_performance_optimization">Scalar 对 NPU 算子性能的影响与优化</a></td><td>从寄存器 Spill、I-Cache、指针解引用等角度分析 ScalarBound 并给出优化方法</td><td>2026.5</td></tr>
<tr><td><a href="./blogs/operator/tilelang_ascend_operator_optimization">TileLang-Ascend 算子性能优化</a></td><td>流水级数、核间同步、数据切分和调试分析方面的优化方法</td><td>2026.5</td></tr>
<tr><td><a href="./blogs/operator/dumptensor_operator_debugging">DumpTensor 定位算子计算异常</a></td><td>利用 DumpTensor 观察 GM、UB、L1 中间数据，逐步定位结果异常</td><td>2026.4</td></tr>
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
<tr><td><a href="./blogs/operator/ms_sanitizer">msSanitizer 异常检测工具</a></td><td>单算子异常检测，定位内存访问、数据竞争与同步问题</td><td>2025.10</td></tr>
<tr><td><a href="./blogs/operator/nddma_introduction">NDDMA 多维数据搬运</a></td><td>多维 DMA 搬运与 Padding、Transpose、Broadcast、Slice 变换</td><td>2025.10</td></tr>
<tr><td><a href="./blogs/operator/regbase_vec_add">Regbase 编程范式</a></td><td>从向量加法理解寄存器级编程与底层性能优化</td><td>2025.10</td></tr>
<tr><th colspan="3" align="left">推理</th></tr>
<tr><td><a href="./blogs/inference/aot_superkernel_graph_execution">AOT SuperKernel 图执行优化</a></td><td>融合任务并减少启动、调度等待和算子间流水开销</td><td>2026.7</td></tr>
<tr><td><a href="./blogs/inference/longcat_flash_lite_inference_optimization">LongCat-Flash-Lite 昇腾推理优化</a></td><td>N-gram Embedding 与 EAGLE3 投机解码，结合 SuperKernel 提升推理效率</td><td>2026.6</td></tr>
<tr><td><a href="./blogs/inference/spark_model_cluster_throughput_optimization">星火大模型昇腾集群吞吐优化</a></td><td>CANN DSA 算子将长上下文 Attention 动态稀疏化，128K 场景约 4.5 倍 Decode 提升</td><td>2026.6</td></tr>
<tr><td><a href="./blogs/inference/tensorflow_autofuse_recommendation_fusion">TensorFlow 与 AutoFuse 推荐模型融合</a></td><td>打通 TensorFlow 到 Ascend IR 与 AutoFuse 自动融合链路</td><td>2026.5</td></tr>
<tr><td><a href="./blogs/inference/autofuse_torchinductor_deepseek_fusion">AutoFuse 与 TorchInductor 的 DeepSeek 融合</a></td><td>扩展 TorchInductor NPU Codegen，将 Loop IR 转 Ascend IR 接入 AutoFuse</td><td>2026.4</td></tr>
<tr><td><a href="./blogs/inference/deepseek_v4_npu_inference_optimization">NPU DeepSeek-V4 推理优化</a></td><td>稀疏 Attention 与 mHC 结构，结合融合算子、上下文并行、量化与多流并行</td><td>2026.4</td></tr>
<tr><td><a href="./blogs/inference/deepseek_v4_supernode_support">DeepSeek V4 昇腾超节点支持</a></td><td>昇腾950 与 A3 超节点适配，含融合 Kernel、多流并行、量化与自动融合</td><td>2026.4</td></tr>
<tr><td><a href="./blogs/inference/sals_long_sequence_inference">SALS 长序列推理优化</a></td><td>稀疏 Token 选择、SFAA 计算、离散访存、Preload 流水和 AICPU Tiling 优化</td><td>2026.4</td></tr>
<tr><td><a href="./blogs/inference/xllm_inference_performance_optimization">xLLM 大模型推理性能优化</a></td><td>图融合、多流并行、投机推理和动态负载均衡优化 Qwen/DeepSeek/ChatGLM</td><td>2026.4</td></tr>
<tr><td><a href="./blogs/inference/hixl_nixl_ascend_backend">HIXL 快速适配 NIXL 昇腾后端</a></td><td>基于 NIXL 插件架构将 HIXL 点对点通信映射为昇腾后端</td><td>2026.3</td></tr>
<tr><td><a href="./blogs/inference/overlap_scheduling_throughput_optimization">Overlap Scheduling 吞吐优化</a></td><td>CPU 与 NPU 执行重叠，TPS 提升约 70%</td><td>2026.3</td></tr>
<tr><td><a href="./blogs/inference/hixl_fabricmem_kv_cache_transfer">HIXL FabricMem 高性能 KV Cache 传输</a></td><td>超节点 DRAM 统一编址与 HCCS/SDMA 单边传输，提供高带宽 FabricMem 通道</td><td>2026.2</td></tr>
<tr><td><a href="./blogs/inference/npugraph_ex_third_party_framework_integration">npugraph_ex 第三方框架集成</a></td><td>图编译与编译缓存能力接入，降低冷启动耗时</td><td>2026.2</td></tr>
<tr><td><a href="./blogs/inference/deepseek_r1_superpod_inference_optimization">DeepSeek-R1 SuperPoD 推理优化</a></td><td>全栈协同，TTFT<2s、TPOT<50ms，608 QPM</td><td>2025.12</td></tr>
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

## 🔬 科研案例

> 高校教师贡献的科研实践案例，聚焦昇腾 CANN 在真实科研场景中的落地，促进高校科研经验与开源生态的交流共享。案例持续征集中，欢迎高校教师投稿。

| 案例名称 | 作者 / 单位 | 案例介绍 | 发布时间 |
| :--- | :--- | :--- | :---: |
| _（待贡献）_ | — | — | — |

> 💡 投稿方式、案例模板与署名要求见 [contrib/blogs/research_cases](./contrib/blogs/research_cases)。

---

## 🤝 参与贡献

欢迎贡献教程、文档与案例！请阅读 [贡献指南](CONTRIBUTING.md)，并前往 [cann/community](https://gitcode.com/cann/community) 了解社区行为准则与 CLA 签署流程。

| 渠道 | 入口 |
| :--- | :--- |
| 提问 / 反馈 Bug | [Issues](https://gitcode.com/cann/cann-learning-hub/issues) |
| 交流讨论 | [Discussions](https://gitcode.com/cann/cann-learning-hub/discussions) |
| 技术专栏 | [Wiki](https://gitcode.com/cann/cann-learning-hub/wiki) |

---

## 🔥 Latest News

- **[2026/09]** 新增[大模型 RL 训练系列教程](./tutorials/rl_training_pipeline)，使用 verl + GRPO 完成 Qwen3-1.7B Wordle 强化学习训练
- **[2026/09]** 新增[面向高性能计算的数据结构](./contrib/tutorials/data_structure_for_hpc)课程
- **[2026/08]** 新增 [PyPTO](./tutorials/pypto_development)、[PyASC](./tutorials/pyasc_operator_development)、[CANNBot](./tutorials/CANNBot/README.md)、[TorchAir](./tutorials/TorchAir_development)、[GE](./tutorials/ge_development)、[AutoFusion](./tutorials/autofusion_development)、[HCCL](./tutorials/hccl_development) 系列教程
- **[2026/08]** 新增[大模型训练系列课程](./tutorials/sft_training_pipeline)（SFT 初阶）
- **[2026/07]** 新增 [Ascend C 算子开发（Kernel 直调版）](./tutorials/ascendc_operator_development_light)、[大模型推理系列](./tutorials/llm_inference)、[Conv 算子实战](./tutorials/conv_operator_development)
- **[2026/06]** 在线体验适配 CANN 9.0.0
- **[2026/03]** cann-learning-hub 首次上线

---

## 📄 版本与信息

**已验证版本配套**

| CANN 版本 | 验证日期 |
| :--- | :---: |
| 9.0.0 | 2026.06.30 |

- [贡献指南](CONTRIBUTING.md) · [安全声明](SECURITY.md) · [许可证](LICENSE) · [所属 SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/doc)
- 问题反馈：[Issues](https://gitcode.com/cann/cann-learning-hub/issues) · 社区互动：[Discussions](https://gitcode.com/cann/cann-learning-hub/discussions) · 技术专栏：[Wiki](https://gitcode.com/cann/cann-learning-hub/wiki)

---

<details>
<summary><b>查看目录结构</b></summary>

```
├── quick_start                  # 快速入门
│   ├── cann_basics              # CANN 基础知识（5 门课）
│   ├── first_custom_operator    # 第一个自定义算子
│   ├── first_operator_api_call  # 第一个算子 API 调用
│   ├── first_llm_inference      # 第一个大模型推理
│   └── git_basics               # Git 基础与 GitCode 协作
├── tutorials                    # 开发教程
│   ├── ascendc_operator_development          # Ascend C 算子开发（工程版）
│   ├── ascendc_operator_development_light    # Ascend C 算子开发（Kernel 直调版）
│   ├── conv_operator_development             # Conv 算子开发
│   ├── MC2_fused_operator_development        # MC2 融合算子
│   ├── llm_inference                         # 大模型推理
│   ├── sft_training_pipeline                 # SFT 训练
│   ├── rl_training_pipeline                  # RL 训练
│   ├── ge_development                        # GE 图引擎
│   ├── TorchAir_development                  # TorchAir 图模式
│   ├── autofusion_development                # AutoFusion 自动融合
│   ├── hccl_development                      # HCCL 集合通信
│   ├── hixl_development                      # HiXL 单边通信
│   ├── CANNBot                               # CANNBot 算子生成
│   ├── pyasc_operator_development            # PyASC 算子开发
│   └── pypto_development                     # PyPTO 算子开发
├── standard_course               # 标准课程体系
│   ├── 00_atomic_courses         # 原子课程总纲（135 门）
│   ├── 01_bootcamp               # 启航营
│   ├── 02_traditional_courses    # 传统学科课程
│   ├── 03_new_courses            # 新增课程
│   └── 04_new_books              # 教材区
├── reference_practice            # 参考实践
├── blogs                         # 技术博客（算子 23 / 推理 23 / 训练 3）
├── contrib                       # 社区贡献
│   ├── tutorials                 # 外部教程（15+）
│   └── blogs/research_cases      # 科研案例（高校教师投稿，含模板）
├── skills                        # CANNBot 技能
└── docs                          # 文档与指南
```

</details>
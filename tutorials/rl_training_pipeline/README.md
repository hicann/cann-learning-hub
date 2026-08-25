# Qwen3-1.7B Wordle 强化学习实战

本课程面向 Ascend NPU 开发者，使用 verl + GRPO 完成 Wordle 多轮强化学习训练。第 1～4 章从环境准备和 RL 核心概念讲起，随后实现 Wordle Agent Loop、奖励函数、数据准备、训练监控与稳定性分析；第 5～8 章进一步将 Actor 与 Reference Model 的训练后端切换为 TorchTitan-NPU，并介绍 FSDP2、offload、TND 变长注意力和上下文并行。

模型使用已完成 SFT 的 `Qwen3-1.7B-Wordle-SFT`，规范猜词格式为 `<guess>[word]</guess>`。配套训练代码位于 `cann-recipes-train/llm_rl/qwen3_wordle/`。

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上 |
| Python | 3.11 |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 环境要求 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 云开发环境 | 已安装 CANN 和 ATB | Python 3.11 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建 CANNLab 环境运行 Notebook |

本课程从 `cann-learning-hub` 课程仓进入 CANNLab，课程 notebook 已随仓库提供，无须再次克隆课程仓。首次进入后，请打开 [01.01 章节介绍](01_environment_setup/01.01_chapter_intro.ipynb)，运行其中的仓库拉取单元格，将 `cann-recipes-train` 克隆到课程仓的同级目录。环境安装、数据准备和检查由 notebook 单元格完成，长时间训练在终端运行；TensorBoard 日志由训练脚本自动生成，并按 03.04 节说明复制到本地查看。

```text
/mnt/workspace/gitCode/cann/
├── cann-learning-hub/                 # CANNLab 入口及默认工作目录
│   └── tutorials/rl_training_pipeline/ # 本课程
└── cann-recipes-train/                # 01.01 单元格拉取的训练代码仓
    └── llm_rl/qwen3_wordle/           # 配套训练代码
```

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/index.html)。\
> Notebook 用于阅读教程和章节练习。训练任务需在配备 Ascend NPU 的服务器上独立运行。

## 课程内容

| 序号 | 主题 | 主要内容 | 课件 |
|---|---|---|---|
| 01 | RL 强化学习 | verl/vLLM-Ascend 环境、RL 与 GRPO 原理、Wordle AgentLoop、奖励设计、训练指标与稳定性调优 | [01_rl_training_pipeline.pptx](slides/01_rl_training_pipeline.pptx) |


## 教程结构

### 初阶课程（第 1～4 章）

#### 第 1 章：强化学习训练环境准备

| Notebook | 内容 | 状态 |
|---|---|---|
| [01.01 章节介绍](01_environment_setup/01.01_chapter_intro.ipynb) | CANNLab 目录布局、学习目标、SFT 与 RL 的区别 | ✅ 已发布 |
| [01.02 安装 verl 与 vLLM-Ascend](01_environment_setup/01.02_install_verl_and_vllm_ascend.ipynb) | 安装依赖、应用 Wordle Agent Loop patch、验证环境 | ✅ 已发布 |
| [01.03 verl 框架概览](01_environment_setup/01.03_verl_framework_overview.ipynb) | Agent Loop、rollout、FSDP、vLLM 与资源切换 | ✅ 已发布 |
| [01.04 章节练习](01_environment_setup/01.04_chapter_practice.ipynb) | 选择题与判断题 | ✅ 已发布 |

#### 第 2 章：RL 核心概念

| Notebook | 内容 | 状态 |
|---|---|---|
| [02.01 章节介绍](02_rl_core_concepts/02.01_chapter_intro.ipynb) | 本章目标与内容导航 | ✅ 已发布 |
| [02.02 策略与奖励](02_rl_core_concepts/02.02_policy_and_reward.ipynb) | Policy、Reward、Advantage 与奖励塑形 | ✅ 已发布 |
| [02.03 PPO 与 GRPO](02_rl_core_concepts/02.03_ppo_and_grpo.ipynb) | GRPO 分组采样、优势归一化和策略更新 | ✅ 已发布 |
| [02.04 KL 与训练稳定性](02_rl_core_concepts/02.04_kl_and_stability.ipynb) | KL、Entropy Bonus 与策略崩塌 | ✅ 已发布 |
| [02.05 章节练习](02_rl_core_concepts/02.05_chapter_practice.ipynb) | 选择题与判断题 | ✅ 已发布 |

#### 第 3 章：Wordle RL 训练

| Notebook | 内容 | 状态 |
|---|---|---|
| [03.01 章节介绍](03_wordle_rl_training/03.01_chapter_intro.ipynb) | Wordle 任务与训练目标 | ✅ 已发布 |
| [03.02 环境与 Agent Loop](03_wordle_rl_training/03.02_wordle_env_and_agent_loop.ipynb) | 多轮交互、G/Y/X 反馈和 token mask | ✅ 已发布 |
| [03.03 奖励函数与数据](03_wordle_rl_training/03.03_reward_and_data.ipynb) | 四项奖励、Prime-RL 差异和 parquet 数据生成 | ✅ 已发布 |
| [03.04 运行训练与指标](03_wordle_rl_training/03.04_run_training_and_metrics.ipynb) | 终端启动训练、日志解读与 TensorBoard | ✅ 已发布 |
| [03.05 章节练习](03_wordle_rl_training/03.05_chapter_practice.ipynb) | 选择题与判断题 | ✅ 已发布 |

#### 第 4 章：调优与问题排查

| Notebook | 内容 | 状态 |
|---|---|---|
| [04.01 章节介绍](04_tuning_and_troubleshooting/04.01_chapter_intro.ipynb) | 调优目标与诊断方法 | ✅ 已发布 |
| [04.02 超参数调优](04_tuning_and_troubleshooting/04.02_hyperparameter_tuning.ipynb) | entropy、KL 和学习率 | ✅ 已发布 |
| [04.03 训练崩塌分析](04_tuning_and_troubleshooting/04.03_training_collapse_analysis.ipynb) | 异常指标、根因定位与修复 | ✅ 已发布 |
| [04.04 章节练习](04_tuning_and_troubleshooting/04.04_chapter_practice.ipynb) | 选择题与判断题 | ✅ 已发布 |

### 中阶课程（第 5～8 章）

#### 第 5 章：从 FSDP 到 TorchTitan-NPU FSDP2

| Notebook | 内容 | 状态 |
|---|---|---|
| 05.01 章节介绍 | 学习目标、前置条件与后端切换范围 | 🚧 建设中 |
| 05.02 训练后端切换原理 | 配置映射、模块调用链和 Actor 到 vLLM 的权重同步 | 🚧 建设中 |
| 05.03 章节练习 | 后端职责、调用链和权重同步练习 | 🚧 建设中 |

#### 第 6 章：TorchTitan-NPU 核心特性

| Notebook | 内容 | 状态 |
|---|---|---|
| 06.01 章节介绍 | TorchTitan 与 TorchTitan-NPU 的定位 | 🚧 建设中 |
| 06.02 FSDP2 与可组合并行 | DeviceMesh、两卡 FSDP2 与 CP 长序列扩展 | 🚧 建设中 |
| 06.03 Wordle 训练使用的 TorchTitan-NPU 特性 | offload、TND 变长注意力、NPU converter 和权重同步 | 🚧 建设中 |
| 06.04 章节练习 | FSDP2、TND、内存策略、NPU converter 和序列长度预算练习 | 🚧 建设中 |

#### 第 7 章：Wordle 训练后端切换实践

第 7 章需要单机两张 Ascend NPU。环境准备 Cell 会在同级目录获取或复用 `cann-recipes-train`，使用 uv 创建独立的 `.venv`，并复用或补齐 SFT 模型与 Wordle parquet。三步训练沿用初阶课程的模型、数据、batch、序列长度、rollout、AgentLoop、GRPO 和奖励函数配置。

| Notebook | 内容 | 状态 |
|---|---|---|
| 07.01 章节介绍 | 实践目标、训练资源和运行要求 | 🚧 建设中 |
| 07.02 准备运行环境与训练资产 | 安装独立后端环境，并准备 SFT 模型与 Wordle parquet | 🚧 建设中 |
| 07.03 确认训练配置 | 使用 DRY_RUN 核对 TorchTitan、FSDP2、TND 和原有 RL 配置 | 🚧 建设中 |
| 07.04 运行三步训练 | 以 FSDP2 + TND 连续完成 3 个训练 step | 🚧 建设中 |
| 07.05 章节练习 | 环境、配置和三步训练练习 | 🚧 建设中 |

#### 第 8 章：训练后端切换总结

| Notebook | 内容 | 状态 |
|---|---|---|
| 08.01 章节介绍 | 学习目标和内容安排 | 🚧 建设中 |
| 08.02 后端切换总结 | 切换步骤、常见问题和性能指标 | 🚧 建设中 |
| 08.03 综合练习 | 配置、特性和实践综合练习 | 🚧 建设中 |

## 参考

- [Prime-RL Wordle 示例](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/basic/wordle)
- [Verifiers Wordle 奖励源码（课程参考版本）](https://github.com/PrimeIntellect-ai/verifiers/blob/8d4b332477aea4a34bbf9fb821e3e3bc8b0e2e74/environments/wordle/wordle.py)
- [verl](https://github.com/volcengine/verl)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [TorchTitan-NPU](https://gitcode.com/cann/torchtitan-npu)
- [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train)

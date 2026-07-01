# Wordle 两阶段训练教程

本系列教程教授 Ascend NPU 开发者完成 Wordle 猜词任务的两阶段训练：**SFT（监督微调）→ RL（强化学习）**。


- **SFT 阶段**（`01-sft`）：使用 `torchtitan-npu` 框架，让 Qwen3-1.7B 学会按 `<guess>word</guess>` 格式输出，理解 G/Y/X 反馈规则。
- **RL 阶段**（`02-rl`）：使用 `verl` 框架 + GRPO 算法，让模型学会策略性猜词，提高 6 轮内猜中率。


## 前置条件

- 了解 PyTorch 模型训练的基本流程（forward → loss → backward → update）。
- 了解 Transformer 模型的基本结构（attention、FFN、layer norm）。

## 环境准备

1. 在 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) 点击 CANNLab 创建开发环境。
![CANNLab Environment](images/cann-lab-env.png)

2. 镜像选择：`cann_9.0.0-A3`。

3. Notebook 用于阅读教程和章节练习。训练任务需在配备 Ascend NPU 的服务器上独立运行。


## 教程结构

### 01-sft：Wordle SFT 监督微调

#### 第 1 章：SFT 概念与 Wordle 任务 (`01-sft/01_sft_and_wordle/`)

| Notebook | 内容 |
|---|---|
| [01.01 章节介绍](01-sft/01_sft_and_wordle/01.01_chapter_intro.ipynb) | 两阶段训练方案（SFT → RL），本章目标 |
| [01.02 Wordle 任务介绍](01-sft/01_sft_and_wordle/01.02_wordle_task_intro.ipynb) | 游戏规则、为什么选 Wordle、评测指标、数据格式 |
| [01.03 SFT 概念与原理](01-sft/01_sft_and_wordle/01.03_sft_concepts.ipynb) | SFT vs 预训练、为什么 Wordle 需要 SFT、三阶段对比 |
| [01.04 章节练习](01-sft/01_sft_and_wordle/01.04_chapter_practice.ipynb) | 选择题 + 判断题 |

#### 第 2 章：TorchTitan 框架与环境配置 (`01-sft/02_torchtitan_framework/`)

| Notebook | 内容 |
|---|---|
| [02.01 章节介绍](01-sft/02_torchtitan_framework/02.01_chapter_intro.ipynb) | 教程进度回顾、本章结构、前置条件、预期成果 |
| [02.02 FSDP 原理与 TorchTitan 框架](01-sft/02_torchtitan_framework/02.02_torchtitan_and_fsdp.ipynb) | FSDP 原理（正向/反向/FSDP2）、TorchTitan 启动流程、model_registry |
| [02.03 环境配置](01-sft/02_torchtitan_framework/02.03_environment_setup.ipynb) | Clone 仓库、安装依赖、Debug 验证 |
| [02.04 章节练习](01-sft/02_torchtitan_framework/02.04_chapter_practice.ipynb) | 选择题 + 判断题 |

#### 第 3 章：训练配置与基线训练 (`01-sft/03_sft_training/`)

| Notebook | 内容 |
|---|---|
| [03.01 章节介绍](01-sft/03_sft_training/03.01_chapter_intro.ipynb) | 三步闭环（配置 → 训练 → 评测），预期成果 |
| [03.02 训练配置构建](01-sft/03_sft_training/03.02_training_preparation.ipynb) | 数据管线、训练配置、学习率调度、多卡并行、activation checkpoint |
| [03.03 运行基线 SFT](01-sft/03_sft_training/03.03_run_baseline_sft.ipynb) | 训练命令、日志解读、Loss 曲线 |
| [03.04 推理验证](01-sft/03_sft_training/03.04_inference_eval.ipynb) | infer_server 推理、vf-eval 评测、SFT 前后对比 |
| [03.05 章节练习](01-sft/03_sft_training/03.05_chapter_practice.ipynb) | 选择题 + 判断题 |

#### 第 4 章：融合算子优化与 Profiling (`01-sft/04_fused_operators/`)

| Notebook | 内容 |
|---|---|
| [04.01 章节介绍](01-sft/04_fused_operators/04.01_chapter_intro.ipynb) | 基线瓶颈分析、融合算子概述、预期收益 |
| [04.02 融合算子原理与接入](01-sft/04_fused_operators/04.02_adding_fused_operator.ipynb) | torch_npu API、融合代码、ModelConverter 全链路 |
| [04.03 Profiling 实操](01-sft/04_fused_operators/04.03_running_and_profiling.ipynb) | 跑 baseline + fused profiling、分析脚本、验证融合生效 |
| [04.04 Profiling 结果分析](01-sft/04_fused_operators/04.04_profiling_output_analysis.ipynb) | 性能对比图、收益分解、算子级分析 |
| [04.05 章节练习](01-sft/04_fused_operators/04.05_chapter_practice.ipynb) | 选择题 + 判断题 |

### 02-rl：Wordle RL 强化学习

#### 第 1 章：环境准备 (`02-rl/01_environment_setup/`)

| Notebook | 内容 |
|---|---|
| [1.1 章节介绍](02-rl/01_environment_setup/01.01_chapter_intro.ipynb) | 章节概述与目标 |
| [1.2 安装 verl + vllm-ascend](02-rl/01_environment_setup/01.02_install_verl_and_vllm_ascend.ipynb) | 安装依赖、环境配置 |
| [1.3 verl 框架架构速览](02-rl/01_environment_setup/01.03_verl_framework_overview.ipynb) | verl 框架架构与核心组件 |
| [1.4 章节练习](02-rl/01_environment_setup/01.04_chapter_practice.ipynb) | 选择题 + 判断题 |

#### 第 2 章：RL 核心概念 (`02-rl/02_rl_core_concepts/`)

| Notebook | 内容 |
|---|---|
| [2.1 章节介绍](02-rl/02_rl_core_concepts/02.01_chapter_intro.ipynb) | 章节概述与目标 |
| [2.2 策略与奖励](02-rl/02_rl_core_concepts/02.02_policy_and_reward.ipynb) | 策略网络、奖励信号设计 |
| [2.3 PPO 与 GRPO](02-rl/02_rl_core_concepts/02.03_ppo_and_grpo.ipynb) | PPO vs GRPO 算法对比与原理 |
| [2.4 KL 散度与训练稳定性](02-rl/02_rl_core_concepts/02.04_kl_and_stability.ipynb) | KL 散度的作用、训练稳定性分析 |
| [2.5 章节练习](02-rl/02_rl_core_concepts/02.05_chapter_practice.ipynb) | 选择题 + 判断题 |

#### 第 3 章：Wordle RL 训练实战 (`02-rl/03_wordle_rl_training/`)

| Notebook | 内容 |
|---|---|
| [3.1 章节介绍](02-rl/03_wordle_rl_training/03.01_chapter_intro.ipynb) | 章节概述与目标 |
| [3.2 Wordle 游戏与 AgentLoop 设计](02-rl/03_wordle_rl_training/03.02_wordle_env_and_agent_loop.ipynb) | Wordle 环境、AgentLoop 设计 |
| [3.3 奖励函数与数据准备](02-rl/03_wordle_rl_training/03.03_reward_and_data.ipynb) | 奖励函数设计、训练数据准备 |
| [3.4 运行训练与指标解读](02-rl/03_wordle_rl_training/03.04_run_training_and_metrics.ipynb) | 训练命令、指标解读 |
| [3.5 章节练习](02-rl/03_wordle_rl_training/03.05_chapter_practice.ipynb) | 选择题 + 判断题 |

#### 第 4 章：训练调优与问题排查 (`02-rl/04_tuning_and_troubleshooting/`)

| Notebook | 内容 |
|---|---|
| [4.1 章节介绍](02-rl/04_tuning_and_troubleshooting/04.01_chapter_intro.ipynb) | 章节概述与目标 |
| [4.2 超参数调优](02-rl/04_tuning_and_troubleshooting/04.02_hyperparameter_tuning.ipynb) | 超参数调优策略与实践 |
| [4.3 训练崩溃分析](02-rl/04_tuning_and_troubleshooting/04.03_training_collapse_analysis.ipynb) | 训练崩溃诊断与修复 |
| [4.4 章节练习](02-rl/04_tuning_and_troubleshooting/04.04_chapter_practice.ipynb) | 选择题 + 判断题 |



## 参考链接

- *本教程的任务设计参考了 [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)的 [Wordle 示例](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/wordle/README.md)，包括多轮猜词格式、环境反馈机制和训练流程。*
- [torchtitan-npu 仓库](https://gitcode.com/mystri/torchtitan-npu/tree/teaching-pipeline)
- [torchtitan-npu 安装指南](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/user-guides/installation.md)
- [SFT Recipe 附录](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/recipe/sft.md)
- [verl 框架](https://github.com/volcengine/verl)

# Qwen3-1.7B Wordle 强化学习实战

本课程面向 Ascend NPU 开发者，使用 verl + GRPO 完成 Wordle 多轮强化学习训练。课程从环境准备和 RL 核心概念讲起，随后实现 Wordle Agent Loop、奖励函数、数据准备、训练监控与稳定性分析。

模型使用已完成 SFT 的 `Qwen3-1.7B-Wordle-SFT`，规范猜词格式为 `<guess>[word]</guess>`。配套训练代码位于 `cann-recipes-train/llm_rl/qwen3_wordle/`。

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上 |
| Python | 3.11 |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| CANNLab 云开发环境 | cann_9.0.0 py3.11-A3-arm | Python 3.11.4 |参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)创建CANNLab环境运行notebook |

本课程从 `cann-learning-hub` 课程仓进入 CANNLab，课程 notebook 已随仓库提供，无须再次克隆课程仓。首次进入后，请打开 [01.01 章节介绍](01_environment_setup/01.01_chapter_intro.ipynb)，运行其中的仓库拉取单元格，将 `cann-recipes-train` 克隆到课程仓的同级目录。环境安装、数据准备、检查和 TensorBoard 启动由 notebook 单元格完成，长时间训练在终端运行。

```text
/mnt/workspace/gitCode/cann/
├── cann-learning-hub/                 # CANNLab 入口及默认工作目录
│   └── tutorials/rl_training_pipeline/ # 本课程
└── cann-recipes-train/                # 01.01 单元格拉取的训练代码仓
    └── llm_rl/qwen3_wordle/           # 配套训练代码
```

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)。\
> Notebook 用于阅读教程和章节练习。训练任务需在配备 Ascend NPU 的服务器上独立运行。

## 课程内容

| 序号 | 主题 | 主要内容 | 课件 |
|---|---|---|---|
| 01 | RL 强化学习 | verl/vLLM-Ascend 环境、RL 与 GRPO 原理、Wordle AgentLoop、奖励设计、训练指标与稳定性调优 | [01_rl_training_pipeline.pptx](https://gitcode.com/cann/cann-learning-hub/blob/test/tutorials/rl_training_pipeline/slides/01_rl_training_pipeline.pptx) |


## 教程结构

### 01-sft：Wordle SFT 监督微调

#### 第 1 章：SFT 概念与 Wordle 任务 (`01_sft_and_wordle/`)

| Notebook | 内容 |
|---|---|
| [01.01 章节介绍](01_environment_setup/01.01_chapter_intro.ipynb) | CANNLab 目录布局、学习目标、SFT 与 RL 的区别 |
| [01.02 安装 verl 与 vLLM-Ascend](01_environment_setup/01.02_install_verl_and_vllm_ascend.ipynb) | 安装依赖、应用 Wordle Agent Loop patch、验证环境 |
| [01.03 verl 框架概览](01_environment_setup/01.03_verl_framework_overview.ipynb) | Agent Loop、rollout、FSDP、vLLM 与资源切换 |
| [01.04 章节练习](01_environment_setup/01.04_chapter_practice.ipynb) | 选择题与判断题 |

### 第 2 章：RL 核心概念

| Notebook | 内容 |
|---|---|
| [02.01 章节介绍](02_rl_core_concepts/02.01_chapter_intro.ipynb) | 本章目标与内容导航 |
| [02.02 策略与奖励](02_rl_core_concepts/02.02_policy_and_reward.ipynb) | Policy、Reward、Advantage 与奖励塑形 |
| [02.03 PPO 与 GRPO](02_rl_core_concepts/02.03_ppo_and_grpo.ipynb) | GRPO 分组采样、优势归一化和策略更新 |
| [02.04 KL 与训练稳定性](02_rl_core_concepts/02.04_kl_and_stability.ipynb) | KL、Entropy Bonus 与策略崩塌 |
| [02.05 章节练习](02_rl_core_concepts/02.05_chapter_practice.ipynb) | 选择题与判断题 |

### 第 3 章：Wordle RL 训练

| Notebook | 内容 |
|---|---|
| [03.01 章节介绍](03_wordle_rl_training/03.01_chapter_intro.ipynb) | Wordle 任务与训练目标 |
| [03.02 环境与 Agent Loop](03_wordle_rl_training/03.02_wordle_env_and_agent_loop.ipynb) | 多轮交互、G/Y/X 反馈和 token mask |
| [03.03 奖励函数与数据](03_wordle_rl_training/03.03_reward_and_data.ipynb) | 四项奖励、Prime-RL 差异和 parquet 数据生成 |
| [03.04 运行训练与指标](03_wordle_rl_training/03.04_run_training_and_metrics.ipynb) | 终端启动训练、日志解读与 TensorBoard |
| [03.05 章节练习](03_wordle_rl_training/03.05_chapter_practice.ipynb) | 选择题与判断题 |

### 第 4 章：调优与问题排查

| Notebook | 内容 |
|---|---|
| [04.01 章节介绍](04_tuning_and_troubleshooting/04.01_chapter_intro.ipynb) | 调优目标与诊断方法 |
| [04.02 超参数调优](04_tuning_and_troubleshooting/04.02_hyperparameter_tuning.ipynb) | entropy、KL 和学习率 |
| [04.03 训练崩塌分析](04_tuning_and_troubleshooting/04.03_training_collapse_analysis.ipynb) | 异常指标、根因定位与修复 |
| [04.04 章节练习](04_tuning_and_troubleshooting/04.04_chapter_practice.ipynb) | 选择题与判断题 |

## 参考

- [Prime-RL Wordle 示例](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/wordle)
- [Verifiers Wordle 奖励源码](https://github.com/PrimeIntellect-ai/verifiers/blob/main/environments/wordle/wordle.py)
- [verl](https://github.com/volcengine/verl)
- [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train)

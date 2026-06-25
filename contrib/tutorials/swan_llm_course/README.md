<div align="center" style="border: 1px solid #eee; padding: 20px; border-radius: 10px; margin-bottom: 20px; max-width: 320px; margin-left: auto; margin-right: auto;">

![SwanLab](images/swanlab.png)

SwanLab x CANN 社区合作课程

</div>

---

# Swan LLM 大模型实战课程

## 课程作者与联系方式

- **作者**：韩翔宇（情感机器 SwanLab 实验室 AI 研究员）等
- **邮箱**：[pescn@115lab.club](mailto:pescn@115lab.club)
- **课程背景**：本课程内容来源于 CANN 社区与 SwanLab 团队正在开展的**线下启航营**，面向高校同学讲解昇腾算力上的大语言模型训练、对齐、推理与加速。如果你所在的学校 / 社团 / 实验室希望接入这套教材，或者希望在你们学校落地一期启航营，欢迎通过讨论区或邮件联系。

## 课程简介

本课程面向**高校在校学生**，结合正在开展的**线下启航营**实践内容，围绕大语言模型在昇腾 NPU 上的**基础理论 → 微调 → 强化学习 → 推理部署 → 性能调优**展开。我们希望通过这套课程，帮助同学们：

- 建立大模型从训练到部署的完整心智模型，而不是孤立地学某一个工具
- 真正理解微调（SFT / LoRA）、强化学习对齐（RLHF / DPO / GRPO）的工程实现细节
- 在昇腾 NPU + CANN 的真实硬件上完成训练 / 推理，体会国产算力栈下的工程权衡
- 通过 AscendC 自定义算子优化实战，理解大模型训练 / 推理性能瓶颈的来源及优化思路

## 适合人群

> 建议学习者具备 Python 基础和线性代数基础。但 **不要求** 事先有 Ascend C 经验

- 对大模型有兴趣，但还没真正跑通一次端到端训练 / 部署的本科生 / 研究生
- 已经在其他智算卡上跑过 transformers / trl，想了解昇腾 NPU 上有什么不一样的同学
- 想理解为什么大家都在写自定义算子，并亲手优化一次的同学

## 课程目录

| 章节 | 标题 | 内容概要 | 状态 |
|------|------|----------|------|
| [01](./01_llm_basic/) | 大语言模型基础理论介绍 | Transformer、自注意力、预训练 / SFT / RLHF 三段式、推理与 KV cache | 建设中 |
| [02](./02_llm_finetune/) | 大语言模型微调 | Qwen3 全参数 SFT、LoRA 微调、Loss Mask、AST + 可执行性评估 | 已发布部分节次 |
| [03](./03_llm_rl/) | 大语言模型强化学习 | RLHF / PPO、DPO、GRPO，以代码可执行性作为奖励信号的案例 | 建设中 |
| [04](./04_llm_inference_deploy/) | 大语言模型推理部署 | KV cache、continuous batching、量化、推理引擎与服务化 | 建设中 |
| [05](./05_performance_optimization/) | 性能调优 | AscendC 自定义算子接入 PyTorch、Amdahl 律、推理 / 训练端到端加速 | 已发布部分节次 |

### 已发布节次速览

| 节次 | 标题 |
|------|------|
| [02.04](./02_llm_finetune/02.04_qwen3_instruction_sft.ipynb) | Qwen3 基座模型指令微调（SFT） |
| [05.02](./05_performance_optimization/05.02_swan_rmsnorm_acceleration.ipynb) | SwanRmsNorm AscendC 算子加速 Qwen3 微调 |

## 目录与命名规范

每个章节目录的统一约定如下：

```
0X_<chapter_slug>/
├── README.md                    # 章节定位、节次清单、运行说明
├── 0X.01_<section_slug>.ipynb   # 节次 Notebook
├── 0X.02_<section_slug>.ipynb
├── data/                        # 节次配套数据集（可选）
├── pdf/                         # 节次配套 PDF（可选）
├── images/                      # 章节配图（可选）
└── answer/                      # 实操题参考答案（可选）
```

## 运行环境

| 项目 | 推荐配置 |
|------|----------|
| 硬件 | 昇腾 910C / 910B |
| 软件 | CANN ≥ 8.5、Python 3.10/3.11、PyTorch 2.x + `torch_npu` |
| 平台 | 推荐 [CANNLab](https://gitcode.com/) |
| 第三方 | SwanLab（实验可视化）、ModelScope（模型与数据集下载）、TRIO |

## 反馈与贡献

发现 Notebook 里的 bug、对某一章节有改进建议，或希望补充占位章节的内容，欢迎在 [cann/cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) 仓库提 Issue 或 PR。

---

> 如果这套课程帮到了你，欢迎给本课程项目点个 Star ⭐，分享给更多对大模型感兴趣的同学！

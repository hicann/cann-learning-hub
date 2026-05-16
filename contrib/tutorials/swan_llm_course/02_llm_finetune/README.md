<div align="center" style="border: 1px solid #eee; padding: 20px; border-radius: 10px; margin-bottom: 20px; max-width: 320px; margin-left: auto; margin-right: auto;">

![SwanLab](../images/swanlab.png)

SwanLab x CANN 社区合作课程

</div>

---

# 第 02 章 · 大语言模型微调

## 章节定位

本章围绕 Qwen3 系列模型，按"**原理 → 数据 / 模板 → SFT → LoRA → 工程实践**"的顺序，完整跑通一次端到端微调流程。
预期读者在学完本章后，能理解 Chat Template、Special Token、Loss Mask、LoRA 低秩适配、超参数选择、效果评估等工程细节，并能在昇腾 NPU 上独立完成一次面向自己业务数据的指令微调，而不只是停留在套用 LLaMA Factory 等工具。

## 计划节次

| 节次 | 标题 | 状态 |
|------|------|------|
| 02.01 | 章节介绍 | 建设中 |
| 02.02 | 微调与 LoRA 原理 | 建设中 |
| 02.03 | Chat Template 与 Special Token | 建设中 |
| [02.04](./02.04_qwen3_instruction_sft.ipynb) | Qwen3 基座模型指令微调（SFT） | 已发布 |
| 02.05 | Qwen3 LoRA 微调：量化因子代码生成实战 | 建设中 |
| 02.06 | 用 ms-swift 复现 SFT / LoRA 微调（成熟工具链对照） | 建设中 |
| 02.07 | 数据合成 | 建设中 |
| 02.08 | 微调超参数与科学调参 | 建设中 |
| 02.09 | 微调效果评估 | 建设中 |

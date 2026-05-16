<div align="center" style="border: 1px solid #eee; padding: 20px; border-radius: 10px; margin-bottom: 20px; max-width: 320px; margin-left: auto; margin-right: auto;">

![SwanLab](../images/swanlab.png)

SwanLab x CANN 社区合作课程

</div>

---

# 第 05 章 · 性能调优

## 章节定位

本章面向**已经跑通过 02 章 SFT / LoRA 的同学**，时长约 **45–60 分钟**，作为大模型课程通往 **AscendC 算子开发课程**的过渡节。
预期读者在学完本章后，能用 Amdahl 律解释"为什么单算子 ~7× 加速到端到端只剩 ~1.5×"，能跑通一次别人写好的 fused AscendC kernel 在 Qwen3 上的接入与提速，并理解下一步为什么要自己动手写 kernel。

## 计划节次

| 节次 | 标题 | 状态 |
|------|------|------|
| 05.01 | 性能调优全景与 Amdahl 律 | 建设中 |
| [05.02](./05.02_swan_rmsnorm_acceleration.ipynb) | SwanRmsNorm AscendC 算子加速 Qwen3 微调 | 已发布 |
| 05.03 | 章节小结：从"接算子"走向"写算子" | 建设中 |

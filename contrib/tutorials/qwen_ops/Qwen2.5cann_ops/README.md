# Qwen2.5 CANN 自定义算子实验项目

本项目面向 `Qwen2.5-0.5B`，在 Ascend NPU / CANN 8.5.0 环境中实现并验证五类 Transformer 核心计算的 AscendC 自定义算子：GEMM、RMSNorm、SwiGLU、RoPE 与 GQA-Attention。每类算子均提供基础版和优化版；项目不仅比较单算子，还把同一组算子统一接入 Qwen2.5 前向，用 logits 对齐验证端到端可用性。

## 项目目标

项目回答三个问题：

1. 如何把 Qwen2.5 的关键计算写成可编译、可被 PyTorch 调用的 AscendC 自定义算子？
2. 如何在不改变数学语义的前提下，对每类算子实施可审计的优化？
3. 如何将五类动态库同时接入真实 Qwen2.5 模型，并把单算子收益与模型级结果分开解释？

基础版刻意保留易读、可验证的朴素实现，作为功能与性能基线；优化版针对实际瓶颈引入 tile、局部 buffer、向量计算、数据布局、调用边界或 GQA 数据复用等措施。所有性能结论均标明计时范围，避免将 wrapper、H2D/D2H 或模型其他层的时间误归因给 kernel。

## 工程结构

```text
Qwen2.5cann_ops/
├── Gemm{Baseline,Optimized}Experiment/        # Linear 的矩阵乘法
├── RmsNorm{Baseline,Optimized}Experiment/     # Qwen RMSNorm
├── SwiGlu{Baseline,Optimized}Experiment/      # MLP 激活
├── Rope{Baseline,Optimized}Experiment/        # Q/K Rotary Position Embedding
├── GqaAttention{Baseline,Optimized}Experiment/# GQA causal attention
├── Qwen2.5{Baseline,Optimized}IntegrationExperiment/
│   └── qwen2_5_five_ops_benchmark.py          # 五算子统一接入入口
├── Qwen2.5FiveOpsIntegration/common/          # 共享接入逻辑
├── scripts/benchmark_rope_only.py              # RoPE-only 对比入口
├── 实验手册/                                   # Markdown 原始实验记录
└── 实验手册word_version/                       # 基于模板生成的 12 份 Word 手册
```

每个算子工程采用相同分层：`op_kernel/` 是 AscendC kernel 与 tiling，`torch_extension/` 是 ACL/PyTorch 注册层，`tests/` 是正确性和 Qwen 链路测试，`scripts/` 是环境检查、构建与 profile 入口，`out/` 保存构建产物。

## 实现方法

### 1. 算子开发链路

每个算子按以下路径完成：

```text
Qwen/PyTorch Tensor
  → torch.ops.<namespace>.<op>
  → C++ ACL host wrapper（shape 校验、tiling、内存/stream）
  → ACLRT_LAUNCH_KERNEL
  → AscendC kernel（GM/UB/Vector/AI Core）
  → 输出 Tensor
```

基础版优先保证公式、shape 和布局与 Qwen 对齐；优化版在相同 schema 或兼容 schema 下调整内存访问、并行分工和调用边界。所有基础/优化版的动态库分别在独立 Python 进程加载，避免 `torch.ops` namespace 或 CANN kernel 库互相干扰。

### 2. 五类算子的优化思路

| 算子 | 基础版 | 优化版的主要措施 |
|---|---|---|
| GEMM | GM 标量三重循环 | M/N/K tile、A/B 分块 CopyIn、向量 Mul + ReduceSum、二维输出 tile 分配 |
| RMSNorm | 两次 GM 扫描：平方和、再归一化 | UB CopyIn/CopyOut、向量平方/ReduceSum、局部乘缩放 |
| SwiGLU | 标量近似 exp、逐元素计算 | tile 内 `Exp/Div/Mul` 矢量流水、gate/up 局部块、一次 kernel 完成激活与乘法 |
| RoPE | 展开 trig，Q/K 分两次调用，GM 标量读写 | 最多 32 核、动态 tile、GM↔UB 批量 DataCopy、`Mul/Sub/Add` 向量计算；compact API 在 wrapper 展开 trig 后复用稳定二维 tiled kernel |
| GQA | 标量点积与逐标量累积 | Q/K/V 局部 buffer、向量 `Mul/ReduceSum`、online softmax、复用 KV head 映射 |

## 构建与单算子验证

以任一工程为例：

```bash
cd /home/user/Qwen2.5cann_ops/GemmOptimizedExperiment
source /home/user/Ascend/ascend-toolkit/cann-8.5.0/set_env.sh
bash scripts/check_env.sh
bash scripts/build.sh
bash scripts/run_test.sh tests/test_torch_op.py
```

单算子正确性以 PyTorch/NumPy golden 为参照；测试检查最大绝对误差、平均绝对误差或 `allclose`。`out/bin/*_standalone` 使用 warmup、repeat、rounds 和 `aclrtEventElapsedTime` 记录纯设备时间。RoPE 基础/优化版本比较例外：使用 `scripts/benchmark_rope_only.py --compare`，只计时 Q/K RoPE API 路径（包含各自 wrapper 的 H2D/kernel/D2H）。

## 已实测的单算子结果

下表来自宿主 Ascend NPU。除 GEMM 基础版（rounds=1）外，standalone 均为多轮均值；RoPE 为专用 API benchmark，口径已单列。

| 算子 | 基础版 | 优化版 | 基础÷优化 |
|---|---:|---:|---:|
| GEMM，M=128 K=1024 N=512 | 832385.742 us | 9873.167 us | 84.31× |
| RMSNorm，128×1024 | 490.150 us | 9.565 us | 51.24× |
| SwiGLU，128×1024 | 2065.420 us | 8.499 us | 243.02× |
| RoPE，B=1 S=128 Hq=14 Hkv=2 D=64 | 885.626 us | 待按当前稳定版重测 | — |
| GQA，B=1 Hq=8 Hkv=2 S=32 D=64 | 684.147 us | 212.200 us | 3.224× |

这些数据是同口径的单算子指标，不能直接当作完整模型吞吐或端到端加速比。RoPE 原 388.143 us / 2.282× 数据来自历史 compact/fused 实现；当前稳定版为解决 Ascend 910B3 上的非确定性错误，已改为 wrapper 展开 trig 并分别计算 Q/K，因此旧数据不再代表当前代码。

## 五算子接入 Qwen2.5

统一接入脚本先运行原生 Qwen 前向，再加载五类 `.so` 并 monkey patch 模块 `forward`：

- `torch.nn.Linear` 替换为自定义 GEMM；
- Qwen RMSNorm 替换为 `rms_norm`；
- 每层 MLP 的 gate/up 激活替换为 SwiGLU；
- 每层 attention 在 Q/K 位置调用 RoPE，并以自定义 GQA 替换 score/value 聚合；
- 保留原模型权重、position embedding、mask/cache 参数解析和输出投影。

```bash
cd /home/user/Qwen2.5cann_ops
source /home/user/Ascend/ascend-toolkit/cann-8.5.0/set_env.sh
python3 Qwen2.5BaselineIntegrationExperiment/qwen2_5_five_ops_benchmark.py \
  --model /home/user/Models/Qwen2.5-0.5B --repeat 1
python3 Qwen2.5OptimizedIntegrationExperiment/qwen2_5_five_ops_benchmark.py \
  --model /home/user/Models/Qwen2.5-0.5B --repeat 1
```

2026-07-15 宿主 NPU 实测，两套统一接入均完成 169 个 Linear、49 个 RMSNorm、24 个 SwiGLU、24 个 Attention/RoPE/GQA 替换，最终 logits 均满足 `torch.allclose(atol=1e-2, rtol=1e-2)`：

| 版本 | 原生前向 | 五算子自定义前向 | max abs diff | 结论 |
|---|---:|---:|---:|---|
| 基础版 | 280.864 ms | 50792.863 ms | 0.001199722 | PASS |
| 优化版 | 293.109 ms | 2185.963 ms | 0.001208305 | PASS |

统一接入的时间包含 CPU 模型前向及所有 NPU/CPU bridge copy，主要用于验证接入链路与当前 wrapper 边界。优化版相对基础版显著降低了该路径的时间，但要获得真正部署级性能，下一步仍需要将 host wrapper 改造成连续的 NPU-resident 数据流，减少 CPU↔NPU 往返和临时 `.contiguous()`。

## 文档与复现

- Markdown 实验记录位于 `实验手册/`；
- 12 份模板化 Word 手册位于 `实验手册word_version/`；
- 全项目测试状态见 [TEST_STATUS.md](TEST_STATUS.md)；
- 单算子实测汇总见 `test_logs/2026-07-15/single_op_benchmark/RESULTS.md`；
- Word 手册可由 `python3 scripts/generate_word_manuals.py` 重新生成，生成时保留模板的页眉、页脚、主题与页面设置。

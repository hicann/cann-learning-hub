# QwenRoPeCustomOpt 实验记录

> **项目**: AscendC 自定义 RoPE 算子优化
> **硬件**: Ascend 910B4 ×8 (CANN 8.5.0), Kunpeng-920
> **模型**: Qwen2.5-0.5B (24 layers, 14 Q-heads / 2 KV-heads, head_dim=64)
> **作者**: ChenHui 2026-07

---

## 目录

1. [优化总览](#1-优化总览)
2. [V1: 紧凑布局 + 动态 Tiling](#2-v1-紧凑布局--动态-tiling)
3. [V2: 增量索引](#3-v2-增量索引)
4. [V3: 短序列 Tiling 启发](#4-v3-短序列-tiling-启发)
5. [V4: Q/K 融合调用](#5-v4-qk-融合调用)
6. [V5: NPU-Resident 算子路径](#6-v5-npu-resident-算子路径)
7. [Profile 方法论](#7-profile-方法论)
8. [E2E 性能数据](#8-e2e-性能数据)
9. [版本历史](#9-版本历史)

---

## 1. 优化总览

| 版本 | 优化手段 | 作用原理 | 预期收益 |
|:---:|---|---|---|
| V1 | Layout + Tiling | cos/sin 紧凑存储减少 H2D；32 核动态分配提升并行 | trig 数据 8–16× 减少，kernel 3.7× 加速 |
| V2 | Index | 边界自增替代 per-row 除法取模 | 消除 AI Core 上高开销除法 |
| V3 | Heuristic | 短序列封顶 8 核，避免核调度反效 | 短序列稳定 |
| V4 | Q/K Fusion | 一次 op 调用完成 Q 和 K 的 RoPE | 省 1 次 dispatcher + trig H2D + sync |
| V5 | NPU-Resident | 输入输出在 NPU 上，跳过 CPU↔NPU 数据搬运 | 消除 ~430us/层 传输 |

---

## 2. V1: 紧凑布局 + 动态 Tiling

### 2.1 紧凑 cos/sin 布局

**问题**: 基线将 cos/sin 按 head 维度展开再传给 kernel。

```
基线:
  Q RoPE: cos/sin 形状 [batch × heads × seq, dim] = [1792, 64]  (458KB)
  K RoPE: cos/sin 形状 [batch × kvheads × seq, dim] = [256, 64]  (65KB)
  H2D 传输: 523KB

优化:
  Q RoPE: cos/sin 形状 [batch × seq, dim] = [128, 64]  (32KB)
  K RoPE: 同上（所有 head 共享同一 trig 表）
  H2D 传输: 32KB  (16× 减少)
```

**原理**: cos/sin 只依赖 batch 和 seq position，与 head 无关。展开后同一个 seq 位置的 cos/sin 被复制了 `num_heads` 次。紧凑存储后，由 kernel 内部通过 row 索引映射到正确的 trig 行：

```
trig_row = batch × seqLen + seqPos
```

Tiling 结构从 4 字段扩展到 8 字段，新增 `seqLen`、`numHeads`、`trigTokens`、`compactTrig`。

### 2.2 动态 Tiling（多核分片）

**问题**: 基线固定使用 8 核，对于大 token 量未充分利用 910B 的 32 核。

**原理**: 根据总行数动态选择核数，ceiling division 均分行：

```cpp
coreNum     = totalTokens <= 16 ? 8 : 32;
rowsPerCore = (totalTokens + coreNum - 1) / coreNum;
```

内核通过 `GetBlockIdx() × rowsPerCore` 分片，每核独立处理自己的行。

**效果**:

| totalTokens | blockDim=8 | blockDim=32 | 加速比 |
|------------:|-----------:|------------:|------:|
| 10 | 6.8 us | 7.8 us | 0.87× (核调度开销 > 计算) |
| 70 | 13.1 us | 7.4 us | 1.77× |
| 1792 | 262.5 us | 69.8 us | **3.76×** |

---

## 3. V2: 增量索引

### 3.1 问题

紧凑布局下，每行需要除法取模来映射 trig row：

```cpp
for (row = startRow; row < endRow; row++) {
    batch  = row / (numHeads * seqLen);   // 除法 (~30 cycles)
    seqPos = row % seqLen;                // 取模
    trigRow = batch * seqLen + seqPos;
}
```

### 3.2 原理

首行算一次，后续行通过边界自增替代除法：

```cpp
// 初始化（核内只执行一次）
batch       = startRow / (numHeads * seqLen);
rowInBatch  = startRow - batch * (numHeads * seqLen);
seqPos      = rowInBatch % seqLen;

for (row = startRow; row < endRow; row++) {
    trigRow = batch * seqLen + seqPos;   // 只有乘法加法

    seqPos++;                             // 自增，过 seqLen 回零
    if (seqPos >= seqLen) seqPos = 0;

    rowInBatch++;                         // 自增，过 rowsPerBatch 换 batch
    if (rowInBatch >= numHeads * seqLen) {
        rowInBatch = 0; batch++; seqPos = 0;
    }
}
```

**不要求 tiling 对齐 batch 边界**——增量机制自动处理跨 batch、跨 head 的情况。切在任意位置都能正确推进。

---

## 4. V3: 短序列 Tiling 启发

**原理**: totalTokens ≤ 16 时，核调度开销 > 额外核的并行收益。封顶 8 核避免退化。

```cpp
uint32_t choose_core_num(int64_t totalTokens) {
    int64_t maxCore = totalTokens <= 16 ? 8 : 32;
    return std::max(1u, uint32_t(std::min(maxCore, totalTokens)));
}
```

---

## 5. V4: Q/K 融合调用

### 5.1 问题

基线每层发两次 op 调用，每次独立传输 cos/sin 和同步：

```
rope_compact(q, cos, sin) → H2D(q) + H2D(cos/sin) + launch + sync + D2H(q)
rope_compact(k, cos, sin) → H2D(k) + H2D(cos/sin) + launch + sync + D2H(k)
```

cos/sin 被传了两遍，做了两次 sync。

### 5.2 原理

合并为一次 op 调用 `rope_qk_compact(q, k, cos, sin, seq, q_heads, k_heads)`：

```
h2d(q) + h2d(k) + h2d(cos/sin 一次) + h2d(tiling_q + tiling_k)
→ launch Q kernel
→ launch K kernel  (同 stream，不同 tiling)
→ sync (一次)
→ d2h(q_out) + d2h(k_out)
```

Q 和 K 仍然独立计算（head 数不同），只是 host 侧调度合并了。

---

## 6. V5: NPU-Resident 算子路径

### 6.1 问题

CPU-path 每层 RoPE 有固定的数据搬运开销：

| 组件 | 耗时 |
|---|---|
| aclrtMemcpy (H2D/D2H ×8) | ~304 us |
| aclrtSynchronizeStream | ~147 us |
| Kernel compute | ~152 us |
| **总计** | **~531 us** |

### 6.2 原理—PyTorch Dispatch 双路径

在 `rope_basline_torch.asc` 中注册两条 dispatch 路径：

```cpp
// 路径 1: CPU tensor → H2D→kernel→D2H
TORCH_LIBRARY_IMPL(qwen_rope_custom_opt, CompositeExplicitAutograd, m) {
    m.impl("rope_qk_compact", rope_qk_compact_npu);
}

// 路径 2: NPU tensor → 零拷贝
TORCH_LIBRARY_IMPL(qwen_rope_custom_opt, AutogradPrivateUse1, m) {
    m.impl("rope_qk_compact", rope_qk_compact_npu_resident);
}
```

NPU-resident 实现的关键差异：

```cpp
// CPU-path (旧):
auto x_c = x.contiguous().to(at::kFloat);            // 强制 CPU 内存
aclrtMemcpy(g_xD, ..., x_c.data_ptr(), ..., HOST_TO_DEVICE);  // H2D
// ... kernel ...
auto y = at::empty({...}, at::kFloat);               // CPU 上分配
aclrtMemcpy(y.data_ptr(), ..., g_yD, ..., DEVICE_TO_HOST);    // D2H

// NPU-resident (新):
auto x_c = x.contiguous();                           // 已在 NPU 上
auto y = at::empty({...}, x_c.options());            // 在 NPU 上分配
ACLRT_LAUNCH_KERNEL(rope_baseline_kernel)(
    ..., x_c.data_ptr(),           // ★ NPU 设备指针，零 H2D
    ..., y.data_ptr(),             // ★ NPU 设备指针，零 D2H
    ...);
```

### 6.3 何时生效

当 `model.to("npu")` 后，所有权重和激活在 NPU 上，Q/K/V projection 输出自然在 NPU 上。monkey-patch `apply_rotary_pos_emb` 后，dispatcher 自动选 NPU-resident 路径。

```
q_proj(hidden_states) → NPU tensor
    → apply_rotary_pos_emb(q, k, cos, sin)
        → dispatcher: tensor.device = "npu:0"
            → AutogradPrivateUse1
                → rope_qk_compact_npu_resident()
                    → data_ptr() 直接传 NPU 指针
                    → 零 H2D/D2H
```

### 6.4 验证

msprof profile 原生 NPU forward 证实 100% 算子在 NPU 上：

| 算子类型 | 执行引擎 |
|---|---|
| MatMulV2 (Q/K/V/O + FFN) | AI_CORE (Cube) |
| FlashAttentionScore | MIX_AIC |
| RoPE 子算子 | AI_VECTOR_CORE |
| Swish/Silu | AI_VECTOR_CORE |
| RMSNorm | AI_VECTOR_CORE |

---

## 7. Profile 方法论

### 7.1 msprof 驱动优化决策

V4 完成后，使用 CANN 8.5 的 `msprof` 对 fused QK RoPE op 进行完整 profile：

```bash
~/Ascend/ascend-toolkit/cann-8.5.0/tools/profiler/bin/msprof \
    --application="python3 /tmp/profile_target.py" \
    --output="./output/msprof_torch_op" \
    --runtime-api=on --task-time=on --ai-core=on \
    --aic-metrics=Memory --sys-hardware-mem=on
```

### 7.2 关键发现

| 组件 | 耗时 (us) | 占比 |
|---|---|---|
| aclrtMemcpy | ~304 | 57% |
| aclrtSynchronizeStream | ~147 | 28% |
| Kernel compute | ~152 | 29% |

**AI Vector 内存带宽仅 0.089 GB/s**（HBM 标称 1200 GB/s 的 0.007%）——标量 GM 访问是 kernel 瓶颈。

### 7.3 数据驱动决策链

```
msprof → memcpy+sync 占 85% → 先做 NPU-resident (V5)
      → kernel 仅占 29%   → 后做 vector/UB/pipeline (Route B)
      → 带宽 0.089 GB/s   → vector 化优先级 > pipeline
```

---

## 8. E2E 性能数据

### 8.1 CPU-path E2E

| seq | 原生 RoPE (ms) | 自定义 CPU-path (ms) | speedup |
|----:|---------------:|---------------------:|--------:|
| 32 | 934 | 971 | 0.96× |
| 64 | 2038 | 1569 | 1.30× |
| 128 | 847 | 480 | 1.76× |

自定义 CPU-path H2D/D2H 开销固定 ~12ms，序列越长原生 CPU RoPE 越重，custom 反超。

### 8.2 NPU E2E (model.to("npu"))

| seq | 原生 NPU (ms) | 自定义 NPU-resident (ms) | speedup |
|----:|--------------:|-------------------------:|--------:|
| 8 | 40.9 | 38.9 | 1.05× |
| 32 | 43.6 | 41.0 | 1.06× |
| 64 | 40.8 | 41.7 | 0.98× |

短序列下 FFN (4864→896→4864 matmul) 占主导，RoPE 差异淹没在噪声中。自定义的 `.contiguous()` 在长序列引入 NPU 内部 device copy 抵消了 fused kernel 收益。

### 8.3 单算子 NPU-resident

| seq | CPU-path (us) | NPU-resident (us) | speedup |
|----:|--------------:|------------------:|--------:|
| 64 | 611 | 646 | 0.95× |
| 128 | 917 | 660 | 1.39× |

---

## 9. 版本历史

```
f74ff1b  基线 (QwenRoPeCustom)
  │
ae6d295  V1: 紧凑 cos/sin layout + 动态 core count (8→32)
bd79231  V2: 增量索引替代 per-row 除法取模
c758adf  V3: 短序列 Tiling 启发 (≤16 tokens → 8 cores)
2b639ef  V4: Q/K fused RoPE op (一次调用完成 QK)
05891ba  V5: NPU-resident tensor path (AutogradPrivateUse1 dispatch)
06073a0  测试脚本: test_npu_e2e.py + test_npu_benchmark.py
6e0c1d3  自动环境: _setup_env.py (detect CANN + os.execve inject)
```

### 项目测试脚本

| 脚本 | 用途 |
|---|---|
| `tests/test_torch_op.py` | CPU-path 单算子正确性 |
| `tests/test_npu_resident.py` | NPU-resident 三算子正确性 (baseline/compact/fused) |
| `tests/test_npu_e2e.py` | E2E 前向 token 匹配验证 |
| `tests/test_npu_benchmark.py` | NPU 原生 vs 自定义性能对比 |
| `tests/_setup_env.py` | 自动 CANN 环境检测注入 |

运行方式: `cd ~/Projects/QwenRoPeCustomOpt && python3 tests/<script>.py`

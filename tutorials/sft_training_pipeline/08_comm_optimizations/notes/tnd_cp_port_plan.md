# TND + CP 移植排期（工程文档，不面向学员）

> ## ⚠️ 更正记录（2026-09-16）
>
> 本文早期版本基于一个**错误的代理假设**：以为 `cp=2` 会让 FSDP mesh 退化成 1（参数/梯度/优化器状态不切分），于是得出"CP2 多付约 17 GiB 固定状态 → 2 卡两条容量墙几乎重合 → 容量实验必须上 4 卡"的结论。**该结论作废。**
>
> 事实：TorchTitan 的 legacy 路径把 `dp_shard` 与 `cp` **折进同一个 FSDP mesh**：
>
> ```python
> # torchtitan/distributed/parallel_dims.py
> fsdp = self.dp_shard * self.cp          # pinned 2807d3f 与旧 pin ac13e53 都是这个规则
> expected_sizes["fsdp"] = self.dp_shard * self.cp
> ```
>
> 实测（2 卡，当前版本）：`dp_shard=1, cp=2` 的 active meshes = `['loss', 'cp', 'fsdp']`，**`fsdp_size = 2`**；`dp_shard=2, cp=1` 同样是 2。因此 CP2 的参数/梯度/优化器状态照常分片，固定状态 ≈ `19.0/2 = 9.5 GiB/rank`，与 no-CP 相同。
>
> 影响：08.02 的"两条路线的 FSDP degree 都是 2"是对的；08.03 原文的"CP2 的 `dp_shard=1` 约 19.0 GiB/rank / 固定状态增加 9.5 GiB"是错的，已在 notebook 中修正。**旧版参考数据的结论（CP2 可运行、no-CP OOM）成立，不需要 4 卡。**

## 1. 现状与证据（2026-09-16，wordle-latest @ `0ea4b8d` + torchtitan `2807d3f`）

| 路线 | 状态 | 证据 |
|---|---|---|
| `sft_qwen3_1_7b_wordle_tnd` + `cp=1` | ✅ 可用 | `S=16,384` 跑满 10 步；peak active 28.30 GiB / reserved 32.58 GiB（双 rank 相同，采 profiler） |
| `sft_qwen3_1_7b_wordle_tnd` + `cp=2` | ❌ 启动即失败 | `NotImplementedError: Context Parallel only supports SDPA and FlexAttention. Varlen attention is not supported with CP.` |
| `sft_qwen3_1_7b_wordle`（flex）+ `cp=2` | ✅ 可用 | 上游 K/V all-gather CP，2 步完成；flex 在 NPU 上效率很低（MFU ≈ 2%），只能作能力探针 |
| SDPA + Ulysses（仓库代码） | ⚠️ 不可达 | `get_attention_config` 只提供 `flex / flex_flash / varlen`；SDPA 无法消费 per-document positions |

两处硬 gate（pin 版本）：

- `torchtitan/models/qwen3/model.py:91` —— `update_from_config` 中 `cp>1` 且 `inner_attention` 是 `VarlenAttention.Config` → `NotImplementedError`。
  该处**先于** override 执行（`torchtitan/trainer.py:348` 调用 `update_from_config`，`:355` 才 `apply_overrides`）。
- `torchtitan/distributed/context_parallel/api.py:103` —— `apply_cp_to_forward` 对 `VarlenAttention` 抛 `NotImplementedError`。

## 2. 参考实现（仓库内已有，同架构）

| 文件 | 作用 |
|---|---|
| `torchtitan_npu/override/qwen3_5/varlen_attention.py`（`asc_cp`） | CP-aware TND attention：GQA repeat → `exchange_sequence_heads` → FA v3（全局 `actual_seq_qlen`）→ `head_to_sequence_shard` |
| `torchtitan_npu/override/qwen3_5/parallelize.py` | `exchange_sequence_heads` / `head_to_sequence_shard`（DTensor `Shard(seq) ↔ Shard(head)` 重分布）、`build_sequence_metadata`（由 `CPVarlenMetadata` 重建全局 `cu_seq`）、`parallelize_qwen3_5_cp`（挂 mesh + metadata pre-hook + AC/compile/FSDP） |
| `torchtitan_npu/patches/torchtitan/distributed/context_parallel.py` + `varlen_cp.py` | 已把 `cp_shard` 的 varlen 分支做成 `CPVarlenMetadata`（上游 PR #3430 的一半，被 DeepSeek-V3.2/V4 消费） |
| `torchtitan_npu/override/deepseek_v3_2/sparse_attn/ascendc.py` | 备选方案（Q 分片 + K/V all-gather，用 `k_global_gather_indices` 取 K） |

**旧版本（08.03 参考数据的来源，当前仓库已无）**：`torchtitan_npu/models/common/npu_varlen_attention.py`（`NPUVarlenAttention`）、`torchtitan_npu/distributed/context_parallel/npu_varlen_cp.py`（`NPUVarlenUlyssesCP` + `_varlen_cp_mask_handler`）、`torchtitan_npu/models/qwen3/tnd_config.py`（放行上游 VarLen+CP 校验）。

上游状态：PR #3430 截至 torchtitan main tip `8f0e856` **仍未合入**（`context_parallel/` 下没有 `varlen_cp.py`）。

## 3. 改动清单

### M1：共享 Ulysses 工具下沉（约 0.5 天）

- 把 `exchange_sequence_heads` / `head_to_sequence_shard` / `sequence_to_head_shard` / `build_sequence_metadata` 从 `override/qwen3_5/parallelize.py` 移到公共位置（建议 `torchtitan_npu/models/common/ulysses.py`），qwen3_5 与 qwen3 共用；原位置保留 re-export。
- `build_sequence_metadata` 泛化：Qwen3 的 metadata 字段是 `cu_seq_q/cu_seq_k`，Qwen3.5 用 `actual_seq_qlen`；统一返回 `(cu_seqlens, cu_seqlens_cpu, actual_seq)`。

### M2：CP-aware TND attention（约 0.5 天）

- 在 `torchtitan_npu/override/qwen3/varlen_attention.py` 内加 CP 分支（或新增 `asc_cp` override）：`context_parallel_mesh` 由 parallelize 注入，`cp=1` 时行为不变；forward 顺序 = GQA 处理 → `exchange_sequence_heads` → BSND→TND → FA v3（全局 `actual_seq`，`sparse_mode=7`）→ `head_to_sequence_shard`。
- metadata 需同时接受 `VarlenMetadata`（no-CP）与 `CPVarlenMetadata`/`QwenCPMetadata`（CP），并在 debug 下断言两个 rank 的全局 `cu_seq` 一致。
- `n_kv=8, cp=2` 可整除，无需 `repeat_interleave`；`cp>8` 才需要（照 qwen3_5 写法）。

### M3：parallelize 接线（约 0.5 天）

在 `torchtitan_npu/models/qwen3/parallelize.py` 的 `parallelize_qwen3` 里加 varlen 分支（与现有 SDPA 分支同构）：① 注入 `context_parallel_mesh`；② 注册 metadata pre-hook（重建全局 `cu_seq`）；③ 把 `_upstream.apply_cp_to_forward` 临时置空以避免 `api.py:103` 的报错；④ FSDP/TP/AC/compile 继续交给上游（**不要**复制 qwen3_5 的手写 FSDP）。FSDP mesh 由上游按 `dp_shard × cp` 计算，CP2 会自动分片，无需额外处理。

### M4：放行 model-level gate（约 0.2 天）

- 必须打补丁：该 gate 在 override 之前运行，且不能把 attention Config 换成非 `VarlenAttention.Config` 子类（`Decoder.get_attention_masks` 依赖该 `isinstance`）。
- 实现：`torchtitan_npu/models/qwen3/cp_support.py` 提供 `allow_varlen_cp()`（包装 `update_from_config`，仅当 `inner_attention` 是 `AscVarlenAttention.Config` 时吞掉该 `NotImplementedError`），由 CP recipe 显式调用。

### M5：recipe 与参数（约 0.2 天）

- 新增 `sft_qwen3_1_7b_wordle_tnd_cp()`：`context_parallel_load_balancer = None`（让 Ulysses 交换后的序列保持原始顺序；flex 的 BlockMask 才用 `ptrr`）+ 注册 `parallelize_fn` 与 `asc_cp` override。
- 不改 `sft_qwen3_1_7b_wordle_tnd()` 的默认行为。

## 4. 正确性验证清单（防静默错误）

| 层级 | 检查 | 判据 |
|---|---|---|
| 单元 | 同一 packed 输入，`cp=1` 的 TND attention vs `cp=2` 的 exchange+TND | 拼回后 `allclose`（bf16 容差） |
| 单元 | 两个 CP rank 的全局 `cu_seq` | 完全相同；segments 数与 dataloader 重放一致 |
| 训练 | `cp=2` 与 `cp=1` 同数据、同 seed 的第一步 loss | 量级一致 |
| 训练 | 10 步 loss/grad_norm | 有限、无 NaN/Inf |
| trace | `npu::npu_fusion_attention_v3` 的 Input Dims | `[B×S, heads/C, D]`（token 维保持全局，head 维减半） |
| trace | 通信 | 出现 `c10d::alltoall_base_` 或 DTensor 重分布对应的 collective |
| 回归 | 08.03 采集命令 | 由 `NotImplementedError` 变为成功进入训练循环 |

## 5. 容量对照（更正后）

**正确的 CP2 等价代理**必须同时满足：与 CP2 相同的 **FSDP degree（`dp_shard × cp`）** 和相同的 **per-rank token 数**。因此目标点 `S=32768, LBS=4, GBS=8, cp=2, dp_shard=1`（per-rank 65,536 token，fsdp=2）的代理就是 `S=32768, LBS=2, GBS=4, dp_shard=2, cp=1`（per-rank 65,536 token，fsdp=2）。

已测数据（TND、bf16、无 profiler；`memory` 为日志 step 级读数）：

| 配置 | per-rank token | 口径 | 结果 |
|---|---:|---|---|
| `S=16384, LBS=2, dp_shard=2` | 32,768 | 2 步 step 1 / 10 步稳态 | 25.59 / 33.46 GiB |
| `S=32768, LBS=2, dp_shard=2` | 65,536 | 2 步 step 1 | 39.93 GiB |
| `S=20480, LBS=4, dp_shard=2` | 81,920 | 10 步稳态 | 53.31 GiB |
| `S=32768, LBS=3, dp_shard=2` | 98,304 | 2 步 | ❌ OOM（请求 9.28 GiB 时 active 46.83 GiB） |

（另有若干**代理无效**的测量——单卡不切分、`dp_replicate=2`（走 HSDP，模型常驻 8.46 GiB）——它们改变了 FSDP degree，不能代表 CP2，仅作废数据保留。）

**结论**：按正确代理，`S=32768, LBS=4` 下 CP2（per-rank 65,536 token）≈ 39.93 + 约 7（稳态增量，同口径实测 +6.9 ~ +7.9）+ 0.3–1.0（AllToAll buffer）≈ **47–48 GiB < 61.27 GiB → 可运行**；no-CP 同 workload（per-rank 131,072 token，模型外推 ≈ 68.6 GiB）→ **OOM**。与旧版参考数据的结论一致，**不需要 4 卡**。

**待办**：MVP 完成后用真 CP2 复测该点，把 47–48 GiB 的投影换成实测，并补 AllToAll 事件数与非重叠通信。

## 6. 排期与风险

| 里程碑 | 内容 | 估算 |
|---|---|---|
| M1 | Ulysses 工具下沉 + metadata 泛化 | 0.5 天 |
| M2–M4 | attention / parallelize / gate | 0.7 天 |
| M5 | recipe 与参数 | 0.2 天 |
| V1 | 正确性与回归验证（2 卡） | 0.5–1 天 |
| V2 | 在 `S=32768, LBS=4` 采真 CP2 的双 rank 峰值 | 0.3 天 |
| D1 | 08.02/08.03 回填实测（替换旧版参考数据） | 0.5 天 |

风险与对策：

- **DTensor 重分布在 NPU 的 alltoall 支持**：先用 2 卡小脚本验证 `Shard(1) ↔ Shard(2)`；不通过则退回显式 `dist.all_to_all`（旧仓库 `npu_varlen_cp.py` 的写法）。
- **静默数值错误**：Ulysses+varlen 的边界错误不报错、只让 loss 变差，必须执行 §4 的单元级对比。
- **上游 PR #3430 合入后的冲突**：本移植与其 backport 有重叠，合入后应删除 backport 并复核 `cp_shard`。
- **回滚**：所有改动限定在 `cp>1` 分支与独立 recipe 内。

## 7. 关于"能否直接复用 Qwen3.5 的 CP"（已实测）

**工具函数可以直接复用，整条 3.5 CP 不能直接用。**

1. **可直接复用**：`exchange_sequence_heads` / `head_to_sequence_shard` / `sequence_to_head_shard` / `QwenCPMetadata` / `build_sequence_metadata` 与模型无关。用 2 进程 CPU/gloo 验证：`exchange_sequence_heads` 把 `[B, S/C, H, D]` 变成 `[B, S, H/C, D]` 且与全量张量的 head 分片逐位一致，`head_to_sequence_shard` 逆变换误差 0。
2. **不能直接用（gate）**：把 `asc_cp` 写进 Qwen3 recipe 的 `override.imports` 后，两种顺序都失败——trainer 真实顺序（`update_from_config` 先）被拦；先 `apply_overrides`（`inner_attention` 变成 `qwen3_5.varlen_attention.Config`）**仍然**被拦，因为它同样是 `VarlenAttention.Config` 子类。
3. **不能直接用（metadata 契约不同）**：3.5 的 attention 读 `attention_masks.actual_seq_qlen`，而 Qwen3 的 `VarlenMetadata` 字段是 `(cu_seq_q, cu_seq_k, max_q, max_k, cu_seq_q_host)`，没有该字段；它必须与 3.5 自己的 `build_sequence_metadata` pre-hook 成对使用。
4. **不能直接用（parallelize 绑模型结构）**：`parallelize_qwen3_5_cp` 直接操作 `block.attn / block.full_attn`（混合线性注意力）与 `model.vision_encoder`，并手写 AC/compile/FSDP；Qwen3 应改为在 `parallelize_qwen3` 内加 varlen 分支 + 置空 `apply_cp_to_forward`，其余交给上游。

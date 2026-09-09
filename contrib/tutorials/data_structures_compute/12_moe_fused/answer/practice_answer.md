# 第 12 章编程实践题 — 参考实现

> 以下参考均基于 12.02 工程（`src/custom_op/`）。每题完成后务必运行
> `bash tools/run_all.sh` 确认 8/8 PASS。

## 实践题 1（简单）：新增形状的测试用例

### 修改

`tools/gen_test_data.py` 的 `CASES` 列表新增一行：

```python
CASES = [
    ...
    ("case_64_256_8_2", 64, 256, 8, 2),     # 新增：N=64
]
```

### 运行与验证

```bash
python3 tools/gen_test_data.py
bash src/custom_op/test/run.sh data/case_64_256_8_2
# 预期：PASS（128/128 idx 匹配）
```

### 切分参数解释

K=2 → `R_align = 64/gcd(4,64) = 16` 行；`blockDim = min(40, 64/16) = 4`；
`rowsPerCore = 16 × ceil(64/(4×16)) = 16`，`blockDim = ceil(64/16) = 4`。

即 4 核 × 16 行：每核的 idx 输出段 = 16×8B = 128B（2 条完整 line），
wt 段 = 16×4B = 64B（1 条完整 line），无跨核 line。

## 实践题 2（中等）：W_gate 的 UB 驻留优化

### 栈帧约束推导

910B 单核栈帧上限 32KB（第 8 章验证的约束）。成员数组
`float wT[kMaxE][kMaxD]` 占 `E·D·4B`，故可整体驻留的条件：

```
E·D·4 ≤ 32768  →  D·E ≤ 8192 元素
```

| 用例 | D·E | 4B×D·E | 可否整体驻留 |
| -- | -- | -- | -- |
| case_16_256_8_2 | 2048 | 8KB | ✅ |
| case_128_512_16_2 | 8192 | 32KB | ✅（临界） |
| case_1024_1024_32_4 | 32768 | 128KB | ❌ 需分块 |
| case_4096_2048_8_2 | 16384 | 64KB | ❌ 需分块 |

### 分块方案（D 维分块）

不能整体驻留时，按 d 维分块：把 [D, E] 的 W 按 `dBlock` 行分段，
每段 `dBlock·E·4B ≤ 32KB`（如 E=32 时 dBlock ≤ 256）。外层循环：

```cpp
// 伪代码：d 维分块驻留
for (uint32_t d0 = 0; d0 < D; d0 += dBlock) {
    // 1) 标量读 w_gate[d0:d0+dBlock, :] 转置进 UB 数组 wT[e][d_local]
    // 2) 对所有行：部分点积 scores[n][e] += Σ_{d in 块} x[n][d]·wT[e][d-d0]
}
// 注意：scores 需在块循环外维持累加状态（仍为 UB 数组，逐行处理时
// 需要把一行的 E 个部分和暂存，或改为"一行算完所有块再换行"的组织方式）
```

一种更简单的组织：**外层遍历行、内层遍历块**时，每行需要重复驻留 W 的所有块，
驻留失去意义；因此推荐**外层遍历块、内层遍历行**，并把每行的 E 维部分和
保留在 `scoresRow`（E ≤ 32 时仅 128B，可整体常驻）。

### 验证

```bash
cd src/custom_op && bash build.sh
bash ../../tools/run_all.sh          # 8/8 PASS（必须）
bash test/run.sh ../../data/case_4096_2048_8_2 --bench 10   # 前后对比
```

预期：W 的 GM 标量读从 `N·D·E` 次降到 `D·E`（每核一次）+ L2/UB 命中，
大 N 用例耗时有可观下降；但 x 的读取与标量 FMA 仍在，数量级不变。

## 实践题 3（困难）：多核共写 cache line 丢写的复现与修复

### 复现步骤

1. 临时修改 `op_kernel/moe_router_fused.cpp` 的 `Process()`：

```cpp
const uint32_t coreId = static_cast<uint32_t>(GetBlockIdx());
const uint32_t blockDim = static_cast<uint32_t>(GetBlockNum());
for (uint32_t n = coreId; n < N; n += blockDim) {   // 跨步行进（错误示范）
    ...
}
```

并把 `op_host/moe_router_fused.cpp` 的切分临时改回
`blockDim = min(coreNum, N)`、`rowsPerCore = 0`（kernel 不再使用）。重新编译：

```bash
cd src/custom_op && bash build.sh
```

2. 连续运行 3 次并导出输出：

```bash
for i in 1 2 3; do
  bash test/run.sh ../../data/case_128_512_16_2 --dump /tmp/repro_$i | grep -E "idx |wt "
done
```

3. 预期现象：三次运行 `idx match / wt fail` 数值互不相同（例如 134/256、152/256、
   118/256……），且 `--dump` 输出逐次比对可看到错误行集合不同——非确定性。

### 原因解释

case_128_512_16_2（K=2）：idx 输出共 128×8B = 1KB = 16 条 64B line；
跨步行进时，core c 负责行 {c, c+blockDim, ...}，任意一条 64B line 容纳 8 行，
这 8 行分属 8 个不同核（blockDim ≥ 8 时）——所有核都对每条 line 写入，
核间对同一 line 的并发标量写无一致性保证，部分写丢失。
`SetL2CacheHint(CACHE_MODE_DISABLE)` 在 3510 上无效（地址高位被屏蔽），
只能靠切分规避。

### 恢复

还原 `Process()` 的连续块循环与 host 的 `R_align` 切分（即 12.02 原始实现），
重新编译，确认：

```bash
bash ../../tools/run_all.sh    # 8/8 PASS 恢复
```

### 选做扩展（向量化）

把点积改为 `DataCopy` 批量读 + `Mul`/`Add` 向量指令，与第 8 章向量化实践题相同路径。
本环境已知限制（12.02 步骤 4.2 列举）可能导致无法 PASS——若如此，
请记录 `aiv_vec_ratio` 的 msProf 读数与具体错误现象（如 MIX 模式、
标量读竞态），"向量化思路 + 环境限制结论"同样视为完成。

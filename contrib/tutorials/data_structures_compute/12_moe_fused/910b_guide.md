# 实验 12 平台运行指南（MoE 融合算子）

## 环境要求

| 项目 | 要求 |
|------|------|
| 芯片 | 华为昇腾 910B3（Atlas A2 系列） |
| CANN SDK | 9.0.0+ (ascend-toolkit) |
| Python | 3.11+（含 numpy / torch / torch_npu，torch_npu 仅 baseline 计时需要） |
| 编译工具 | cmake 3.19+、gcc |

## 确认目标平台

```bash
# 查看 NPU 型号
npu-smi info

# 确认 CANN 环境
source $ASCEND_HOME_PATH/set_env.sh
```

| NPU 型号 | TARGET 值 | SoC 版本 |
|----------|-----------|----------|
| Ascend910B3 (Atlas A2) | `ascend910b` | ascend910b |

## 一键运行

```bash
# 1. 设置 CANN 环境
source $ASCEND_HOME_PATH/set_env.sh

# 2. 进入实验目录
cd 12_moe_fused

# 3. 生成测试数据（8 组边界用例，固定 seed=42，幂等）
python3 tools/gen_test_data.py

# 4. 编译算子包（约 2-3 分钟）
cd src/custom_op && bash build.sh && cd ../..

# 5. 单用例验证（以标准形态 case_128_512_16_2 为例）
bash src/custom_op/test/run.sh data/case_128_512_16_2

# 6. 全部 8 用例一键回归
bash tools/run_all.sh

# 7. 性能测量（事件计时，--bench 后跟迭代次数）
bash src/custom_op/test/run.sh data/case_128_512_16_2 --bench 100
```

`test/run.sh` 自动完成：部署算子包到 `build_out/opp_pkg`（作为 `ASCEND_CUSTOM_OPP_PATH`，
无需 root 权限安装）→ 编译 aclnn 测试程序 → 运行并输出比对结果。

## 预期输出

单用例（case_128_512_16_2）：

```text
[tiling] moe_router_fused N=128 D=512 E=16 realE=16 K=2 blockDim=8 rowsPerCore=16
[case] N=128 D=512 E=16 K=2
[idx ] match=256/256 tie_diff=0 real_diff=0
[wt  ] max_abs=6.066561e-04 max_rel=1.816886e-03 fail=0/256
PASS
```

回归（`tools/run_all.sh`）：8/8 PASS。idx 逐元素精确匹配，wt 误差为
FP16 量化量级（实测 max_abs ≤ 7.6e-4，阈值 1e-2）。

性能（纯标量融合版，910B3，40 AIV 核）：

| 用例 (N·D·E·K) | blockDim | fused 实测 | 4 算子 baseline | 说明 |
| -- | -- | -- | -- | -- |
| 16·256·8·2 | 1 | ~0.34 ms | —（发射开销下限 ~0.23 ms） | 小 N 发射主导 |
| 128·512·16·2 | 8 | ~2.4 ms | 0.226 ms | 标量读延迟主导 |
| 1024·1024·32·4 | 32 | ~106 ms | 0.263 ms | 计算密集 |
| 4096·2048·8·2 | 37 | ~61 ms | 0.310 ms | 计算密集 |

结论（详见 12.02 步骤 7）：融合的访存收益为真（中间张量零落 GM、4 发射合 1），
但纯标量实现的计算吞吐是新瓶颈——这是"融合改变访存形态、不改变计算吞吐"的实例。

## 平台差异说明

| 项目 | 910B3（Atlas A2） |
|------|-------------------|
| AI Core（AIV）数量 | 40 |
| UB 大小 | ~192KB/核（本算子仅用 < 0.5KB） |
| L2 cache line | 64B（`kernel_utils_constants.h: CACHE_LINE_SIZE = 64`） |
| TARGET 值 | ascend910b |
| 多核切分 | 连续块：`[c*rowsPerCore, (c+1)*rowsPerCore)`，尾核收尾 |

## 常见问题

### 1. 多核下输出元素随机错（丢写成 0），单核正常？

**原因**：标量 GM 写（`SetValue`）经 L2 缓存，**多核并发写同一条 64B
cache line 会非确定性丢失部分写**（核间无写一致性）。若输出小张量采用
跨步行进切分（`n = coreId; n += blockDim`），各核输出按字节交错，必然共写
同一 line。

**解决**：本工程 host TilingFunc 已采用连续块切分 + 64B 对齐块边界
（`R_align = 64/gcd(2K,64)` 行），保证每条 line 只被一个核写入；
rowsPerCore 向上取整均衡各核负载。详见 `docs/design.md` §3。

### 2. 为什么不用向量指令/高阶库加速？

本环境（910B + CANN 9.0.0）实测存在多处工具链缺陷（向量 Cast 不支持
BF16、SyncAll 触发 MIX 模式、reduce 指令族失效、"循环内标量读 + 向量指令混用"
误编译等），本章刻意采用与第 8 章同构的纯标量范式以聚焦融合的数据结构分析。
向量化的收益与风险作为 12.03 实践题（困难）讨论。

### 3. `SetL2CacheHint(CACHE_MODE_DISABLE)` 能绕过多核写冲突吗？

不能。3510（910B）架构上 `SetValue`/`GetValue` 内部会屏蔽地址高位的
cache 模式位，该 hint 为死代码（实测无效）。正确做法是切分上避免共写。

### 4. 编译环境找不到 msopgen / msprof 等工具？

**原因**：未加载 CANN 环境。

**解决**：`source $ASCEND_HOME_PATH/set_env.sh` 后
`ls $ASCEND_HOME_PATH/bin/ | grep -E "msopgen|msprof|mssanitizer"`。

## 注意事项

1. 算子包通过 `ASCEND_CUSTOM_OPP_PATH` 环境变量加载（`test/run.sh` 自动部署），
   不修改系统目录，无需 root；
2. 测试数据（`data/`，约 59MB）不入库，由 `tools/gen_test_data.py` 再生（幂等）；
3. `test/main.cpp` 支持 `--dump <prefix>` 导出内核输出（离线逐行分析）、
   `--bench <iters>` 事件计时（性能测量）。

# Reduce算子与优先队列模拟堆 — 910B 平台运行指南

## 环境要求

| 项目 | 要求 |
|------|------|
| 芯片 | 华为昇腾 910B (Ascend910B) |
| CANN SDK | 8.0+ / 9.0+ (ascend-toolkit / cann) |
| Python | 3.8+ (含 numpy) |
| 编译工具 | cmake, gcc |

## 一键运行

```bash
# 1. 设置 CANN 环境（路径根据实际安装位置调整）
source /home/developer/Ascend/cann-9.0.0/set_env.sh
# 或
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 2. 进入实验目录
cd 04_Reduce算子与优先队列模拟堆/src/reduce_lab

# 3. 编译算子（TARGET=ascend910b，自动设置 BLOCK_DIM=20）
TARGET=ascend910b bash scripts/build_ops.sh

# 4. 设置自定义 OPP 路径
source scripts/env_custom_opp.sh

# 5. 编译 runner
bash scripts/build_runner.sh

# 6. 生成测试数据
python3 scripts/gen_data.py --num_tokens 1024 --top_k 4

# 7. 运行 benchmark
aclnn_runner/build/main_reduce_benchmark data 1024 4
```

## 预期输出

```
=== Reduce Operator Lab Benchmark ===
N=1024 K=4 BLOCK_DIM=20
Timing: aclrtEvent, 3 warmup + 10 timed iterations

[ReduceSum]  time=0.0XX ms  result=43.3495  ref=43.3750  error=0.0255  PASS
[ReduceMax]  time=0.0XX ms  result=4.0820   ref=4.0820   error=0.0000  PASS
[TopK]       time=0.0XX ms
  result values: 4.0820 3.6641 2.9141 2.9043
  result indices: 857 908 5 521
  ref values:    4.0820 3.6641 2.9141 2.9043
  ref indices:   857 908 5 521
  PASS

=== Done ===
```

## 910B 与 310B 的差异

| 项目 | 310B | 910B |
|------|------|------|
| AI Core 数量 | 8 | 20+ |
| BLOCK_DIM | 8 | 20 |
| UB 大小 | ~256KB/核 | ~512KB/核 |
| Cube Core | 无 | 有 |
| 编译命令 | `TARGET=ascend310b bash scripts/build_ops.sh` | `TARGET=ascend910b bash scripts/build_ops.sh` |

## 注意事项

1. **BLOCK_DIM 自动设置**：`build_ops.sh` 根据 `TARGET` 环境变量自动 patch HOST 文件中的 `BLOCK_DIM`（310B=8, 910B=20）
2. **输出大小动态传递**：算子通过 tiling 数据中的 `outputSize` / `blockDim` 字段将 BLOCK_DIM 传递给 kernel，避免硬编码
3. **Python 版本**：推荐 Python 3.8+，需安装 numpy（`pip3 install numpy`）
4. **首次编译慢**：TBE 编译器首次编译 kernel 约需 2-3 分钟
5. **精度验证**：ReduceSum 的 FP16 累加有微小精度误差（~0.025），属正常现象

## 环境清理

```bash
cd 04_Reduce算子与优先队列模拟堆/src/reduce_lab
rm -rf custom_ops/generated aclnn_runner/build data/input data/output

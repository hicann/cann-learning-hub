# 910B 运行指南

## 目标平台

本实验默认面向 Ascend 910B。910B 的 AI Core 数量与 310B 不同，因此不能把 310B 的 `BLOCK_DIM=8` 直接带到 910B。`scripts/build_ops.sh` 在生成工程后会按 `TARGET` 将 Host 侧默认值切换为 910B 的 `16`。当前 310B 与 910B 均已完成设备验证。

## 运行步骤

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cd src/moe_sort_lab

TARGET=ascend910b bash scripts/build_ops.sh
source scripts/env_custom_opp.sh
bash scripts/build_runner.sh

python3 scripts/gen_data.py \
  --num_tokens 1024 \
  --num_experts 64 \
  --hidden_size 128 \
  --top_k 2

aclnn_runner/build/main_benchmark data 1024 128 2
aclnn_runner/build/main_full_pipeline_benchmark data 1024 128 2
```

## 910B 与 310B 的差异

| 项目 | 310B | 910B |
|------|------|------|
| 构建目标 | `ascend310b` | `ascend910b` |
| 示例 BLOCK_DIM | 8 | 16 |
| 构建命令 | `TARGET=ascend310b bash scripts/build_ops.sh` | `TARGET=ascend910b bash scripts/build_ops.sh` |
| Kernel 数据切分 | 由 tiling 的 `tokensPerCore` / `rowsPerCore` 决定 | 同左，不在 Kernel 中写死芯片型号 |

Kernel 只读取 tiling 中的分块参数；Host 侧负责设置 `SetBlockDim` 和每个核处理的 token/row 范围。这样 910B 适配集中在构建目标和 Host tiling，不需要修改 Kernel 算法。

## 注意事项

1. 本教学实现固定 `top_k=2`，数据生成和 runner 参数必须保持一致。
2. `main_full_pipeline_benchmark` 会单独报告 Host 构造 `sortedOrder` 的时间；这部分不应误读为 AI Core 排序 kernel 时间。
3. `expertOut` 在示例中使用 identity 数据，实验重点是路由与 token 搬运路径，不代表真实专家 FFN 计算。
4. 首次构建会自动准备自定义算子工程，完成后继续执行 runner 编译和 benchmark。

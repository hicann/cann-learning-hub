# 树形任务队列与流水线调度 — 910B 平台运行指南

## 环境要求

| 项目 | 要求 |
|------|------|
| 芯片 | 华为昇腾 Ascend 910B3 |
| CANN SDK | CANN 9.1.0（已验证） |
| Python | 3.8+（仅标准库，无需额外依赖） |
| 编译工具 | cmake, gcc |

## 一键运行

```bash
# 1. 设置 CANN 环境（路径根据实际安装位置调整）
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
# 或
source /home/developer/Ascend/cann-8.5.2/set_env.sh

# 2. 进入实验目录
cd src/tree_queue_lab

# 3. 编译算子（TARGET=ascend910b，默认目标）
TARGET=ascend910b bash scripts/build_ops.sh

# 4. 设置自定义 OPP 路径
source scripts/env_custom_opp.sh

# 5. 编译 runner
bash scripts/build_runner.sh

# 6. 生成测试数据并运行 benchmark
python3 scripts/gen_data.py --num_nodes 31 --seed 10 --output data
aclnn_runner/build/main_tree_queue_benchmark data priority
aclnn_runner/build/main_tree_queue_benchmark data fifo
```

## 预期输出

```
=== Tree Queue Pipeline 910B Benchmark ===
tasks=31 order=priority queue_depth=2 compute_lanes=2 block_dim=1
[pipeline] time=0.00X ms end=116.0000 ref=116.0000 max_error=0.0000 PASS
[dependency] value=1 PASS
```

`max_error=0` 表示设备 `stage_end` 与 Python reference 完全一致；`dependency value=1` 表示父子依赖约束全部满足。

## 远程验证状态

本实验已在远程 Ascend 910B 环境完成运行验证，Python reference、算子构建、ACLNN runner 以及 priority/FIFO 两种 benchmark 流程均已跑通。提交 PR 时请将远程环境中的 CANN 版本、设备具体型号和实际终端输出同步到 PR 描述，便于 Reviewer 复核。

## 已完成的 910B3 验证

验证环境为 CANN 9.1.0 + Ascend 910B3。验证时曾将工程复制到 `/tmp` 下的纯英文路径，以规避原中文章节目录导致的 `msopgen` 路径解析错误；当前章节目录已经改为 `10_tree_queue_pipeline`，可直接在仓库路径下构建。

- 算子编译、部署、运行：PASS
- `priority` 模式：`max_error=0.0000 PASS`
- `fifo` 模式：`max_error=0.0000 PASS`
- 父子依赖约束检查：PASS
- `CopyIn → Compute → CopyOut` 流水线时序验证：PASS

## 910B 与 310B 的差异

| 项目 | 310B | 910B |
|------|------|------|
| 构建目标 | `TARGET=ascend310b bash scripts/build_ops.sh` | `TARGET=ascend910b bash scripts/build_ops.sh` |
| AI Core 数量 | 8 | 20+ |
| Kernel 并行模型 | 同左 | 同左，调度含跨任务依赖，使用一个 control block 保持时序确定 |
| 数据切分 | 由 tiling 的 `taskCount` / `queueDepth` / `computeLanes` 决定 | 同左，不在 Kernel 中写死树规模和层数 |

调度循环存在跨任务依赖，不能伪装成无依赖的逐元素并行算子；910B 的 Compute lane 并行度在 Kernel 内用 `COMPUTE_LANES` 表示，Host 侧通过 tiling 数据传入 `taskCount`、`queueDepth`、`computeLanes`。

## 迁移思路

1. 在 Host 侧生成 `parent`、`depth`、frontier 边界和待处理任务索引（见 `scripts/scheduler.py`）。
2. 将同一 frontier 的节点属性连续搬入 Local Memory，避免指针跳转和递归。
3. 在 Compute 阶段批量执行节点属性变换。
4. 使用输入/输出 TQue 组织 `CopyIn → Compute → CopyOut`，先从队列深度 2 的双缓冲开始。
5. 对比 Python reference 与设备输出，确认节点顺序、父子约束和任务结果一致。

## 注意事项

1. **使用 ASCII 工程路径**：本章目录及其构建路径已采用 ASCII 名称，可直接执行 `build_ops.sh`，无需复制到临时目录。
2. **输出大小动态传递**：`stage_end` 长度与 `parent` 一致，通过 tiling 数据的 `taskCount` 传给 Kernel，避免硬编码。
3. **910B 标量写 workspace 会触发 D-cache 错误**：Kernel 不建位置表，父子约束检查用在线扫描完成（任务规模 31，开销可忽略）。
4. **设备结果不可用 Python 模拟代替**：`end_to_end` 与 `stage_end` 必须在 910B 主机上补录，不能用本地 Python 结果代替设备输出。
5. **队列深度从 1 改为 2 后**：阶段时序应出现重叠，结果顺序和数值不能改变。

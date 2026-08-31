# tree_queue_lab 源码说明

本目录包含第 10 章的 Python reference 和 910B Ascend C 工程。Python 版本使用标准库模拟“树依赖释放 → 就绪队列调度 → `CopyIn / Compute / CopyOut`”；设备版本通过 `TreeQueuePipelineLite` 的 ACLNN 接口验证同一顺序和流水线时序。

## 输入数据

`data/input.json` 包含以下字段：

| 字段 | 含义 |
|------|------|
| `parent` | `parent[i]` 是节点 `i` 的父节点，根节点为 `-1` |
| `cost` | 节点 `i` 的 Compute 阶段耗时 |
| `copy_in` | 单个任务的 CopyIn 耗时 |
| `copy_out` | 单个任务的 CopyOut 耗时 |
| `queue_depth` | 可复用的缓冲槽数量，默认模拟双缓冲 |
| `compute_lanes` | 并行 Compute lane 数量 |

## 运行流程

```bash
python3 scripts/gen_data.py --num_nodes 31 --seed 10
python3 scripts/run_lab.py --data_dir data --output data/output.json
python3 scripts/verify_results.py data/output.json
```

`run_lab.py` 会输出 BFS 层级、FIFO 顺序、优先级顺序及两种顺序的流水线完成时间。`verify_results.py` 检查每个任务只执行一次、父节点先于子节点执行，并检查每个任务的三个流水线阶段没有倒序或重叠错误。

## 910B 设备流程

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
TARGET=ascend910b bash scripts/build_ops.sh
source scripts/env_custom_opp.sh
bash scripts/build_runner.sh
python3 scripts/gen_data.py --num_nodes 31 --seed 10 --output data
aclnn_runner/build/main_tree_queue_benchmark data priority
```

`TreeQueuePipelineLite` 接收 `parent`、`cost` 和 Host 侧生成的 `order`，输出 `stage_end` 和 `dependency_ok`。由于树调度包含跨任务依赖，设备 Kernel 使用一个 control block；双缓冲深度和 Compute lane 数作为 Tiling 参数传入。

## 算法对应关系

- `bfs_frontier`：使用 `deque` 维护 FIFO frontier，计算层级和依赖释放顺序。
- `priority_schedule`：使用 `heapq` 维护已就绪任务，优先处理剩余子树工作量更大的分支。
- `pipeline_schedule`：使用有限数量的缓冲槽和 Compute lane，记录三阶段的起止时间。

这个实现用于算法学习和结果校验；`910b_guide.md` 给出了将同一数据流映射到 Ascend C 的后续拆分思路。

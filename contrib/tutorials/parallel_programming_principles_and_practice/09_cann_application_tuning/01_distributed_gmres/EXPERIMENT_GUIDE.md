# 分布式 GMRES 单变量调优实验手册

## 1. 后端与目标

正式路径是 Device-resident GMRES：SpMV、Dot、Norm、AXPY、Scale 和残差更新由 Ascend C RTC kernel 执行，HCCL 直接规约 Device 结果；Host 只处理 Hessenberg/Givens 小标量和最终校验。目标是在相同矩阵、restart、容差与 Rank/Device 下，分别观察 NPU compute、HCCL、Host↔Device、同步和 solve 时间，并进行单变量调优。

## 2. 构建、基线与 msprof

```bash
cd src/dis_gmres
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDIS_GMRES_FORCE_STUB=OFF
cmake --build build -j
./build/bin/dis_gmres --matrix U2 --rank 0 --world-size 1 --device 0 \
  --warmup 0 --repeat 5 --restart 30 --max-iterations 300 --tolerance 1e-6

msprof --application="./build/bin/dis_gmres --matrix U2 --rank 0 --world-size 1 --device 0 --warmup 0 --repeat 5 --restart 30 --max-iterations 300 --tolerance 1e-6" \
  --output=./profiling/baseline --runtime-api=on --task-time=on
msprof --export=on --output=./profiling/baseline
```

多 Rank 采集时，把 `msprof --application=...` 作为每个 Rank 的启动命令，并使用项目 Rank Table；先用普通运行确认通信，再采集，避免把初始化失败误判为性能问题。不同 CANN 版本若导出选项不同，以 `msprof --help` 的同义选项为准，但应用参数保持不变。

## 3. 分析、单变量优化与验证

从 Summary/Timeline 对照程序计时字段，定位 SpMV、Dot、AXPY、Communication、Synchronization。只选择占比最大的一个阶段；例如仅切换 `--partition rows|nnz`、`--no-openmp` 或 `--unfused-vector-ops` 中一个选项，其他条件固定（`cgs` 当前未启用，只能作为预期失败/功能缺口验证）。优化后重新采集同样的 Profile。

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| Total time | | | |
| SpMV | | | |
| Dot | | | |
| AXPY | | | |
| Communication | | | |
| Residual/error | | | |
| Speedup | 1.00 | | |

通过条件：两次均满足 `residual <= 1e-6` 且迭代设置相同。练习：用 Timeline 证据说明所选瓶颈，并解释其他阶段为何不应同时调整。

## 4. Ascend 真机验收命令

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
cd contrib/tutorials/parallel_programming_principles_and_practice/09_cann_application_tuning/01_distributed_gmres/src/dis_gmres
DIS_GMRES_REQUIRE_REAL=1 bash scripts/build.sh

# 单 Rank：正确性与 baseline
bash scripts/run.sh --npus 1 --matrix U1 --orthogonalization mgs \
  --warmup 0 --repeat 3 --restart 30 --max-iterations 300 --tolerance 1e-6

# 多 Rank：用实际 HCCN IP 生成的 rank table
RANK_TABLE_FILE=/path/to/rank_table_4p.json \
  bash scripts/run.sh --npus 4 --matrix U2 --orthogonalization mgs \
  --warmup 0 --repeat 3 --restart 30 --max-iterations 300 --tolerance 1e-6

# Profile
rm -rf profiling/u2_device_baseline
msprof --application="./build/bin/dis_gmres --matrix U2 --rank 0 --world-size 1 --device 0 --orthogonalization mgs --warmup 0 --repeat 3" \
  --output=profiling/u2_device_baseline --runtime-api=on --task-time=on
msprof --export=on --output=profiling/u2_device_baseline
```

验收日志必须显示 `compute backend = Ascend C RTC Device GMRES`、目标 Device、迭代数、残差、SpMV/Dot/AXPY/Norm、HCCL、transfer、synchronization 和退出码。多 Rank 的 Host staging 仅用于最终输出/标量控制；全局向量通过 Device placement + HCCL AllReduce 形成。

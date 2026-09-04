# Xyce 线性求解器 Adapter 原型部署实验手册

## 1. 实验范围

当前工程是 Xyce 风格 workload/Adapter benchmark，不等同于完整 Xyce 可执行程序；但 `AscendDevice`（`LinearSolverKind` 枚举）线性求解路径已真实调用 Device-resident Ascend C GMRES。被加速的是稀疏线性求解热点，不是电路装配或完整 Xyce 进程。只有 `fetch_xyce.sh`/`build_xyce.sh` 成功并实际启动 Xyce 可执行文件，才能记录为完整 Xyce 部署。

## 2. 依赖、编译与运行

```bash
cd src/ascend_xyce
bash scripts/prepare_backend.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/xyce_benchmark --warmup 0 --repeat 5
```

在调度系统中，将上述命令写入作业脚本，显式设置工作目录、CPU/内存/时限和日志路径，然后用集群规定的 `sbatch`/`qsub` 提交；使用 `squeue`/`qstat` 检查状态并保存标准输出、错误日志和作业 ID。命令名称以所在集群调度器为准。

## 3. 正确性与交付物

检查进程退出码为 0、日志无初始化/求解失败、residual 或 relative error 满足程序阈值；比较相同输入下 Adapter 与 CPU reference。`results/xyce_benchmark.csv` 由当前作业生成且不随仓库提交，报告必须引用本次运行的日志。

| Job ID | Backend | Exit code | Iterations | Residual/error | Runtime | Status |
|---|---|---:|---:|---:|---:|---|
| | Ascend C RTC Device GMRES | | | | | |

练习：给出从提交、排队、运行到归档的状态链；区分“Adapter benchmark 成功”和“真实 Xyce 仿真成功”所需证据。

## 4. Ascend 真机验收命令

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
cd contrib/tutorials/parallel_programming_principles_and_practice/10_cluster_job_deployment/src/ascend_xyce
bash scripts/build.sh

# CPU baseline 与 NPU solver 由同一 benchmark、同一矩阵参数依次运行
DEVICE_ID=0 ./build/bin/xyce_benchmark --matrix U1 --warmup 0 --repeat 3 \
  --matrix-dir ./matrices --results-dir ./results --csv ./results/xyce_npu.csv

# profiler 证据
rm -rf profiling/xyce_u1
msprof --application="./build/bin/xyce_benchmark --matrix U1 --warmup 0 --repeat 3 --csv ./results/xyce_profile.csv" \
  --output=profiling/xyce_u1 --runtime-api=on --task-time=on
msprof --export=on --output=profiling/xyce_u1

# 如需验证完整 upstream Xyce 构建（不同集群依赖可能不同）
ASCEND_XYCE_FETCH_XYCE=1 ASCEND_XYCE_BUILD_XYCE=1 bash scripts/build.sh
```

验收时分别记录 CPU baseline、`Xyce Ascend Device GMRES (Ascend C RTC)`、Device ID、迭代数、残差、solution error、线性求解时间与 profiler kernel timeline。wrapper benchmark 通过不等于完整 upstream Xyce 仿真通过。

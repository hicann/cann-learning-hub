# 分布式 GMRES 调优实验指导

## 实验目标

使用同一份 `CSR1` 数据分别运行 1、2、4、8 个 rank，观察 restarted GMRES 中 SpMV、向量操作、HCCL 集合通信、数据搬移与同步的占比。实验不改变数学问题、矩阵格式或收敛阈值，每次只改变一个优化变量。

## 四组对照实验

1. 计算：固定单 Rank，用 `msprof` 分别观察 `gmres_spmv`、`gmres_dot`、`gmres_axpy`、`gmres_scale`；当前 dot 是单 AI Core 正确性 baseline，适合继续做分块归约优化。
2. 通信：固定设备数，记录 Device AllReduce 次数与同步时间。正式 Device 路径当前支持 MGS；`cgs` 会明确拒绝运行，直到实现可验证的 Device multi-dot，而不会静默回退 CPU。
3. 内存：检查 rank 日志中的 `rank_rows/rank_nnz` 与进程内存；CSR 只保留本 rank 行分片，HCCL send/receive buffer 按最大需求持久化并复用。
4. 调度：观察 `synchronization` 与 AllReduce 次数（当前 Device 路径不使用 AllGather，其计数为 0），并用 timeline 判断能否把下一步局部工作与集合通信重叠。当前版本使用单 ACL stream，适合作为异步双 stream 扩展的 baseline。

## 推荐记录表

| Matrix | NPU/rank | partition | orthogonalization | iter | residual | total | SpMV | Dot | AXPY | Norm | HCCL | transfer | sync | speedup |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

每组使用 `--warmup 0`（矩阵读取/分区与 communicator 初始化在 repeat 外只做一次；每次 solve 重建 solver 内的 RTC/Device 状态；warmup 只是丢弃额外 solver 调用，不是 warm cache 稳态指标）、`--repeat 10`，报告中同时给出平均值与离散程度。加速比使用完全相同参数的单设备 wall time 除以分布式 wall time；并行效率为 `speedup / N`。

分布式 wall time 的定义：每个 rank 输出自己的本地摘要（`[rank r] observed global residual / local solution relative error / local total / local SpMV / local HCCL comm / local AllReduce calls`），rank 0 的 `total_ms` 与各阶段耗时是 communicator 上的 MAX 聚合（关键路径 = 各 rank 的最大值）；SUM/world_size 的 rank 均值不是 wall time，不得作为 total 或 speedup 依据。AllReduce/AllGather 调用次数同样取各 rank 的最大单-rank 次数（一次 solver 的逻辑 collective 调用数），不是跨 rank 总和，避免随 world size 人为放大而干扰 MGS/CGS 对比。`residual` 是全局收敛残差，rank 摘要中标注为 observed global residual；`local solution relative error` 是本 rank 分片的真实解误差。

## 正确性门槛

- `converged = yes`；
- 显式相对残差 `||b-Ax||₂/||b||₂ <= 1e-6`；
- 解向量相对误差 `< 1e-3`；
- 所有 rank 正常退出；
- CANN 实验必须显示 `backend = ACL + HCCL`，`host stub` 结果只能用于功能验证。

## 进一步的学生任务

- 将单 AI Core `gmres_dot` 改为分块归约并用 FP32 reference 回归；
- 把 CSR values 压缩为 BF16、FP32 累加，对比误差和带宽；
- 为长尾矩阵比较 rows/nnz 两种划分的负载偏差；
- 使用双 stream、事件和 HCCL 异步调用实现计算通信重叠；
- 尝试 s-step/TSQR Arnoldi，量化集合通信次数下降与正交性损失。

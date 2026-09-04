# Dis-GMRES：CANN/HCCL 分布式 GMRES 调优实验

本项目把 restarted GMRES 扩展为多 Rank Ascend Device 求解器。每个 Rank 保存连续 CSR 行分片，CSR、解、残差与 Arnoldi basis 在一次 solve 内常驻 Device；SpMV、Dot、Norm、AXPY、Scale 和 residual update 由 `kernels/gmres_ops.cpp` 中的 Ascend C RTC kernel 执行，跨 Rank 标量规约由 HCCL 直接读取 Device buffer。

> 正式构建缺少 ACL、HCCL 或 `acl_rtc` 会直接失败。`DIS_GMRES_FORCE_STUB=ON` 仅用于无 CANN 主机上的 CPU 单元测试；Stub 不允许多 Rank，也不能生成 NPU 性能结论。Host 只执行小型 Hessenberg/Givens 控制和最终正确性比较，不执行正式路径的大向量数值 kernel。

## 工程结构

```text
Dis-GMRES/
├── CMakeLists.txt
├── include/                  # CSR、GMRES、HCCL、profiling 接口
├── src/
│   ├── csr_matrix.cpp        # 与 Ascend-GMRES 兼容的 CSR1 读写/生成
│   ├── npu_compute.cpp       # ACL RTC、持久 DeviceVector 与 kernel launch
│   ├── npu_gmres.cpp         # Device-resident Arnoldi/GMRES 主路径
│   ├── spmv.cpp              # Host Stub 与正确性测试 backend
│   ├── hccl_comm.cpp         # ACL runtime、HCCL、持久化 device buffer
│   ├── gmres.cpp             # 分布式 restarted GMRES
│   └── main.cpp              # benchmark、正确性、CSV 输出
├── tools/matrix_generator.cpp
├── scripts/
│   ├── build.sh
│   ├── run.sh
│   ├── run_scaling.sh
│   └── generate_rank_table.py
├── tests/test_core.cpp
├── matrices/                # CSR1 缓存，不提交大文件
├── results/                 # CSV 和 rank 日志
└── docs/experiment_guide.md
```

## 算法与优化点

### 分布式数据流

```text
local Device v --Device placement + HCCL AllReduce--> global Device v
                                  |
Device CSR rows ------------Ascend C SpMV----------> local Device w
                                                    |
Ascend C dot/norm (Device scalar) --HCCL AllReduce--+
                                                    |
                              Ascend C AXPY/Scale / x update
```

- `--partition rows`：行数均分，是负载不均衡 baseline。
- `--partition nnz`：按 CSR `row_ptr` 的累计 nnz 切分，默认用于长尾矩阵。
- `--orthogonalization mgs`：modified Gram-Schmidt，每轮包含多次标量 AllReduce。
- `--orthogonalization cgs`：communication-avoiding classical Gram-Schmidt（设计目标）。当前仓库的 Device 源码明确拒绝并输出 `Device CGS multi-dot kernel is not enabled`，不会回退 CPU；章测按运行结果探测能力，只有成功且通过正确性门槛时才允许进入对照。
- CSR 分片、Arnoldi buffer 和 HCCL device buffer 均复用；矩阵生成/读取、分区和 communicator 初始化位于 repeat 循环外只做一次；每次 solve 重建 solver 内的 RTC/Device 状态（NpuCompute、RTC 编译/加载、Device vectors），`--warmup N` 只是额外执行并丢弃 N 次完整 solver invocation，不是 warm cache 稳态指标，也不等于完整进程冷启动。
- 每个进程绑定一个 NPU device 和一条 ACL stream；HCCL 调用后显式同步，作为后续异步/双 stream 优化的 baseline。
- `world_size=1`（`--npus 1`）走最小快路径：`gather_spmv` 直接执行局部 SpMV，dot/norm 直接下载局部 Device 标量，旁路无实际通信的 identity AllReduce 和全局向量拼接，也不分配对应 Device buffer；该快路径只消除单卡无意义的 collective 开销，不改变数值流程，多卡路径仍保留显式 HCCL 与既有同步策略，同步/算子时序边界不变。

## 数据兼容性

数据规模、随机种子、矩阵类型和 GMRES 默认参数与本地 `Ascend-GMRES` 保持一致：

| Matrix | rows | cols | 目标原始 nnz | 类型 |
|---|---:|---:|---:|---|
| U1 | 100,000 | 100,000 | 1,000,000 | Uniform |
| U2 | 1,000,000 | 1,000,000 | 10,000,000 | Uniform |
| L1 | 100,000 | 100,000 | 1,000,000 | Long-tail |
| L2 | 1,000,000 | 1,000,000 | 10,000,000 | Long-tail |
| B1 | 100,000 | 100,000 | 1,000,000 | Block structured |
| B2 | 1,000,000 | 1,000,000 | 10,000,000 | Block structured |

`make_gmres_ready_matrix` 会加入或改写对角元，保证严格对角占优，因此最终 CSR nnz 可能略大于表中的原始目标。文件布局与参考项目完全相同：

```text
magic "CSR1"
int32 rows, int32 cols, int64 nnz
int32 row_ptr[rows+1]
int32 col_idx[nnz]
float values[nnz]
```

默认 GMRES：`restart=30`、`max_iterations=10000`、`tolerance=1e-6`，停止条件为显式 `||b-Ax||₂ / ||b||₂ <= 1e-6`。若把参考项目生成的 `.csrbin` 放到 `matrices/`，本项目会直接读取，不重新生成。

## 本地开发与基础验证

macOS/无 CANN 环境只构建 host stub；stub 只允许 `world-size=1`，防止把无通信的多进程结果误当 HCCL：

```bash
cd src/dis_gmres
DIS_GMRES_STUB=1 bash scripts/build.sh
bash scripts/run.sh --npus 1 --matrix U1 --warmup 0 --repeat 1 --no-openmp
```

构建脚本会运行 `ctest`。快速检查代码格式和脚本：

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
bash -n scripts/build.sh scripts/run.sh scripts/run_scaling.sh
python3 -m py_compile scripts/generate_rank_table.py
```

## 输出与正确性

每个 rank 输出自己的最小本地摘要（`[rank r] observed global residual / local solution relative error / local total / local SpMV / local HCCL comm / local AllReduce calls`），保留每 rank 的可审计证据；rank 0 再输出全局关键路径摘要。`last_result.residual` 是 solver 的全局收敛残差（每个 rank 相同），在 rank 摘要中标注为 observed global residual，不冒充 local residual；`local solution relative error` 是本 rank 分片相对真实解的误差。分布式耗时是 wall-time 性质：`total_ms` 及全部阶段耗时通过 communicator 上的 `HcclAllReduce(MAX)` 聚合（关键路径 = 各 rank 的最大值），并用于 speedup；rank 均值（SUM/world_size）不是 wall time，绝不冒充端到端耗时。调用次数（`AllReduce/AllGather calls`）同样取各 rank 的最大单-rank 次数（一次 solver 的逻辑 collective 调用数），不是跨 rank 总和。

rank 0 输出基础信息和如下分解（除计数外均为 MAX 聚合；计数为最大单-rank 次数）：

- `SpMV / Dot / AXPY / Norm / Givens`：局部计算；
- `HCCL communication`：集合通信从发起到 stream 完成；
- `ACL transfer`：Host/Device 输入输出搬移；
- `kernel launch`：Ascend C RTC kernel 的 Host 提交开销；纯 Device 执行时间应结合 `msprof` timeline 判断；
- `synchronization`：`aclrtSynchronizeStream` 等待时间；
- `AllReduce / AllGather calls`：输出两个计数；当前 Device 路径不使用 AllGather（全局向量经 Device placement + AllReduce 形成），其计数为 0。

结果写入 `results/dis_gmres.csv`，各进程日志写入 `results/rank_<rank>.log`。有效结果必须同时满足：

```text
backend = ACL + HCCL          # 远程多设备实验
converged = yes
residual <= 1e-6
solution relative error < 1e-3
所有 rank 退出码为 0
```

## 性能实验

先建立 baseline，再逐项改变变量：

```bash
# MGS 通信 baseline
bash scripts/run.sh --npus 4 --matrix U2 --orthogonalization mgs --warmup 0 --repeat 10

# Device MGS baseline（当前仓库的 CGS 会明确拒绝，不会回退 CPU）
bash scripts/run.sh --npus 4 --matrix U2 --orthogonalization mgs --warmup 0 --repeat 10

# 行均分 vs nnz 均分（L2 最明显）
bash scripts/run.sh --npus 4 --matrix L2 --partition rows --warmup 0 --repeat 10
bash scripts/run.sh --npus 4 --matrix L2 --partition nnz --warmup 0 --repeat 10

# 1/2/4/8 scaling
NPU_LIST=1,2,4,8 MATRIX=U2 WARMUP=0 REPEAT=10 bash scripts/run_scaling.sh
```

完整测试全部矩阵的 2/4/8 卡，并自动生成 CSV 和 Markdown 表格：

```bash
# 推荐为每种 communicator 准备对应的 rank_table_2p/4p/8p.json
RANK_TABLE_DIR=/path/to/rank-tables \
WARMUP=0 REPEAT=10 \
bash scripts/run_full_scaling.sh
```

脚本默认测试 `U1,U2,L1,L2,B1,B2` 和 `2,4,8` 卡，共 18 组配置，结果为：

```text
results/dis_gmres_scaling.csv
results/dis_gmres_scaling.md
results/dis_gmres_scaling_logs/<matrix>_<npus>p.log
```

如果不使用 rank table，2/4/8 卡可依赖 `run.sh` 的单机 root-info fallback：

```bash
WARMUP=0 REPEAT=10 bash scripts/run_full_scaling.sh
```

对于真实 HCCL rank table，不能把 8 卡 JSON 直接用于 2 卡或 4 卡；应分别生成：

```bash
python3 scripts/generate_rank_table.py --server-ip <server-ip> \
  --device-id 0 1 --device-ip <ip0> <ip1> --output rank_tables/rank_table_2p.json
python3 scripts/generate_rank_table.py --server-ip <server-ip> \
  --device-id 0 1 2 3 --device-ip <ip0> <ip1> <ip2> <ip3> \
  --output rank_tables/rank_table_4p.json
python3 scripts/generate_rank_table.py --server-ip <server-ip> \
  --device-id 0 1 2 3 4 5 6 7 --device-ip <ip0> <ip1> <ip2> <ip3> <ip4> <ip5> <ip6> <ip7> \
  --output rank_tables/rank_table_8p.json
```

脚本会从每组 rank 0 日志提取 rows/cols/nnz、迭代次数、残差、Single baseline、Distributed total、SpMV、HCCL、ACL transfer、同步、调用次数、误差和状态，并生成可直接粘贴到实验报告或 README 的表格。Single baseline 列直接复用每矩阵已解析的 1-rank `RESULT_TOTAL_MS`（不再搜索程序不存在的 `Single NPU:` 标签）；`status=pass` 前严格校验 baseline、distributed、residual、backend、solution error 等必填字段非空且数值格式有效，任何缺失或格式错误都会把该行降级为 `fail` 并使脚本非零退出，绝不写假 pass。解析/校验逻辑可通过无硬件检查验证：`bash scripts/run_full_scaling.sh --selfcheck`。

## Ascend 910B 实测结果（历史：旧 Host-compute + HCCL 实现，非当前 Device GMRES）

下表结果来自原实验工程已有测试记录，用于展示历史趋势。实际结果会受到硬件、软件版本和运行配置影响，不作为课程的固定标准答案。

本节数值属于整改前的旧 Host-compute + HCCL 实现（C++ reference compute + OpenMP 优化 + communication-avoiding CGS），不是当前 Device-resident GMRES（Ascend C RTC kernel + Device AllReduce）的实测。测试覆盖 U1/U2/L1/L2/B1/B2 六个矩阵和 2/4/8 卡，共 18 组配置；每个矩阵只测一次 Single baseline，所有卡数共享该基线。旧版本运行参数为每 rank 16 个 CPU helper thread、`DIS_GMRES_OMP_MIN_ELEMENTS=262144`、融合向量算子和 communication-avoiding CGS。所有配置均为 `backend=ACL + HCCL`、`status=pass`。时间单位为 ms，Speedup 定义为 `Single baseline / Distributed total`。注意：该历史版本的 Distributed total 是各 rank 的均值（rank mean），不是当前版本的 MAX 关键路径 wall time，不能把历史均值当作端到端 wall time 使用。

| Matrix | NPU | Threads/rank | Single baseline | Distributed total | HCCL communication | SpMV | Speedup | Error | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| U1 | 2 | 16 | 19.129057 | 19.538063 | 8.906129 | 2.215168 | 0.979066x | 3.620848e-07 | pass |
| U1 | 4 | 16 | 19.129057 | 16.292006 | 8.808275 | 1.587339 | 1.174138x | 3.620859e-07 | pass |
| U1 | 8 | 16 | 19.129057 | 21.390001 | 13.790679 | 3.600613 | 0.894299x | 3.621108e-07 | pass |
| U2 | 2 | 16 | 233.990051 | 154.624176 | 20.760078 | 64.896622 | 1.513282x | 3.645335e-07 | pass |
| U2 | 4 | 16 | 233.990051 | 143.348724 | 33.566620 | 35.184116 | 1.632313x | 3.636084e-07 | pass |
| U2 | 8 | 16 | 233.990051 | 126.988274 | 43.687115 | 26.551434 | 1.842611x | 3.636169e-07 | pass |
| L1 | 2 | 16 | 136.957077 | 112.483116 | 37.229870 | 11.265843 | 1.217579x | 7.182257e-07 | pass |
| L1 | 4 | 16 | 136.957077 | 91.816193 | 42.587967 | 9.391436 | 1.491644x | 7.243675e-07 | pass |
| L1 | 8 | 16 | 136.957077 | 117.064957 | 76.069054 | 17.276428 | 1.169924x | 7.364425e-07 | pass |
| L2 | 2 | 16 | 1125.075195 | 662.428467 | 80.596817 | 247.254944 | 1.698410x | 8.461028e-07 | pass |
| L2 | 4 | 16 | 1125.075195 | 589.755676 | 97.935974 | 110.216820 | 1.907697x | 8.572833e-07 | pass |
| L2 | 8 | 16 | 1125.075195 | 514.265991 | 164.219788 | 79.827782 | 2.187730x | 8.348097e-07 | pass |
| B1 | 2 | 16 | 50.121986 | 45.301388 | 19.529480 | 3.481659 | 1.106412x | 2.680080e-07 | pass |
| B1 | 4 | 16 | 50.121986 | 35.103085 | 19.521671 | 1.685282 | 1.427851x | 2.668504e-07 | pass |
| B1 | 8 | 16 | 50.121986 | 51.526711 | 37.715828 | 4.155228 | 0.972738x | 2.733200e-07 | pass |
| B2 | 2 | 16 | 345.893982 | 281.466400 | 37.651932 | 38.619873 | 1.228900x | 2.670109e-07 | pass |
| B2 | 4 | 16 | 345.893982 | 242.054367 | 63.657917 | 17.552639 | 1.428993x | 2.740948e-07 | pass |
| B2 | 8 | 16 | 345.893982 | 199.688873 | 90.198616 | 8.945267 | 1.732165x | 2.669805e-07 | pass |

### 正确性

18 组结果全部通过（历史记录）。显式 relative residual 位于 `3.58e-7` 到 `6.71e-7`，均低于 `1e-6`；solution error 位于 `2.67e-7` 到 `8.57e-7`。U 类矩阵收敛需要 6 次迭代，B 类需要 12 次，L 类需要 24 次；2/4/8 卡没有改变同一矩阵的迭代数量，说明旧实现中 CSR 分片、AllGather、AllReduce 和融合正交化没有破坏数值一致性。

### 最佳配置与总体效果

| Matrix | 最佳卡数 | Single baseline | 最佳 Distributed | Speedup | 耗时下降 | 并行效率 |
|---|---:|---:|---:|---:|---:|---:|
| U1 | 4 | 19.129 | 16.292 | 1.174x | 14.8% | 29.4% |
| U2 | 8 | 233.990 | 126.988 | 1.843x | 45.7% | 23.0% |
| L1 | 4 | 136.957 | 91.816 | 1.492x | 33.0% | 37.3% |
| L2 | 8 | 1125.075 | 514.266 | **2.188x** | **54.3%** | 27.3% |
| B1 | 4 | 50.122 | 35.103 | 1.428x | 30.0% | 35.7% |
| B2 | 8 | 345.894 | 199.689 | 1.732x | 42.3% | 21.7% |

18 组中有 15 组超过 `1x`。最高加速是 L2 的 8 卡 `2.188x`；U2 和 B2 的 8 卡分别达到 `1.843x` 和 `1.732x`。小矩阵的最优点是 4 卡：U1、L1、B1 分别为 `1.174x`、`1.492x`、`1.428x`。这说明最优卡数由单次计算量能否摊薄 collective 固定延迟决定，而不是卡数越多越好。

### 计算与通信扩展性

- U2 随 2/4/8 卡稳定加速，SpMV 从 `64.897` 降至 `26.551 ms`；HCCL 从 `20.760` 增至 `43.687 ms`，但计算缩短仍大于通信增长，因此 8 卡最佳。
- L2 的 SpMV 从 `247.255` 降至 `79.828 ms`，即使 8 卡 HCCL 达到 `164.220 ms`，完整求解仍从 `1125.075` 降到 `514.266 ms`，获得全场最高加速。
- B2 的 SpMV 从 `38.620` 降至 `8.945 ms`，8 卡 HCCL 增至 `90.199 ms`；当前仍有 `1.732x`，但继续增加卡数预计会进入通信主导区。
- U1/L1/B1 在 8 卡出现回退。U1-8 和 B1-8 分别只有 `0.894x` 和 `0.973x`；L1 从 4 卡的 `1.492x` 回落到 8 卡的 `1.170x`。三者的 8 卡 HCCL 分别为 `13.791/76.069/37.716 ms`，固定通信和同步成本已经超过进一步减少的局部计算。
- ACL transfer 没有随本地 SpMV 等比例下降：U2 为 `13.009--21.125 ms`，L2 约 `68.495--69.842 ms`，B2 为 `27.565--46.003 ms`。这部分来自旧实现中 host reference compute 与 device HCCL buffer 之间的 staging；当前版本已改为 Device-resident（CSR、`x`、`y` 在 solve 内常驻 Device），该历史结论不适用于当前实现。

`synchronization_ms` 是 HCCL timing 的组成部分，不能与 `hccl_ms` 再次相加。8 卡时 U2/L2/B2 的同步分别为 `19.275/78.498/46.327 ms`，说明下一步应重点减少 collective 临界路径和 rank 到达偏差。

### 本轮（旧版本）采用的优化方法

1. **线程与调度优化**：旧版本每个进程都可能创建覆盖整机的 OpenMP 团队，多 rank 严重过度订阅。本版本默认每 rank 最多 16 线程，并根据主机 CPU 容量降档；同时设置 `OMP_WAIT_POLICY=PASSIVE`、`GOMP_SPINCOUNT=0`、`KMP_BLOCKTIME=0`，避免 worker 在 HCCL 同步期间持续忙等。
2. **融合向量算子**：communication-avoiding CGS 的多个 local Dot 共用一个 parallel region；正交化 AXPY 和最终解更新也合并为融合循环，减少内存遍历和 OpenMP team launch。
3. **自适应并行阈值**：当向量长度低于 `262144` 时直接串行执行，避免小矩阵和小分片为短算子支付并行启动成本。
4. **通信优化**：零初值第一轮直接使用 `r=b`，删除一次零向量 SpMV、一次 AllGather 和一次 Norm AllReduce；显式残差复用已有 `||b||`。U 类每次 solve 的 collective 从 16 次 AllReduce、8 次 AllGather 降至 14/7，L 类为 52/27，B 类为 26/13。
5. **内存优化**：AllGather host send/receive buffer、全局输入 workspace 和 Arnoldi basis buffer 持久化复用，避免迭代区反复分配大向量。
6. **测量方法修正**：每个矩阵只运行一次 single-rank baseline，2/4/8 卡共同使用，消除了旧实验中不同卡数组之间 baseline 抖动造成的 speedup 失真。

### 优化前后对比

为了避免旧版 baseline 抖动影响结论，下表直接比较相同卡数下的 Distributed total：

| Matrix | NPU | 优化前 Distributed | 优化后 Distributed | 端到端缩短 |
|---|---:|---:|---:|---:|
| U1 | 4 | 697.988 | 16.292 | 42.84x |
| U2 | 8 | 1646.913 | 126.988 | 12.97x |
| L1 | 4 | 18508.430 | 91.816 | 201.58x |
| L2 | 8 | 18574.357 | 514.266 | 36.12x |
| B1 | 4 | 4290.529 | 35.103 | 122.22x |
| B2 | 8 | 6989.999 | 199.689 | 35.00x |

改善最大的部分不是数学迭代次数，而是消除了 OpenMP 过度订阅、短算子 parallel-region 开销和 rank 到达 HCCL 前的调度偏差。历史记录表明，旧版本优化后的实现已经从“所有配置均慢于单卡”变为“大矩阵在 8 卡达到 1.73x--2.19x，小矩阵在 4 卡达到 1.17x--1.49x”。

需要注意，该结论属于历史版本：旧版本中 `SpMV/Dot/AXPY/Norm` 仍是 C++ reference compute backend，ACL/HCCL 通信是真实 NPU 路径，报告的是旧版本完整应用配置的端到端加速，不应表述为纯 AI Core kernel speedup。当前版本已接入 Ascend C kernel（`kernels/gmres_ops.cpp` 的 `gmres_spmv/gmres_dot/gmres_axpy/gmres_scale/gmres_sub`，经 `npu_compute.cpp` RTC 编译/加载/launch），CSR、`x`、`y` 在 solve 内常驻 Device，局部数值计算在 Device 上执行。

## 附录：优化前 Ascend 910B 基线（历史：旧 Host-compute + HCCL 实现）

以下结果来自第二轮代码优化之前的远程 Ascend 910B `dis_gmres_scaling.csv`，用于定位问题和作为新版本对照。18 组配置全部显示 `backend=ACL + HCCL` 且 `status=pass`。时间单位为 ms；Speedup 定义为 `Single baseline / Distributed total`。

| Matrix | NPU | Iter | Single baseline | Distributed total | Speedup | SpMV | HCCL communication | ACL transfer | Solution error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U1 | 2 | 6 | 63.175 | 1441.713 | 0.044x | 89.187 | 169.878 | 62.531 | 3.62e-7 |
| U1 | 4 | 6 | 50.766 | 697.988 | 0.073x | 62.353 | 105.114 | 7.154 | 3.62e-7 |
| U1 | 8 | 6 | 48.646 | 411.675 | 0.118x | 31.631 | 57.908 | 8.665 | 3.62e-7 |
| U2 | 2 | 6 | 586.891 | 1716.021 | 0.342x | 172.156 | 186.506 | 47.874 | 3.64e-7 |
| U2 | 4 | 6 | 436.540 | 842.613 | 0.518x | 154.103 | 97.809 | 23.340 | 3.64e-7 |
| U2 | 8 | 6 | 406.113 | 1646.913 | 0.247x | 110.387 | 334.984 | 36.223 | 3.64e-7 |
| L1 | 2 | 24 | 264.050 | 11737.031 | 0.022x | 324.828 | 680.572 | 131.956 | 7.18e-7 |
| L1 | 4 | 24 | 281.900 | 18508.430 | 0.015x | 483.201 | 3123.421 | 130.782 | 7.24e-7 |
| L1 | 8 | 24 | 294.613 | 13540.314 | 0.022x | 324.797 | 3989.676 | 78.273 | 7.36e-7 |
| L2 | 2 | 24 | 4523.563 | 14332.820 | 0.316x | 604.022 | 704.753 | 210.965 | 8.35e-7 |
| L2 | 4 | 24 | 2337.410 | 7414.675 | 0.315x | 455.173 | 2066.614 | 76.973 | 8.57e-7 |
| L2 | 8 | 24 | 15368.075 | 18574.357 | 0.827x | 539.561 | 3340.674 | 178.287 | 8.35e-7 |
| B1 | 2 | 12 | 89.560 | 5201.979 | 0.017x | 227.804 | 642.589 | 70.191 | 2.68e-7 |
| B1 | 4 | 12 | 85.655 | 4290.530 | 0.020x | 177.432 | 986.062 | 36.618 | 2.67e-7 |
| B1 | 8 | 12 | 87.322 | 3834.750 | 0.023x | 163.178 | 944.184 | 32.565 | 2.73e-7 |
| B2 | 2 | 12 | 369.832 | 4275.973 | 0.086x | 182.053 | 373.599 | 82.861 | 2.66e-7 |
| B2 | 4 | 12 | 316.369 | 2497.772 | 0.127x | 153.148 | 444.526 | 46.177 | 2.74e-7 |
| B2 | 8 | 12 | 466.411 | 6989.999 | 0.067x | 279.655 | 1409.877 | 104.748 | 2.67e-7 |

### 结果分析

#### 正确性

18 组任务均正常结束，`solution_error` 在 `2.66e-7` 到 `8.57e-7` 之间，低于项目规定的 `1e-3` 门槛。U1/U2 为 6 次迭代，L1/L2 为 24 次，B1/B2 为 12 次；矩阵类型改变了 GMRES 的迭代数量，但没有破坏分布式结果一致性。原始 CSV 的 `residual` 列为空是旧脚本解析 `RESULT_RESIDUAL=` 时的字段分隔符错误，已在 `run_full_scaling.sh` 修正；上述表格使用日志中的解误差和 `pass` 状态，重新跑批量测试后会填充 residual 数值。

#### 总体扩展性

本轮测试没有出现大于 1 的 speedup。最佳配置是 L2 的 8 卡，`0.827x`，仍比单设备 baseline 慢约 1.21 倍；U2 的最佳配置是 4 卡，`0.518x`。小矩阵扩展性最差：U1 最佳为 8 卡 `0.118x`，L1 最佳为 4 卡 `0.015x`，B1 最佳为 8 卡 `0.023x`。B2 最佳为 4 卡 `0.127x`。

L2 的 8 卡行需要特别谨慎解读：该组 Single baseline 为 `15368.075 ms`，明显高于同一矩阵 2 卡/4 卡测试中的 `4523.563/2337.410 ms`，说明测试期间存在 CPU/内存带宽或系统调度抖动。正式报告应对单卡 baseline 独立重复多次并取中位数，同时确认没有残留 rank 进程、其他作业或 CPU 线程过度订阅；不能仅凭这一行得出 8 卡接近线性扩展的结论。

这不是 HCCL 正确性问题，而是旧实现的优化基线暴露的通信和调度成本：旧版本 `SpMV/Dot/AXPY/Norm` 在 C++ reference backend 执行，Arnoldi 每步需要 Host AllGather 全局向量，并通过 AllReduce 汇总内积/范数；每个集合通信还会产生 ACL Host/Device staging 和 stream synchronization。该结论只描述附录中的旧实现，不适用于当前 Device GMRES。对 U1/L1/B1 这类小矩阵，单卡计算量太小，固定通信延迟完全盖过了分片计算收益。

#### 通信与同步瓶颈

- U1：8 卡局部 SpMV 降到 `31.631 ms`，但 HCCL 仍有 `57.908 ms`，分布式总时间为 `411.675 ms`。
- U2：4 卡 HCCL 为 `97.809 ms`，8 卡反而升至 `334.984 ms`，因此 4 卡优于 8 卡。
- L1：4/8 卡 HCCL 分别为 `3123.421/3989.676 ms`，同步分别为 `3069.926/3866.893 ms`，是总时间异常升高的主要原因。
- L2：8 卡 HCCL `3340.674 ms`、同步 `3164.090 ms`，虽然局部 SpMV 只有 `539.561 ms`，但通信仍占主导。
- B1/B2：B1 的 8 卡 HCCL `944.184 ms`，B2 的 8 卡 HCCL `1409.877 ms`，均明显高于局部 SpMV。

旧实现中，U1/U2 使用 CGS 时每轮为 16 次 AllReduce、8 次 AllGather；L1/L2 为 55/28；B1/B2 为 28/14。后续可将这些 collective 与局部 SpMV 放到不同 stream，或继续实现 s-step/批量正交化，减少同步次数。

#### 下一步优化方向（历史版本的规划）

以下方向是历史版本当时提出的优化计划；其中第 1、2 项在当前版本已实现（Ascend C kernel 与 Device 常驻），第 3、4 项可作为后续练习：

1. 将 `src/spmv.cpp` 的 reference kernel 替换为真实 Ascend C CSR SpMV kernel，让局部计算真正运行在 AI Core。
2. 让 CSR、全局 `x` 和局部 `y` 常驻 device，消除当前每轮 H2D/D2H staging。
3. 使用异步 HCCL、双 stream 和 event，在下一轮局部计算与上一轮通信之间形成 overlap。
4. 针对 L1/L2 的长尾行，比较 rows partition 与 nnz-balanced partition，并记录各 rank 的最大/平均 nnz 比值。

因此，历史版本实测的主要结论是：分布式通信流程和结果正确，但旧计算后端尚未达到可扩展的 NPU kernel 实现；这些数据应作为历史 baseline，而不是当前 Device GMRES 的结果。

## 旧优化版本复现与参数调优（历史：针对旧 Host-compute + HCCL 实现）

针对附录基线中的 Dot、AXPY 和 synchronization 异常，旧版本加入以下默认优化（只描述历史版本，当前 Device 路径的向量操作由 Ascend C kernel 执行）：

- `run.sh` 默认每 rank 最多使用 16 个 OpenMP 线程，并根据主机可用 CPU 自动降档，保证总线程数不超过整机容量；这避免旧版每个 rank 占满整机，同时允许多 rank 获得合理的并行计算资源；
- 设置 `OMP_WAIT_POLICY=PASSIVE`、`GOMP_SPINCOUNT=0` 和 `KMP_BLOCKTIME=0`，rank 进入 HCCL 后 OpenMP worker 不再持续忙等；
- `DIS_GMRES_OMP_MIN_ELEMENTS=262144`，小于阈值的向量操作直接串行执行，避免短 Dot/AXPY 的 parallel-region 启动成本；
- CGS 的多个局部 Dot 共用一个 OpenMP parallel region，正交化 AXPY 和最终解更新也使用融合循环；
- AllGather host send/receive buffer 和全局输入 workspace 持久化复用；
- 零初值首轮直接使用 `r=b`，删除一次无意义的零向量 SpMV、一次 AllGather 和一次 Norm AllReduce；
- 显式残差复用已计算的 `||b||`，再减少一次 AllReduce。
- 全矩阵脚本对每个矩阵只测一次 single-rank baseline，2/4/8 卡共同使用该基线，避免旧结果中 L2-8 卡一类 baseline 抖动扭曲 speedup。

U1 的集合通信次数由旧版 16 次 AllReduce、8 次 AllGather 降为 14/7；L/B 矩阵同样每次 solve 减少两次 AllReduce和一次 AllGather。程序输出新增 `OpenMP threads per rank`、`OpenMP minimum elements` 和 `vector operations`，批量 CSV 也会保存这些参数。

远程重新测试前先保留旧结果：

```bash
cp results/dis_gmres_scaling.csv results/dis_gmres_scaling_before_opt.csv
cp results/dis_gmres_scaling.md results/dis_gmres_scaling_before_opt.md
git pull origin main
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
bash scripts/build.sh
```

先快速测试最有希望产生加速的 U2/L2/B2：

```bash
DIS_GMRES_THREADS_PER_RANK=16 \
DIS_GMRES_OMP_MIN_ELEMENTS=262144 \
bash scripts/run_full_scaling.sh \
  --matrices U2,L2,B2 \
  --npus-list 2,4,8 \
  --warmup 2 --repeat 5
```

确认正确性后再跑全部 18 组：

```bash
DIS_GMRES_THREADS_PER_RANK=16 \
DIS_GMRES_OMP_MIN_ELEMENTS=262144 \
WARMUP=3 REPEAT=10 \
bash scripts/run_full_scaling.sh
```

若服务器 CPU 核数充足，可分别测试每 rank 4/8/16 个 CPU helper thread，脚本会自动保证不超过整机 CPU 容量：

```bash
for threads in 4 8 16; do
  DIS_GMRES_THREADS_PER_RANK="${threads}" \
  FULL_SCALING_CSV="results/scaling_threads_per_rank_${threads}.csv" \
  FULL_SCALING_MD="results/scaling_threads_per_rank_${threads}.md" \
  bash scripts/run_full_scaling.sh --matrices U2,L2,B2 --npus-list 2,4,8 \
    --warmup 2 --repeat 5
done
```

如需严格保持单卡与多卡的总 CPU helper 数一致，可设置 `DIS_GMRES_TOTAL_THREADS=16`；该模式用于分离算法/HCCL 扩展性与额外 CPU 资源带来的收益，不是默认性能模式。

最终选择应以 `distributed_ms` 最小且 residual/solution error 达标为准，不能只选一个波动较高的 speedup 数字。

`run.sh` 在 `N>1` 时先用完全相同的矩阵/GMRES 参数运行单设备 baseline，再启动 N 个 rank，并把 baseline wall time传给 rank 0 计算 speedup。并行效率计算为 `speedup / N`。不要跨矩阵、精度、restart 或 tolerance 比较 speedup。

详细学生任务和记录表见 [实验指导](docs/experiment_guide.md)。

## 远程 Ascend NPU 完整测试流程

### 1. 拉取或更新代码

首次拉取：

```bash
git clone git@gitcode.com:maeveyixue/Dis-GMRES.git
cd Dis-GMRES
```

已有工作区：

```bash
cd Dis-GMRES
git pull origin main
```

### 2. 初始化 CANN 环境并检查设备

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
npu-smi info
cmake --version
g++ --version
```

部分版本使用以下路径：

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/bin/setenv.bash"
```

如果 toolkit 不在默认位置：

```bash
export ASCEND_HOME_PATH=/actual/path/to/ascend-toolkit/latest
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
```

确认 `npu-smi info` 中计划使用的设备均为 Healthy，设备数不少于 `--npus`。

### 3. 编译

```bash
bash scripts/build.sh
```

CMake 必须显示：

```text
Dis-GMRES ACL/HCCL backend=1 (1=real, 0=host stub)
```

若显示 `0`，检查 `ASCEND_HOME_PATH`、`acl/acl.h`、`hccl/hccl.h`、`libascendcl.so` 和 `libhccl.so`，不要继续记录 NPU 实验结果。

### 4. 单设备正确性

```bash
bash scripts/run.sh --npus 1 --matrix U1 --warmup 0 --repeat 3
```

确认 `backend = ACL + HCCL`、收敛和误差门槛均满足。注意当前 local compute 是 Ascend C RTC Device kernel（`kernels/gmres_ops.cpp`），这一步验证的是 Device GMRES/HCCL 数据流与 AI Core SpMV。

### 5. 准备单机 HCCL rank table

先查询服务器业务 IP 和每张 NPU 的 HCCN IP；不同机型命令可能略有差异：

```bash
hostname -I
for id in 0 1 2 3; do hccn_tool -i "${id}" -ip -g; done
```

用实际查询结果生成 4 卡配置：

```bash
python3 scripts/generate_rank_table.py \
  --server-ip 192.168.1.10 \
  --device-id 0 1 2 3 \
  --device-ip 192.168.100.101 192.168.100.102 192.168.100.103 192.168.100.104 \
  --output rank_table_4p.json
export RANK_TABLE_FILE="${PWD}/rank_table_4p.json"
```

单机场景也可省略 rank table，`run.sh` 会用 `HcclGetRootInfo` 和临时共享文件初始化；正式教学实验推荐 rank table，便于排查 rank/device 映射。

### 6. 多设备运行

```bash
bash scripts/run.sh \
  --npus 4 \
  --rank-table "${RANK_TABLE_FILE}" \
  --matrix U2 \
  --partition nnz \
  --orthogonalization mgs \
  --warmup 0 \
  --repeat 10
```

查看所有 rank：

```bash
for log in results/rank_*.log; do echo "===== ${log} ====="; tail -n 30 "${log}"; done
```

### 1. 结果验证与 profiling

```bash
tail -n 2 results/dis_gmres.csv
NPU_LIST=1,2,4,8 MATRIX=U2 WARMUP=0 REPEAT=10 bash scripts/run_scaling.sh
column -s, -t < results/scaling_U2.csv
```

使用 CANN profiler（参数以服务器安装版本的 `msprof --help` 为准）：

```bash
msprof --output=results/msprof_4p \
  --application="bash scripts/run.sh --npus 4 --rank-table ${RANK_TABLE_FILE} --matrix U2 --warmup 0 --repeat 3"
```

分析时重点看 HCCL timeline、Host/Device memcpy、同步空洞、不同 rank 的到达时间和 AllReduce 次数。当前版本有 Ascend C local kernel（`gmres_spmv`/`gmres_dot`/`gmres_axpy`/`gmres_scale`/`gmres_sub`），profiler 中应能看到本项目 SpMV 的 AI Core task；用 AI Core 利用率和算子 kernel time 验证计算优化。

## Git 工作流

```bash
git status
git add .
git commit -m "Implement distributed GMRES optimization experiment"
git push origin main
```

提交前不要加入 `matrices/*.csrbin`、`results/*.csv`、rank 日志或本地参考目录；`.gitignore` 已覆盖这些内容。

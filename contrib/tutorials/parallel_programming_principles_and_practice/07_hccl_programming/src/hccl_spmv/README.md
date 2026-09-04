# HCCL Distributed SpMV

本实验在参考 `SpMV` 单 NPU 工程的 CSR、缓存和 benchmark 规模基础上，演示多昇腾 NPU 的 HCCL 编程流程。每个 rank 持有相同的 CSR 元数据，按 row block 计算自己的 `A_r x`，然后使用 HCCL `Broadcast` 同步输入向量、使用 `AllGather` 汇总固定大小的输出块。

## 目录

```text
hccl-spmv/
├── CMakeLists.txt
├── include/{spmv.hpp,hccl_context.hpp}
├── src/{spmv_cpu.cpp,spmv_single_npu.cpp,spmv_distributed_hccl.cpp,hccl_context.cpp,main.cpp}
├── scripts/{build.sh,run.sh}
├── matrices/
└── results/
```

`U1/U2/L1/L2/B1/B2` 的 rows/cols/nnz 与参考工程保持一致：小规模为 100,000/100,000/1,000,000，大规模为 1,000,000/1,000,000/10,000,000。缓存采用参考工程兼容的 `CSR1` 二进制头（`int32 rows, int32 cols, int64 nnz`）。

## 本地构建（无 CANN）

```bash
cd hccl-spmv
HCCL_SPMV_STUB=1 bash scripts/build.sh
bash scripts/run.sh --matrix U1 --warmup 1 --repeat 3
```

无 CANN 时只有显式 `-DHCCL_SPMV_STUB=ON` 才能进行 Host 侧编译诊断；正式可执行程序会拒绝把 Stub 当作实验结果。

## Ascend 构建与单卡检查

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
bash scripts/build.sh
bash scripts/run.sh --matrix U1 --warmup 10 --repeat 100
```

CMake 自动查找 `acl/acl.h`、`hccl/hccl.h`、`ascendcl` 和 `hccl`；其他安装路径可设置 `ASCEND_HOME_PATH`。

## HCCL 多卡运行

先执行 `npu-smi info` 并 source CANN 环境。910B/CANN 9.0 建议使用 JSON rank table。先查询每张卡的 HCCN IP：

```bash
for d in 0 1 2 3 4 5 6 7; do hccn_tool -i "$d" -ip -g; done
```

查询到 IP 后可以使用仓库脚本生成 rank table（参数顺序必须对应设备 0 到 7）：

```bash
python3 scripts/generate_rank_table.py \
  --server-ip <管理网或服务端实际IP> \
  --device-ip <HCCN_DEVICE_IP_0> <HCCN_DEVICE_IP_1> <HCCN_DEVICE_IP_2> <HCCN_DEVICE_IP_3> \
              <HCCN_DEVICE_IP_4> <HCCN_DEVICE_IP_5> <HCCN_DEVICE_IP_6> <HCCN_DEVICE_IP_7> \
  --output /tmp/hccl_spmv_rank_table.json
```

单机 2 卡 JSON 示例（用实际 IP 替换占位符）：

```json
{
  "version": "1.0",
  "server_count": "1",
  "server_list": [{
    "server_id": "127.0.0.1",
    "device": [
      {"device_id": "0", "device_ip": "device0_ip", "rank_id": "0"},
      {"device_id": "1", "device_ip": "device1_ip", "rank_id": "1"}
    ]
  }]
}
```

8 卡时把 `device` 数组扩展到 `device_id/rank_id = 0...7`。保存为 `/tmp/hccl_spmv_rank_table.json`，所有 rank 共享该文件：

```bash
mkdir -p results
bash scripts/run.sh --npus 8 --rank-table /tmp/hccl_spmv_rank_table.json --matrix U1 --warmup 10 --repeat 100
```

如果系统没有安装 `hccn_tool`，也可以省略 `--rank-table`。`scripts/run.sh` 会在单机多卡场景下由 rank 0 生成共享 root-info 文件，其他 rank 自动读取：

```bash
bash scripts/run.sh --npus 8 --matrix U1 --warmup 10 --repeat 100
```

默认测试单机常见的 2/4/8 卡 scaling：

```bash
bash scripts/run_scaling.sh --matrix U1 --warmup 10 --repeat 100
```

若平台确认支持 6-rank communicator，可显式加入 6 卡；其中一个配置失败时脚本仍会继续后续测试：

```bash
bash scripts/run_scaling.sh --matrix U1 --npus-list 2,4,6,8 --warmup 10 --repeat 100
```

测试全部矩阵（每个矩阵分别运行 2/4/8 卡）：

```bash
for matrix in U1 U2 L1 L2 B1 B2; do
  bash scripts/run_scaling.sh --matrix "$matrix" --warmup 5 --repeat 20
done
```

结果保存在 `results/<matrix>_<n>npu.log`。`run.sh` 在全部子进程结束后按 rank 顺序回放每个 `results/rank_N.log`（带 `===== rank N =====` 标题），因此每个 case 的日志文件都保留所有 rank 的完整证据，不会只显示 rank 0 摘要；`results/rank_N.log` 同时保留本次运行的每 rank 原文。

调度器分别启动进程时，每个进程设置不同的 `RANK_ID`（0 到 N-1），并传入相同的 `RANK_SIZE`、`DEVICE_ID` 和 rank table。所有 rank 必须同时进入 Broadcast/AllGather；超时先检查 IP、rank 数、端口和设备映射。

## 分布式实现方式

程序采用单机多进程、一进程对应一个 rank 和一个 NPU 的执行模型。`scripts/run.sh` 为每个进程设置 `RANK_ID`、`RANK_SIZE` 和 `DEVICE_ID`；有 rank table 时调用 `HcclCommInitClusterInfo`，没有 `hccn_tool` 的单机环境则由 rank 0 调用 `HcclGetRootInfo`，通过临时文件把 `HcclRootInfo` 传给其他 rank，所有进程再调用 `HcclCommInitRootInfo` 创建 communicator。

一次 Distributed SpMV 的数据流如下：

1. 所有 rank 使用完全相同的 CSR 矩阵和输入向量种子。
2. rank 0 通过 `HcclBroadcast` 同步完整输入向量 `x`。
3. 按连续行块划分 CSR：`chunk = ceil(rows / world_size)`，rank `r` 负责 `[r * chunk, min(rows, (r + 1) * chunk))`。
4. 每个 rank 只计算自己的 `A_r x`，不足一个 chunk 的尾部用零填充。
5. 使用 `HcclAllGather` 收集所有固定大小的局部输出块。
6. 按 rank 顺序去掉 padding，恢复完整的 `y_distributed`。
7. 与 CPU CSR reference 比较最大逐元素相对误差，要求 `< 1e-6`。

HCCL 的 Broadcast 和 AllGather 缓冲区使用 ACL device memory。缓冲区在第一次迭代分配，之后复用，避免把反复 `aclrtMalloc/aclrtFree` 的成本计入 steady-state（warmup 是可复用 buffer/steady-state 语义，无需强制为 0）。当前输出字段含义：

- 每个 rank 先输出自己的本地摘要（`Local Total Time`、`Local HCCL Communication`、`Local Data Transfer` 等），保留每 rank 的可审计证据。
- 汇总耗时是 wall-time 性质，rank 0 通过 `HcclAllReduce(MAX)` 取各 rank 的最大值（关键路径）：`Total Time (MAX across ranks)`、`Kernel launch overhead (MAX)`、`Local SpMV launch-to-complete (MAX)`、`HCCL Communication (MAX)`、`Data Transfer (MAX)`、`Synchronization (MAX)`。MAX 才是分布式端到端 wall time；SUM/world_size 的 rank 均值会低估真实墙钟时间，不作为 total 或 scaling 结论。
- `HCCL Communication`：Broadcast、AllGather 及相应 stream synchronize 的时间。
- `Data Transfer`：collective 前后的 H2D/D2H 拷贝时间。
- `Local SpMV launch-to-complete`：本 rank 的 Ascend C local SpMV 从 kernel 提交到 stream 同步完成的时间（内部字段 `local_spmv_launch_to_complete_ms`），不是 Host 计算时间。
- `Kernel launch overhead`、`Synchronization`：launch 提交与 stream 同步的分项。
- `Total Time`：包含 collective、传输、局部 SpMV 与同步的端到端时间。

当前版本的 Broadcast/AllGather 与 local row-block SpMV 均在 Device 上执行：HCCL collective 直接处理 Device Buffer，local SpMV 使用 RTC 编译的 Ascend C FP32 Kernel。Host `spmv_cpu` 仅生成 correctness reference，不进入被测分布式计算路径。旧历史表的字段名（`HCCL collectives`/`ACL transfers`/`SpMV compute`/`Distributed time`）来自整改前的 Host-local 实现，只用于理解旧字段，不能作为当前 NPU 性能结论。

## Ascend 910B 实测结果（历史 Host-local + HCCL，非当前 Ascend C local SpMV）

下表结果来自整改前 Host-local 版本的旧实验工程记录，用于展示历史趋势。其 `SpMV compute` 列是旧 Host 计算的数值，不是当前 Ascend C Device kernel 实测；实际结果会受到硬件、软件版本和运行配置影响，不作为课程的固定标准答案。

测试环境为单机 8 张 Ascend 910B、CANN 9.0，使用 root-info 初始化；测试卡数为 2/4/8，`warmup=10`、`repeat=100`。时间单位均为 ms。每行的 speedup 使用同一次运行中测得的 Single baseline 除以 Distributed total，因此能够反映该次运行的端到端对比。所有 18 组结果误差均为 0。

| Matrix | NPU | Single baseline | Distributed total | HCCL collectives | ACL transfers | SpMV compute | Speedup | Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U1 | 2 | 3.245658 | 3.999525 | 1.518316 | 0.633024 | 1.783645 | 0.811511x | 0 |
| U1 | 4 | 3.896933 | 3.394089 | 1.736512 | 0.568609 | 0.998530 | 1.148153x | 0 |
| U1 | 8 | 5.086354 | 7.488879 | 6.030449 | 0.654001 | 0.694621 | 0.679188x | 0 |
| U2 | 2 | 104.579143 | 71.060812 | 7.124309 | 6.965360 | 55.288437 | 1.471685x | 0 |
| U2 | 4 | 104.700471 | 57.622436 | 22.482500 | 6.173255 | 28.200138 | 1.817009x | 0 |
| U2 | 8 | 116.085353 | 47.576786 | 24.928232 | 3.923332 | 17.716705 | 2.439958x | 0 |
| L1 | 2 | 4.182283 | 4.285251 | 1.257252 | 0.735958 | 2.188731 | 0.975972x | 0 |
| L1 | 4 | 4.328189 | 3.434453 | 1.954965 | 0.438698 | 0.968112 | 1.260227x | 0 |
| L1 | 8 | 3.960008 | 8.039647 | 6.560924 | 0.719490 | 0.649091 | 0.492560x | 0 |
| L2 | 2 | 118.723548 | 66.064154 | 7.539527 | 5.787619 | 51.695203 | 1.797095x | 0 |
| L2 | 4 | 101.269589 | 47.474876 | 11.057539 | 5.972956 | 29.618490 | 2.133120x | 0 |
| L2 | 8 | 107.742704 | 51.069537 | 31.519991 | 2.900380 | 15.869139 | 2.090144x | 0 |
| B1 | 2 | 3.401670 | 3.839478 | 1.287782 | 0.660564 | 1.821255 | 0.885972x | 0 |
| B1 | 4 | 3.768352 | 3.590779 | 2.064860 | 0.513348 | 0.927784 | 1.049453x | 0 |
| B1 | 8 | 4.027293 | 8.060664 | 6.816813 | 0.525661 | 0.622236 | 0.499623x | 0 |
| B2 | 2 | 99.932560 | 74.549710 | 15.318643 | 5.982635 | 52.327858 | 1.340482x | 0 |
| B2 | 4 | 127.104311 | 55.724947 | 15.258833 | 5.886794 | 33.330564 | 2.280923x | 0 |
| B2 | 8 | 114.111645 | 53.966385 | 31.784828 | 4.067025 | 17.164223 | 2.114495x | 0 |

### 正确性

历史记录中的误差仅说明旧实现的数据划分与拼接口径。真实 Ascend C + HCCL 路径必须在当前 NPU 环境重新运行，并以 `< 1e-6` 阈值、退出码和 Rank 日志重新验收。

### 小矩阵 U1/L1/B1（历史 Host-local 分析）

以下分析均针对上表的历史 Host-local 实现。小矩阵均为 10 万行、100 万 nnz。局部计算在增加 rank 后明显缩短，例如 U1 从 2 卡的 1.784 ms 降至 8 卡的 0.695 ms；但 HCCL 延迟从 1.518 ms 增至 6.030 ms，最终 8 卡总时间反而达到 7.489 ms。L1 和 B1 也呈现相同趋势。

4 卡是小矩阵的最佳点：U1、L1、B1 分别得到 1.148x、1.260x 和 1.049x。2 卡基本持平或略慢，8 卡则只有 0.49x 到 0.68x。说明小矩阵的有效计算量不足以摊薄 Broadcast、AllGather 和多进程同步的固定延迟。

### 大矩阵 U2/L2/B2

大矩阵均为 100 万行、1000 万 nnz，计算占比明显增加，多卡能够产生实际收益：

- U2 随卡数持续改善：2 卡 1.472x、4 卡 1.817x、8 卡 2.440x。8 卡局部计算降至 17.717 ms，即使 collective 达到 24.928 ms，仍获得本次测试的最高 speedup。
- L2 的最佳点是 4 卡 2.133x。8 卡计算降至 15.869 ms，但 HCCL 时间从 11.058 ms 增至 31.520 ms，因此总时间从 47.475 ms 回升到 51.070 ms。
- B2 同样在 4 卡达到最高 2.281x。8 卡计算继续下降，但 collective 翻倍到 31.785 ms，使 speedup 回落到 2.114x。

这说明 Distributed SpMV 的最优卡数取决于 `nnz/world_size` 带来的计算收益能否超过 collective 增长。增加卡数只会减少每个 rank 的 CSR row block；完整 `x` 仍需要广播，完整 `y` 仍需要聚合，因此通信量不会随局部 nnz 同比例下降。

### 通信开销分析

FP32 下，小矩阵的 `x` 和最终 `y` 各约 0.4 MB，大矩阵各约 4 MB。row partition 模型每轮至少需要一次完整 `x` Broadcast 和一次完整 `y` AllGather。随 rank 增加，SpMV 计算量近似按 `1/N` 下降，而 collective 的消息组织、同步和数据交换成本不会按 `1/N` 下降。

8 卡时 HCCL 在 U1/L1/B1 上约为 6.0--6.8 ms，已经远大于 0.62--0.69 ms 的局部计算；在 L2/B2 上约为 31.5--31.8 ms，也超过局部计算约两倍。U2 的 8 卡仍然获益，是因为其 single baseline 较高且计算缩短幅度足够大。

ACL transfer 在大矩阵上约 2.9--7.0 ms。历史实现中，本地计算还需要把 Broadcast 后的 `x` 从 device 回读到 host，并把局部 `y` 重新传到 device 参加 AllGather；当前版本已改为 Device-resident：`x` 和局部 `y` 常驻 device，Broadcast 后的 local SpMV 与 AllGather 之间不再经过 Host，因此该历史结论不适用于当前实现。

### 实验结论（历史 Host-local + HCCL）

以下结论全部针对上表的历史 Host-local 实现，不代表当前 Ascend C local SpMV 实测：

- HCCL communicator、Broadcast、AllGather、2/4/8 rank 协同和结果拼接已在旧实现上验证（历史记录）。
- 所有矩阵和所有卡数的误差均为 0，正确性通过（旧实现）。
- 小矩阵推荐 4 卡；8 卡由通信和同步主导，不适合当前数据规模。
- U2 推荐 8 卡，实测最高 2.440x；L2/B2 推荐 4 卡，分别达到 2.133x 和 2.281x。
- 6 卡在当前机器的 root-info communicator 初始化中返回参数错误，正式 scaling 使用硬件拓扑稳定支持的 2/4/8 卡。

当前实现已经把 Ascend C FP32 kernel 接入本地 row-block，并让 CSR、广播后的 `x` 与局部 `y` 位于 Device；后续性能工作的重点是减少剩余初始化传输，并评估通信计算重叠。

## 输出与复现实验

rank 0 输出矩阵尺寸、NPU 数量、分布式总时间（MAX 聚合的关键路径 wall time）、通信时间、SpMV 时间、加速比和误差（要求 `<1e-6`）；`results/rank_*.log` 保存各 rank 日志（含每 rank 本地 `Local ...` 摘要）。

课程复现流程：初始化当前 CANN 环境 → `npu-smi info` → 准备与 communicator size 匹配的 rank table → `HCCL_SPMV_REQUIRE_REAL=1 bash scripts/build.sh` → `bash scripts/run.sh --npus N --rank-table FILE`。上表属于原实验工程随附的历史记录，不要求学习者复现相同数值。

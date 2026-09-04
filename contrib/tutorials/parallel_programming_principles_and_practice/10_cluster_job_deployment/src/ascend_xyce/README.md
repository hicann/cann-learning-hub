# Ascend-Xyce：Xyce + Ascend GMRES 实验框架

Ascend-Xyce 提供 Xyce 风格 workload 到真实 Ascend Device GMRES 的 adapter。CPU 完成矩阵装配和参考解；稀疏求解阶段的 SpMV、Dot、Norm、AXPY、Scale 与残差更新通过 Ascend C RTC kernel 执行，Krylov 向量在一次 solve 内常驻 Device。

本项目的目标调用链是：

```text
Xyce
  |
  v
Linear Solver
  |
  v
GMRES
  |
  v
Ascend Device GMRES backend
  |
  v
FP32 CSR SpMV + Device vector kernels
  |
  v
Ascend AI Core
```

当前工程采用 wrapper/adapter 方式接入，不修改 Xyce 核心源码。Xyce 侧通过统一的线性求解接口调用：

```text
solve(A, b, x)
```

Device solver 复用第 07 章 `dis_gmres` 的 ACL RTC/HCCL 实现；CPU single/OpenMP solver 仅作为基线。该 wrapper 证明求解热点进入 NPU，但不等于完整 Xyce 二进制已经运行。

## 运行环境

本项目面向 Ascend NPU 机器运行，推荐环境：

- Device：支持 `dav-2201` 目标的 Ascend NPU
- CANN：9.0.0
- 编译器：支持 C++17 的 GCC/g++
- 构建工具：CMake 3.16 或更高
- CPU 多线程：OpenMP
- Git：用于拉取 Ascend-Xyce、Ascend-GMRES 和 Xyce 源码

进入机器后先检查：

```bash
npu-smi info
cmake --version
g++ --version
git --version
```

加载 CANN 环境：

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
```

如果机器上的 CANN 路径不同，请替换成实际路径下的 `set_env.sh`。

## 获取代码

第一次部署：

```bash
cd /mnt/workspace
git clone git@gitcode.com:maeveyixue/Ascend-Xyce.git
cd Ascend-Xyce
```

如果已经拉取过：

```bash
cd /mnt/workspace/Ascend-Xyce
git pull origin main
```

## 工程结构

```text
Ascend-Xyce/
├── CMakeLists.txt
├── README.md
├── third_party/
│   └── Xyce/
├── backend/
│   ├── gmres/
│   ├── spmv/
│   └── precision/
├── include/
├── src/
├── benchmark/
├── scripts/
├── matrices/
├── results/
└── docs/
```

关键目录说明：

- `third_party/Xyce/source`：Xyce 上游源码的可选目录，仅 `ASCEND_XYCE_FETCH_XYCE=1` 时拉取，不直接修改。
- `backend/gmres`：Ascend-GMRES backend 接入位置。
- `backend/spmv`：BF16 persistent CSR SpMV 说明。
- `backend/precision`：BF16-FP32 mixed precision 说明。
- `benchmark/xyce_benchmark.cpp`：统一 benchmark 入口。
- `docs/xyce_solver_flow.md`：Xyce-style adapter -> Device GMRES 原型流程说明（描述 wrapper 侧装配与求解调用，不是已完成的 upstream 调用链）。
- `results/xyce_benchmark.csv`：最终 benchmark 结果。

## 构建

直接运行：

```bash
bash scripts/build.sh
```

构建脚本默认**只构建 wrapper benchmark，不拉取 Xyce 上游源码**：

1. 查找或拉取 `Ascend-GMRES`。
2. 构建 Ascend-Xyce wrapper benchmark（默认不执行 `fetch_xyce.sh`）。
3. 仅当 `ASCEND_XYCE_FETCH_XYCE=1` 时才把 Xyce 源码拉到 `third_party/Xyce/source`；仅当 `ASCEND_XYCE_BUILD_XYCE=1` 时才尝试构建完整 upstream Xyce。

`Ascend-GMRES` 查找顺序：

1. 环境变量 `ASCEND_GMRES_DIR`
2. 相邻目录 `../Ascend-GMRES`
3. 自动 clone `git@gitcode.com:maeveyixue/Ascend-GMRES.git`

如果 Ascend-GMRES 已经在固定路径，可以显式指定：

```bash
export ASCEND_GMRES_DIR=/mnt/workspace/Ascend-GMRES
bash scripts/build.sh
```

构建成功后，主程序位于：

```text
build/bin/xyce_benchmark
```

如果只想构建 wrapper benchmark，而不拉取 Xyce 上游源码（默认行为，也可显式声明）：

```bash
ASCEND_XYCE_FETCH_XYCE=0 bash scripts/build.sh
```

如果需要尝试完整构建 upstream Xyce 源码（必须先显式拉取）：

```bash
ASCEND_XYCE_FETCH_XYCE=1 ASCEND_XYCE_BUILD_XYCE=1 bash scripts/build.sh
```

Xyce 完整源码构建依赖较多；当前 benchmark 使用 wrapper/adapter 路径，跑通的是 “Xyce-style adapter -> Device GMRES 原型链”：CPU 装配类 Xyce 的稀疏线性系统，`XyceLinearSolverAdapter::solve` 调用 Device GMRES。完整的 upstream Xyce 整链（真实 netlist 解析、Newton 迭代、时域步进、Trilinos/Epetra 接口、`TYPE=ASCEND` 线性求解器注册）仍未验证，也不能由 wrapper 结果证明。

## 运行 Benchmark

正式运行：

```bash
bash scripts/run.sh
```

该命令默认等价于：

```bash
./build/bin/xyce_benchmark \
  --warmup 0 \
  --repeat 10 \
  --matrix-dir matrices \
  --results-dir results \
  --csv results/xyce_benchmark.csv
```

只运行单个矩阵，例如 U2：

```bash
./build/bin/xyce_benchmark --matrix U2 --warmup 0 --repeat 10
```

未知矩阵名在打开/截断 CSV 之前就会被拒绝并返回非零退出码，已有 CSV 不会被覆盖：

```bash
./build/bin/xyce_benchmark --matrix UNKNOWN --warmup 0 --repeat 10
echo $?   # 2
```

无 NPU 的 host 聚焦校验（只检查矩阵名与支持列表的匹配，不运行 solver、不写 CSV）：

```bash
./build/bin/xyce_benchmark --check-matrix U1; echo $?   # 0
./build/bin/xyce_benchmark --check-matrix NOPE; echo $? # 2
```

运行结束后查看 CSV：

```bash
cat results/xyce_benchmark.csv
```

或：

```bash
column -s, -t < results/xyce_benchmark.csv | less -S
```

## 验收门禁

- **残差门禁**：每个 solver 必须收敛（相对残差 `< 1e-6`），否则 `XyceApplicationWrapper::run` 抛错、benchmark 退出非零。
- **解误差门禁**：`kSolutionErrorTolerance = 1.0e-3`（与 dis_gmres 章节的 solution-error 门禁一致）。每个 solver 结果在返回/写 CSV 前必须同时满足“相对构造的 `expected_solution` 的真实解误差”与（提供 CPU reference 时）“相对 CPU reference 的误差”都低于该阈值；任何一项超限都明确报错并返回非零，不会留下 pass 数据。CPU reference 不是唯一门禁：`expected_solution` 始终独立校验，CPU reference（提供时）也独立校验，避免 reference 掩盖真实解误差。
- **行数门禁**：运行结束后校验 CSV 行数：每个选中的矩阵必须恰好有 3 行 solver 结果（CPU single / CPU OpenMP16 / Ascend Device），只写表头或行数不符都会返回非零。
- **CSV 精度**：正确性字段（`final_residual`、`solution_error_vs_cpu`）以 scientific 格式输出且至少 9 位有效数字；timing 字段保持 fixed 6 位小数，两种格式互不污染。

## Solver 对比

每个矩阵会运行三个版本：

- `Xyce CPU single GMRES`
- `Xyce CPU OpenMP16 GMRES`
- `Xyce Ascend Device GMRES (Ascend C RTC)`

GMRES 参数固定：

- restart：`30`
- max iteration：`10000`
- tolerance：`1e-6`
- 停止条件：`||b-Ax|| / ||b|| < 1e-6`

## 测试矩阵

支持六个 CSR 矩阵：

| Matrix | rows | cols | nnz 级别 | 类型 |
|---|---:|---:|---:|---|
| U1 | 100,000 | 100,000 | 1,000,000 | Uniform |
| U2 | 1,000,000 | 1,000,000 | 10,000,000 | Uniform |
| L1 | 100,000 | 100,000 | 1,000,000 | Long-tail |
| L2 | 1,000,000 | 1,000,000 | 10,000,000 | Long-tail |
| B1 | 100,000 | 100,000 | 1,000,000 | Block structured |
| B2 | 1,000,000 | 1,000,000 | 10,000,000 | Block structured |

矩阵格式：

```text
row_ptr
col_idx
values
```

程序读取顺序：

1. 优先读取当前工程 `matrices/*.csrbin`。
2. 如果不存在，尝试读取相邻 `../Ascend-GMRES/matrices`。
3. 如果仍然不存在，自动生成兼容 Ascend-GMRES 的 synthetic CSR 矩阵。

## CSV 字段说明

`results/xyce_benchmark.csv` 每个 solver 一行，主要字段如下：

- `matrix`：矩阵名称。
- `rows` / `cols` / `nnz`：矩阵规模。
- `solver`：当前 solver。
- `total_simulation_ms`：Xyce wrapper 总模拟时间。
- `matrix_assembly_ms`：矩阵/RHS 组装时间。
- `linear_solver_ms`：GMRES 线性求解时间。
- `iterations`：GMRES 迭代次数。
- `final_residual`：最终相对残差（scientific，≥9 位有效数字）。
- `converged`：是否收敛。
- `speedup_vs_cpu_single`：相对 CPU single 的总模拟加速比。
- `solution_error_vs_cpu`：相对 CPU single 解的误差（scientific，≥9 位有效数字）；`run()` 同时校验相对构造 `expected_solution` 的真实解误差并执行 1e-3 门禁。
- `spmv_ms`：GMRES 内部 SpMV 总时间。
- `dot_ms`：Dot product 总时间。
- `axpy_ms`：AXPY 总时间。
- `norm_ms`：Norm 总时间。
- `givens_ms`：Givens/Hessenberg 更新时间。
- `residual_ms`：显式 residual check 时间。
- `other_ms`：其他 solver overhead。
- `device_transfer_ms`：Host/Device 向量搬移（H2D/D2H）总时间。
- `hccl_ms`：HCCL collective（AllReduce/AllGather）从调用到同步完成的累计时间。
- `kernel_launch_ms`：Device kernel 启动（launch 调用本身）总时间。
- `synchronization_ms`：Stream 同步等待累计时间；它与相应算子阶段及 `hccl_ms` 存在包含关系，不能直接相加做总时间分解。
- `avg_spmv_per_iteration`：平均每次迭代 SpMV 时间。
- `avg_dot_per_iteration`：平均每次迭代 Dot 时间。
- `avg_axpy_per_iteration`：平均每次迭代 AXPY 时间。

## 最新实验结果

> **历史 Host 原型数据，非 NPU 实测。** 以下表格仅保留用于复现实验结构和比较 CPU 实现；所有 “Host prototype” 数值均来自 CPU/OpenMP。它们不能证明 HBM、NPU persistent kernel 或 Ascend 加速效果，也不能代表当前 `AscendDevice` 求解器的 NPU 实测；当前版本的 NPU 结果必须在昇腾真机重新采集，并同时记录设备型号、CANN 版本和 profiler 证据。

运行命令：

```bash
bash scripts/run.sh
```

### 总体时间

单位：ms。`Host prototype speedup` 使用 `total_simulation_ms` 计算。下表结果来自原实验工程已有测试记录，用于展示典型性能趋势；实际结果会受到硬件、软件版本和运行配置影响，不作为课程的固定标准答案。

| Matrix | Iter | CPU single sim | CPU OpenMP16 sim | Host prototype optimized sim | Host prototype speedup vs CPU single | Host prototype vs OpenMP16 | Host prototype solution error |
|---|---:|---:|---:|---:|---:|---:|---:|
| U1 | 6 | 55.724 | 32.468 | 29.722 | 1.87x | 1.09x | 0.001812 |
| U2 | 6 | 1263.038 | 365.314 | 390.688 | 3.23x | 0.94x | 0.001817 |
| L1 | 24 | 215.730 | 62.989 | 90.853 | 2.37x | 0.69x | 0.002152 |
| L2 | 24 | 3872.288 | 847.908 | 876.536 | 4.42x | 0.97x | 0.002149 |
| B1 | 12 | 64.742 | 36.779 | 22.138 | 2.92x | 1.66x | 0.000697 |
| B2 | 12 | 757.500 | 218.736 | 234.437 | 3.23x | 0.93x | 0.000701 |

所有矩阵均成功收敛，最终 residual 满足 `1e-6` 要求。Host prototype 解相对 CPU single 解的误差约为 `7e-4` 到 `2.2e-3`，符合 BF16 storage + FP32 accumulation 的预期。

### Linear Solver 时间

单位：ms。

| Matrix | CPU single linear | CPU OpenMP16 linear | Host prototype optimized linear | Host prototype linear speedup vs CPU single |
|---|---:|---:|---:|---:|
| U1 | 48.478 | 24.906 | 21.192 | 2.29x |
| U2 | 1070.211 | 202.853 | 215.011 | 4.98x |
| L1 | 210.594 | 55.352 | 82.403 | 2.53x |
| L2 | 3716.399 | 693.209 | 704.738 | 5.27x |
| B1 | 59.324 | 31.194 | 15.330 | 3.87x |
| B2 | 693.179 | 154.864 | 158.004 | 4.39x |

从 linear solver 时间看，Host prototype optimized 相比 CPU single 全部超过 `2x`，大矩阵 U2/L2 分别达到 `4.98x` 和 `5.27x`。

### Host prototype Persistent CSR 统计

| Matrix | cold start(ms) | warm(ms) | FP32 bytes | BF16 bytes | compression ratio |
|---|---:|---:|---:|---:|---:|
| U1 | 3.476 | 15.317 | 9,199,956 | 6,999,968 | 1.314285 |
| U2 | 44.330 | 158.482 | 91,999,932 | 69,999,950 | 1.314286 |
| L1 | 3.778 | 78.581 | 9,199,916 | 6,999,938 | 1.314285 |
| L2 | 46.268 | 657.213 | 91,999,940 | 69,999,956 | 1.314286 |
| B1 | 3.934 | 13.231 | 9,200,004 | 7,000,004 | 1.314286 |
| B2 | 42.205 | 126.819 | 92,000,004 | 70,000,004 | 1.314286 |

BF16 persistent CSR 的压缩比稳定在 `1.314x`。对 100 万行、约 1100 万 nnz 的大矩阵，CSR footprint 从约 `92MB` 降到约 `70MB`，这直接降低了 SpMV 的访存压力。

## 结果解读

### 1. Host prototype 明显快于 CPU single

按总时间看，Host prototype optimized 相比 CPU single：

- U1：`1.87x`
- U2：`3.23x`
- L1：`2.37x`
- L2：`4.42x`
- B1：`2.92x`
- B2：`3.23x`

除 U1 这种较小规模、固定开销占比更高的矩阵外，其余矩阵都超过 `2x`。如果只看 GMRES linear solver 本体，U1 也达到 `2.29x`。这说明在旧 Host 原型实现中，核心线性求解阶段已经稳定超过 CPU single；该结论只属于历史 Host prototype 版本，不涉及当前 `AscendDevice` Device 路径。

### 2. SpMV 仍然是 GMRES 最核心瓶颈

CPU single 中，SpMV 占比非常高：

- U2：`624.461ms / 1070.211ms = 58.35%`
- L2：`2283.970ms / 3716.399ms = 61.46%`
- B2：`304.036ms / 693.179ms = 43.86%`

Residual check 中也包含显式 `A*x`，本质上仍然是一次 SpMV 相关操作。以 U2 为例：

- CPU single SpMV：`624.461ms`
- CPU single Residual：`327.254ms`
- 两者合计：`951.715ms`
- 占 linear solver：约 `88.9%`

这说明 Xyce/GMRES 的主要性能压力不是 Givens 或 Hessenberg 小矩阵操作，而是稀疏矩阵向量乘及相关 residual 检查。

### 3. Host prototype 优化的主要收益来自 SpMV

Host prototype optimized 相比 CPU single 的 SpMV 加速非常明显：

| Matrix | CPU single SpMV | Host prototype SpMV | SpMV speedup |
|---|---:|---:|---:|
| U1 | 22.605 | 3.346 | 6.76x |
| U2 | 624.461 | 94.919 | 6.58x |
| L1 | 102.792 | 13.568 | 7.58x |
| L2 | 2283.970 | 371.259 | 6.15x |
| B1 | 25.338 | 3.058 | 8.29x |
| B2 | 304.036 | 39.240 | 7.75x |

平均到每次迭代：

- U2：CPU single `104.077ms/iter`，Host prototype `15.820ms/iter`
- L2：CPU single `95.165ms/iter`，Host prototype `15.469ms/iter`
- B2：CPU single `25.336ms/iter`，Host prototype `3.270ms/iter`

该结果只说明旧 Host 原型中的持久化数据结构与 BF16 存储改变了 CPU 侧访存量，不能据此推断 NPU 带宽或传输收益；当前版本已接入 Ascend C Device kernel（`LinearSolverKind::AscendDevice`），NPU 结论必须以真机实测为准。

### 4. CPU OpenMP16 是强 baseline，Host prototype 与其接近，部分矩阵超过

CPU OpenMP16 对 SpMV、dot、axpy、norm 都做了 16 线程并行，因此是很强的 CPU baseline。

按总模拟时间对比：

- U1：Host prototype 比 OpenMP16 快约 `1.09x`
- U2：Host prototype 约为 OpenMP16 的 `0.94x`
- L1：Host prototype 约为 OpenMP16 的 `0.69x`
- L2：Host prototype 约为 OpenMP16 的 `0.97x`
- B1：Host prototype 比 OpenMP16 快约 `1.66x`
- B2：Host prototype 约为 OpenMP16 的 `0.93x`

结论是：Host prototype 优化版已经稳定超过 CPU single；和 CPU OpenMP16 相比，整体接近，在 U1/B1 上超过，在 U2/L2/B2 上基本同一量级。L1 上 OpenMP16 更快，主要因为该矩阵迭代数较多，Dot/AXPY 在 Host prototype 版本中占比更高。

### 5. BLAS-1 没有完全成为瓶颈，但会影响不同矩阵的 Host prototype 表现

Host prototype optimized 中，Dot 和 AXPY 占比：

- U2：Dot `4.20%`，AXPY `4.95%`
- L2：Dot `15.48%`，AXPY `10.76%`
- B2：Dot `21.25%`，AXPY `18.63%`
- L1：Dot `35.34%`，AXPY `33.73%`

这说明对 U2 这种 uniform 大矩阵，SpMV 仍然占主导，BLAS-1 不是主要瓶颈；但对 L1/B2 这类矩阵，Dot/AXPY 的占比会明显上升。整体上，BLAS-1 没有破坏 Host prototype 相比 CPU single 的加速，但会影响它相对 OpenMP16 的表现。

### 6. Givens/Hessenberg 开销可以忽略

所有矩阵中 Givens 时间都在 `0.003ms` 到 `0.042ms` 量级，占比接近 `0%`。这说明 restarted GMRES 中的小规模 Hessenberg 更新不是性能瓶颈，主要成本来自大向量操作和 SpMV。

### 7. Xyce wrapper 总时间中存在 assembly/adapter 开销

`total_simulation_ms` 比 `linear_solver_ms` 更大，因为它包含：

- RHS 构造
- Xyce matrix adapter
- solver prepare
- linear solve 调用包装

例如 U2：

- CPU single simulation：`1263.038ms`
- CPU single linear solver：`1070.211ms`
- Host prototype simulation：`390.688ms`
- Host prototype linear solver：`215.011ms`

因此，评估 GMRES backend 本身时应重点看 `linear_solver_ms` 和 profiling breakdown；评估 Xyce 应用整体时看 `total_simulation_ms`。

## 最终结论（历史 Host prototype 版本）

本次 Host-only 结果只能证明：

- Ascend-Xyce 三个 solver 均能成功收敛。
- Host prototype optimized 版本相对 CPU single 在总时间上达到 `1.87x` 到 `4.42x` 加速。
- Host prototype optimized 版本相对 CPU single 在线性求解阶段达到 `2.29x` 到 `5.27x` 加速。
- GMRES 的核心瓶颈来自 SpMV 和 residual check。
- Host 原型中的 BF16 persistent CSR 相比所选 CPU single baseline 达到约 `6x` 到 `8x`；这不是 NPU 加速比。
- BF16 CSR footprint 压缩比稳定在 `1.314x`。
- Givens/Hessenberg 不是瓶颈。
- CPU OpenMP16 是强 baseline，Host prototype optimized 与其整体接近，并在部分矩阵上超过。

上述结论只属于历史 Host prototype 版本，不能证明 Ascend NPU 已参与计算。当前版本已实现真实设备 backend：`LinearSolverKind::AscendDevice` 经 `XyceLinearSolverAdapter::solve` 调用 `dis_gmres::distributed_gmres`（CMake 目标 `xyce_npu_solver` 直接编译第 07 章 `dis_gmres` 源码，vendored `backend/gmres/Ascend-GMRES` 只是 CPU baselines/legacy Host prototype 源码）；NPU 加速结论必须以当前真机运行时 backend/device 输出、数值误差和 profiler 时间线重新验收。

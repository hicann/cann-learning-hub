# Ascend GMRES 稀疏线性求解器实验框架

> 在本课程仓库中，本目录作为 `ascend_xyce` 的内部求解器库使用；请从上层 `src/ascend_xyce` 按课程脚本构建和运行，不在本目录单独生成 benchmark 和矩阵工具。

本项目实现 restarted GMRES 稀疏线性方程求解器，用于比较：

- `CPU single-thread GMRES`
- `CPU OpenMP16 GMRES`
- `HostPrototype optimized GMRES`

测试目标是验证在迭代科学计算场景中，`CSR persistent`、`BF16 CSR compression` 和 `Host prototype BLAS-1 / SpMV acceleration` 能否显著提升 GMRES 求解性能。

> **Legacy 边界（课程上下文）**：本目录是随 Ascend-Xyce vendored 的独立实验项目，保留其原始 Host prototype 实验与历史数据。08 章 wrapper 中 active 的 `AscendDevice` 求解器并不编译本目录源码：CMake 目标 `xyce_npu_solver` 直接编译第 07 章 `dis_gmres` 源码，`XyceLinearSolverAdapter::solve` 调用 `dis_gmres::distributed_gmres`。本目录仅提供 CPU baselines 与 legacy Host prototype 源码（如 `spmv/npu_spmv.cpp`），其文件名不是 active `AscendDevice` 的 NPU 证据；其 `run.sh` 默认 `--warmup 3` 也只属于 vendored 项目本身，不是当前 wrapper 的默认（wrapper 默认 `ASCEND_XYCE_WARMUP=0`）。本目录内的历史数值均为 Host 原型数据，非当前 NPU 实测。

GMRES 求解的问题形式：

```text
A * x = b
```

其中 `A` 使用 CSR 稀疏矩阵格式：

```text
row_ptr
col_idx
values
```

## 运行环境

推荐在 Host CPU/OpenMP 机器上运行。

环境要求：

- Host prototype：Host CPU/OpenMP
- CANN：9.0.0
- C++ 标准：C++17
- 构建工具：CMake 3.16+
- CPU 多线程：OpenMP

进入机器后，先检查环境：

```bash
npu-smi info
cmake --version
g++ --version
```

加载 CANN 环境变量：

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
```

如果 CANN 安装路径不同，请替换成实际路径下的 `set_env.sh`。

## 获取代码

```bash
git clone git@gitcode.com:maeveyixue/Ascend-GMRES.git
cd Ascend-GMRES
```

如果代码已经存在：

```bash
cd Ascend-GMRES
git pull origin main
```

## 构建

推荐使用脚本：

```bash
bash scripts/build.sh
```

等价命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

构建完成后，主程序位于：

```text
build/bin/gmres_benchmark
```

矩阵生成工具位于：

```text
build/bin/uniform_generator
build/bin/longtail_generator
build/bin/block_generator
```

## 运行 Benchmark

正式运行：

```bash
bash scripts/run.sh --warmup 3 --repeat 10
```

也可以直接运行：

```bash
./build/bin/gmres_benchmark \
  --warmup 3 \
  --repeat 10 \
  --matrix-dir matrices \
  --results-dir results \
  --csv results/gmres_benchmark.csv
```

只测试单个矩阵：

```bash
./build/bin/gmres_benchmark --matrix U2 --warmup 3 --repeat 10
```

第一次运行会自动生成矩阵并缓存到 `matrices/`。之后再次运行会直接读取缓存。

## GMRES 参数

当前 benchmark 使用标准 restarted GMRES：

- restart：`30`
- max iteration：`10000`
- convergence tolerance：`1e-6`
- 停止条件：`||b-Ax|| / ||b|| < 1e-6`

## 测试矩阵

本项目沿用之前 Ascend-SpMV 项目的 synthetic CSR 矩阵。

| Matrix | rows | cols | nnz 级别 | 类型 |
|---|---:|---:|---:|---|
| U1 | 100,000 | 100,000 | 1,000,000 | Uniform |
| U2 | 1,000,000 | 1,000,000 | 10,000,000 | Uniform |
| L1 | 100,000 | 100,000 | 1,000,000 | Long-tail |
| L2 | 1,000,000 | 1,000,000 | 10,000,000 | Long-tail |
| B1 | 100,000 | 100,000 | 1,000,000 | Block structured |
| B2 | 1,000,000 | 1,000,000 | 10,000,000 | Block structured |

## 输出文件

benchmark 结果写入：

```text
results/gmres_benchmark.csv
```

查看结果：

```bash
column -s, -t < results/gmres_benchmark.csv | less -S
```

## Solver 输出字段

终端会输出每个矩阵下三种 solver 的结果：

- `iterations`：GMRES 迭代次数
- `residual`：最终相对残差
- `converged`：是否收敛
- `time(ms)`：完整求解平均耗时
- `speedup_vs_cpu_single`：相对 CPU single 的加速比
- `solution_error_vs_cpu`：相对 CPU single 解向量的误差

Host prototype 版本还会输出：

- `cold_start_ms`：persistent CSR 初始化时间，包括 CSR 准备和 BF16 转换
- `warm_ms`：GMRES 迭代区域耗时
- `fp32_bytes`：FP32 CSR footprint
- `bf16_bytes`：BF16 CSR footprint
- `compression_ratio`：FP32/BF16 压缩比

## GMRES 算子级 Profiling

benchmark 会对完整 GMRES 求解过程做算子级时间拆分：

- `SpMV`：Arnoldi 过程中 `w = A * v`
- `Dot`：Hessenberg 构造中的内积
- `AXPY`：`w = w - h*v`、解向量更新等向量加法
- `Norm`：L2 norm 计算
- `Givens`：Givens rotation、Hessenberg 更新、上三角回代
- `Residual`：显式相对残差检查
- `Other`：未归入上述类别的 solver overhead

CSV 中对应字段包括：

```text
cpu_single_spmv_ms
cpu_single_dot_ms
cpu_single_axpy_ms
cpu_single_norm_ms
cpu_single_givens_ms
cpu_single_residual_ms
cpu_single_other_ms

cpu_openmp16_spmv_ms
cpu_openmp16_dot_ms
cpu_openmp16_axpy_ms
cpu_openmp16_norm_ms
cpu_openmp16_givens_ms
cpu_openmp16_residual_ms
cpu_openmp16_other_ms

host_prototype_spmv_ms
host_prototype_dot_ms
host_prototype_axpy_ms
host_prototype_norm_ms
host_prototype_givens_ms
host_prototype_residual_ms
host_prototype_other_ms
```

同时输出：

```text
*_avg_spmv_per_iteration
*_avg_dot_per_iteration
*_avg_axpy_per_iteration
```

这些字段用于观察每次 GMRES 迭代中 SpMV、Dot、AXPY 的平均成本。

## 最新实验结果

运行命令：

```bash
bash scripts/run.sh --warmup 3 --repeat 10
```

### 总体求解性能

单位：ms。

下表结果来自原实验工程已有测试记录，用于展示典型性能趋势。实际结果会受到硬件、软件版本和运行配置影响，不作为课程的固定标准答案。

| Matrix | Iter | CPU single | CPU OpenMP16 | HostPrototype optimized | Host prototype speedup vs CPU single | Host prototype vs OpenMP16 | Host prototype solution error |
|---|---:|---:|---:|---:|---:|---:|---:|
| U1 | 6 | 49.069 | 30.520 | 25.700 | 1.91x | 1.19x | 0.001812 |
| U2 | 6 | 1233.249 | 212.963 | 214.878 | 5.74x | 0.99x | 0.001817 |
| L1 | 24 | 203.708 | 58.444 | 40.898 | 4.98x | 1.43x | 0.002152 |
| L2 | 24 | 4648.904 | 730.353 | 690.150 | 6.74x | 1.06x | 0.002149 |
| B1 | 12 | 61.991 | 26.758 | 25.964 | 2.39x | 1.03x | 0.000697 |
| B2 | 12 | 710.382 | 151.593 | 153.877 | 4.62x | 0.99x | 0.000701 |

所有矩阵均成功收敛，最终 residual 达到 `1e-6` 要求。Host prototype 解向量相对 CPU single 的误差在 `7e-4` 到 `2.2e-3` 之间，符合 BF16 storage + FP32 accumulation 的预期。

### Host prototype Persistent CSR 统计

| Matrix | cold start(ms) | warm(ms) | FP32 bytes | BF16 bytes | compression ratio |
|---|---:|---:|---:|---:|---:|
| U1 | 5.990 | 19.035 | 9,199,956 | 6,999,968 | 1.314285 |
| U2 | 45.675 | 159.878 | 91,999,932 | 69,999,950 | 1.314286 |
| L1 | 3.624 | 38.603 | 9,199,916 | 6,999,938 | 1.314285 |
| L2 | 48.037 | 643.372 | 91,999,940 | 69,999,956 | 1.314286 |
| B1 | 3.947 | 23.364 | 9,200,004 | 7,000,004 | 1.314286 |
| B2 | 45.781 | 119.970 | 92,000,004 | 70,000,004 | 1.314286 |

BF16 CSR 使矩阵存储 footprint 从约 `92MB` 降到约 `70MB`，压缩比稳定在 `1.314x`。这部分收益直接作用于 GMRES 中反复调用的 SpMV。

## Profiling 结果分析

### 1. SpMV 是最主要瓶颈

CPU single 上，SpMV 占比非常高：

- U2：734.929ms / 1233.249ms，占 `59.59%`
- L2：2930.341ms / 4648.904ms，占 `63.03%`
- B2：310.567ms / 710.382ms，占 `43.44%`

Residual check 里也包含一次显式 `A*x` 计算，因此它本质上仍然大量受 SpMV 影响。以 U2 为例，CPU single 的 residual check 是 `380.823ms`，占 `30.88%`。如果把 Arnoldi SpMV 和 residual check 一起看，U2 上超过 `90%` 的时间都和矩阵向量乘相关。

这说明 GMRES 的核心瓶颈确实来自稀疏矩阵乘法，而不是 Givens rotation 或 Hessenberg 小矩阵操作。

### 2. Host prototype 优化主要收益来自 SpMV

Host prototype optimized 相比 CPU single，在大矩阵上的 SpMV 时间下降明显：

| Matrix | CPU single SpMV | Host prototype SpMV | SpMV speedup |
|---|---:|---:|---:|
| U2 | 734.929ms | 94.832ms | 7.75x |
| L2 | 2930.341ms | 377.807ms | 7.76x |
| B2 | 310.567ms | 36.653ms | 8.42x |

平均到每次迭代：

- U2：CPU single `122.488ms/iter`，Host prototype `15.805ms/iter`
- L2：CPU single `122.098ms/iter`，Host prototype `15.742ms/iter`
- B2：CPU single `25.714ms/iter`，Host prototype `3.054ms/iter`

这正好说明 persistent CSR + BF16 compression 对迭代 SpMV 是有效的：矩阵不在迭代中重复传输，values 用 BF16 压缩后减少访存压力，SpMV 成本显著下降。

### 3. Host prototype 总体性能达到目标

Host prototype optimized 相比 CPU single 的总求解加速：

- U1：`1.91x`
- U2：`5.74x`
- L1：`4.98x`
- L2：`6.74x`
- B1：`2.39x`
- B2：`4.62x`

除 U1 这种较小规模矩阵外，其余矩阵都超过 `2x`。重点大矩阵 U2/L2 分别达到 `5.74x` 和 `6.74x`，说明目标已经实现。

### 4. CPU OpenMP16 是强 baseline，Host prototype 与其接近或超过

CPU OpenMP16 对 SpMV、dot、axpy、norm 都有并行加速，因此它是一个很强的 CPU baseline。

对比 OpenMP16：

- U1：Host prototype `25.700ms`，OpenMP16 `30.520ms`，Host prototype 更快
- U2：Host prototype `214.878ms`，OpenMP16 `212.963ms`，基本持平
- L1：Host prototype `40.898ms`，OpenMP16 `58.444ms`，Host prototype 更快
- L2：Host prototype `690.150ms`，OpenMP16 `730.353ms`，Host prototype 更快
- B1：Host prototype `25.964ms`，OpenMP16 `26.758ms`，基本持平且略快
- B2：Host prototype `153.877ms`，OpenMP16 `151.593ms`，基本持平

这符合实验预期：Host prototype optimized 版本必须超过 CPU single，CPU OpenMP16 不强制要求超过。但从结果看，Host prototype 在 U1/L1/L2/B1 上已经超过 OpenMP16，在 U2/B2 上也基本持平。

### 5. BLAS-1 没有成为新的主要瓶颈

Host prototype optimized 中，Dot 和 AXPY 的占比在不同矩阵上有所变化：

- U2：Dot `4.52%`，AXPY `5.01%`
- L2：Dot `13.86%`，AXPY `10.29%`
- B2：Dot `20.27%`，AXPY `18.04%`

这些操作有开销，但没有像 CPU single 的 SpMV 那样成为压倒性瓶颈。特别是 U2/L2 这种大规模矩阵中，Host prototype 的主要耗时仍集中在 SpMV 和 residual check，符合 GMRES 稀疏求解器的计算特征。

### 6. Givens / Hessenberg 开销可以忽略

所有矩阵中 Givens 时间都接近 `0.01ms` 量级，占比约 `0.00%` 到 `0.03%`。这说明 restarted GMRES 中的小规模 Hessenberg 更新不是性能瓶颈，真正需要关注的是大向量操作和稀疏矩阵操作。

## 最终结论

本次结果证明：

- 三个 solver 均能稳定收敛到 `1e-6`。
- GMRES 的主要瓶颈来自 SpMV，以及包含 SpMV 的 residual check。
- Host prototype optimized 版本通过 persistent CSR、BF16 matrix storage 和 FP32 accumulation 显著降低了 SpMV 成本。
- Host prototype optimized 相比 CPU single 在 U2/L2 大矩阵上达到 `5x` 以上加速。
- Host prototype optimized 与 CPU OpenMP16 相比整体接近，在部分矩阵上已经超过 OpenMP16。
- BF16 压缩带来稳定 `1.314x` CSR footprint reduction，同时最终 residual 仍满足收敛要求。

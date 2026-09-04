# MPI SpMV 实验手册

## 1. 目标、代码与通信流程

本实验在 **CPU** 上执行局部 CSR SpMV，用 MPI 完成按 CSR 行划分计算任务、向量广播和结果汇集。初始化使用 `MPI_Init`、`MPI_Comm_rank`、`MPI_Comm_size`；每个进程（rank）获得一个行区间，`MPI_Bcast` 广播完整 `x`，`MPI_Gatherv` 按不等行数收集 `local_y`。

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &process_count);
MPI_Bcast(x->data(), static_cast<int>(x->size()), MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Gatherv(local_y.data(), local_rows, MPI_FLOAT, /* ... */, MPI_COMM_WORLD);
```

## 2. 编译与单机实验

```bash
cd src/mpi_spmv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
mpirun -np 1 ./build/spmv_mpi
mpirun -np 2 ./build/spmv_mpi
mpirun -np 4 ./build/spmv_mpi
```

单机多进程用于验证 Rank 划分、Broadcast/Gatherv 和正确性，不能替代多节点扩展性实验。

## 3. 多节点实验

创建 `hostfile`：

```text
node01 slots=2
node02 slots=2
```

两台节点使用相同程序路径和 MPI 环境后执行：

```bash
mpirun -np 4 --hostfile hostfile ./build/spmv_mpi
```

## 4. 验证、记录与分析

以 `-np 1` 的 CPU 结果为 reference，比较 2/4 个进程（rank）的最大相对误差，阈值 `1e-6`。逐进程记录通信域规模（size）、row range、compute/communication/total time；负载不均衡定义为 `(max_rank_compute-avg_rank_compute)/avg_rank_compute`。

| Ranks | Compute (ms) | Communication (ms) | Total (ms) | Imbalance | Relative error | Pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | | | | | 0 | |
| 2 | | | | | | |
| 4 | | | | | | |

练习：比较按行数与按 nnz 划分，解释通信占比为何随 Rank 数增长。参考答案见 `answer/`。

# 4.3 课后练习答案

## 选择题

**1. B** — `asc.adv.register_matmul(pipe, workspace, matmul, tiling)` 的核心作用是初始化 Matmul 对象，将 TPipe、workspace 内存和 Tiling 参数与 Matmul 对象关联，完成内部资源的分配和配置。

**2. B** — `host.MultiCoreMatmulTiling.set_dim(USE_CORE_NUM)` 设置多核并行的核数。该参数决定了 Tiling 算法如何将 M×N 矩阵切分到多个核上。

## 填空题

**3.** `matmul.iterate_all(c_global)` — iterate_all 接口一次性完成单核负责的所有 baseM×baseN 分块的计算，并将结果写入 c_global。它是最高层的计算接口，内部自动处理 K 轴迭代和分块循环。

**4.** 创建Matmul对象（声明A/B/C/Bias类型）。完整7步流程为：创建GlobalTensor → 创建TPipe → 创建Matmul对象 → register_matmul → set_tensor_a/b → set_tail + iterate_all → end。

**5.** `single_core_m` 表示每个核负责的M维度大小（多核切分层级），`base_m` 表示单次mmad计算的M大小（单核迭代层级）。当 M=1000、`single_core_m=64` 时，M方向共有 15 个满块（1000//64=15）和 1 个尾块（尾块大小为 1000 - 15×64 = 40）。

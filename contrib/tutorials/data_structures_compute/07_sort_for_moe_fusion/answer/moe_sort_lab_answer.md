# MoE 排序融合实验解析

## 测试题答案

1. B：固定大小最小堆遍历 E 个 expert 时，复杂度约为 `O(E log K)`。
2. B：本实验为 910B 设置 `BLOCK_DIM=16`，310B 构建时切换为 8。
3. B：`sortedOrder` 保存按 expert 分桶后的 token-slot pair 顺序，Permute 据此搬运 token。
4. `blockDim`。
5. `ascend910b`。
6. `权重`。

## 实践任务要点

### 任务 1

QuickSort 为每个 token 建立分数和 expert id 数组，使用显式栈执行分区；HeapSort 先把全部 expert 建成最大堆，再重复取出堆顶。两者都使用“分数更高优先、分数相同时 id 更小优先”的 tie-break，最后将前 `K=2` 个结果写入 indices 和 probs。

### 任务 2

`build_ops.sh` 默认 `TARGET=ascend910b`，并把 Host 源码中的 `BLOCK_DIM` 设置为 16；指定 `TARGET=ascend310b` 时改为 8。每个 Kernel 只读取 tiling 中的 `tokensPerCore` / `rowsPerCore`，所以 Kernel 主体不需要写芯片分支。

### 任务 3

- `route`：TopK、QuickSort 或 HeapSort 选择 expert 的 kernel 时间。
- `order+copy`：将 indices 从 device 拷回 Host、构造 sortedOrder，再拷回 device 的时间。
- `permute`：按 sortedOrder 搬运 token、权重和 token id。
- `unpermute`：按 sortedIndices 将 expert 输出加权还原到原 token 顺序。
- `end_to_end`：从路由开始到还原结束的完整路径时间。

只比较 `route` 会忽略排序顺序构造和 token 搬运；真实融合算子应关注完整路径的端到端代价。

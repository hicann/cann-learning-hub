# 01.03 课后实践参考解析

本目录中的 `run_case.py` 是第 7 题的参考实现。它把参数组织和结果判定拆成两个函数：命令列表保持每个参数为独立元素，判定函数同时检查 Shape、Tile 长度和精度结果，不能只依据进程退出码。

本页用于完成实验后的核对。性能数值会随设备、频率和系统负载变化，因此应核对参数、精度和计时口径，不应要求耗时与某个示例逐字一致。

## 1. 新配置的数据切分

题目参数为 `N=32768`、`blockDim=4`、`tileCount=16`，数据类型仍为 `float32`。

### 推导

```text
blockLength = 32768 / 4 = 8192 个元素
tileLength  = 8192 / 16 = 512 个元素
tileBytes   = 512 × 4 = 2048 Byte
```

三道参数检查依次为：

- `32768 % 4 == 0`；
- `8192 % 16 == 0`；
- `2048 % 32 == 0`。

### 结论

这是一组合法配置。每个逻辑 Block 处理 8192 个元素，并依次处理 16 个真实 GM Tile，每个 Tile 包含 512 个 `float32` 元素。

### 实验现象

在已经完成构建的目录中，可运行：

```bash
./vector_add --length 32768 --block-dim 4 --tile-count 16 \
  --seed 7 --warmup 1 --iterations 5
```

成功时 `METRIC` 行至少应满足：

- `length=32768`、`block_dim=4`、`tile_count=16`；
- `tile_length=512`、`buffer_num=1`；
- `correctness=PASS`，且 `max_abs_error<=1e-6`；
- `timing_scope=launch_plus_sync`。

### 常见错误

- 修改了 `N`，却仍拿默认配置的 `blockLength=2048` 和 `tileLength=256` 对照；
- 认为 `tileCount=16` 会自动启用 16 缓冲。实验 01 默认是单缓冲，`buffer_num=1`；
- 只看到进程退出码为 0，没有核对 `METRIC` 中的 Shape、Tile 长度和精度字段。

## 2. `tileCount` 与队列槽位

### 推导

`tileCount` 参与数据切分：

```text
tileLength = blockLength / tileCount
```

`Process` 的循环次数也是 `tileCount`，所以它表示每个 Block 真正覆盖的 GM Tile 数。`TQue` 的 queue depth 固定为 `kQueueDepth=1`；`kBufferNum` 只传给 `InitBuffer`，决定 Local Memory 中每个队列可轮换使用的物理缓冲块数量。

### 结论

两者是独立概念：

- `tileCount=16`：每个 Block 需要完成 16 次 Tile 数据处理；
- `buffer_num=1`：每个队列只有 1 个 LocalTensor 缓冲槽，即单缓冲；
- 改变队列槽位数不应改变 `Process` 覆盖的 Tile 数和最终数学结果。

对本题，三个队列的教学统计量为：

```text
queue_bytes = 3 × buffer_num × tileLength × sizeof(float)
            = 3 × 1 × 512 × 4
            = 6144 Byte
```

### 实验现象

输出同时出现 `tile_count=16` 和 `buffer_num=1` 是正确现象，并不矛盾。默认配置则是 `tile_count=8`、`buffer_num=1`：每核处理 8 个真实 Tile，但仍是单缓冲。

### 常见错误

- 把循环次数写成 `tileCount / buffer_num`，从而在双缓冲时只处理一半数据；
- 用 `buffer_num` 计算 `tileLength`；
- 把 `queue_bytes` 当作全局输入输出总字节数。它只是课程程序报告的核内队列缓冲统计量。

## 3. 非法长度 `N=32769`

### 推导

```text
32769 % 4 = 1
```

因此它首先违反 `N % blockDim == 0`。若仍用整数除法计算，每个 Block 只能得到 `32769 / 4 = 8192` 个元素，4 个 Block 合计覆盖 32768 个元素，最后一个元素没有归属。

### 结论

Host 必须在分配和 Kernel 启动前拒绝该 Shape，并返回退出码 2。当前实验只支持均匀分核，不包含尾核处理。

### 实验现象

运行：

```bash
./vector_add --length 32769 --block-dim 4 --tile-count 16 \
  --seed 7 --warmup 1 --iterations 5
```

应看到包含 `length must be divisible by blockDim=4` 的第一条错误信息，进程退出码为 2，且不会打印成功的 `correctness=PASS` 指标。

### 常见错误

- 期待 Kernel 自动处理余数；
- 在 Host 中向下取整后仍返回成功，这会隐藏最后一个元素未计算的问题；
- 只检查结果前 32768 个元素，因而漏掉尾元素。

## 4. 遗漏 `GetBlockIdx()` 偏移

### 推导

正确的 GM 起点应为：

```text
blockOffset = blockLength × GetBlockIdx()
```

若把 `blockOffset` 固定为 0，4 个逻辑 Block 的 `GlobalTensor` 都绑定到 `[0, 8192)`，而不是分别绑定到四个互不重叠的区间。

### 结论

首个 Block 区间会被多个逻辑 Block 重复读写，形成写冲突；`[8192, 32768)` 没有被正确写入。即使各 Block 写入首区间的数值相同，这个程序仍然是错误的，因为全局覆盖范围不完整且存在并发写入。

### 实验现象

CPU Golden 会检查全部 `N` 个元素，所以最终应出现 `correctness=FAIL` 或精度用例失败。错误通常呈现为“前一段可能看似正确，后续大段错误”，不能因为前几个输出值正确就判断 Kernel 正确。

### 常见错误

- 只抽查数组开头，未检查完整输出；
- 误以为多核执行相同计算只会浪费性能，不会影响正确性；
- 在 `CopyIn` 中临时叠加 Block 偏移，却没有让 `CopyOut` 使用一致的地址体系。

## 5. 一个 Tile 的队列状态变化

### 推导

输入 Tile 和输出 Tile 分别经过以下状态：

| 阶段 | 输入队列 `inQueueX/inQueueY` | 输出队列 `outQueueZ` |
| --- | --- | --- |
| `CopyIn` | `AllocTensor → DataCopy(GM→Local) → EnQue` | 暂不使用 |
| `Compute` | `DeQue → 读取 → FreeTensor` | `AllocTensor → Add 写入 → EnQue` |
| `CopyOut` | 槽位已经归还 | `DeQue → DataCopy(Local→GM) → FreeTensor` |

`EnQue` 表示生产者已经完成当前 LocalTensor 的写入，下一阶段才可以 `DeQue` 取得它；`FreeTensor` 表示消费者已经完成使用，队列槽位可以交给后续 Tile 复用。

### 结论

输入数据的依赖顺序是：

```text
CopyIn EnQue → Compute DeQue → Compute FreeTensor
```

输出数据的依赖顺序是：

```text
Compute EnQue → CopyOut DeQue → CopyOut FreeTensor
```

默认单缓冲时，每个队列只有一个槽位，因此完整归还当前 Tile 的槽位后，后续 Tile 才能持续推进。

### 实验现象

生命周期正确时，一个 Tile 从 GM 搬入、完成 Vector 加法并搬回 GM，随后队列恢复为可分配状态。遗漏 `EnQue` 会破坏生产者到消费者的就绪通知；遗漏 `FreeTensor` 则会使槽位一直处于占用状态，后续 `AllocTensor` 无法正常取得缓冲。

### 常见错误

- `DataCopy` 后直接在另一阶段访问 LocalTensor，没有通过 `EnQue/DeQue` 建立阶段依赖；
- 在 `Add` 尚未完成前释放输入；
- `CopyOut` 尚未消费输出就提前释放 `zLocal`；
- 只给 `x` 完成入队和释放，遗漏 `y` 的对称操作。

## 6. 如何解释 `avg_kernel_us`

### 推导

程序在预热后，用 Host 侧时钟包围多次 Kernel launch 和随后的一次 Stream 同步，再用总时间除以迭代次数。H2D、D2H 和 CPU Golden 校验不在该计时区间内，但 Host 发射开销及同步开销会计入或被均摊。

### 结论

`avg_kernel_us` 的准确口径是 Host 侧 `launch_plus_sync` 平均耗时，不是设备侧纯 Kernel Task Duration。若要得到设备任务执行时间，应使用性能分析工具中的设备侧指标，并清楚标注数据来源。

### 实验现象

同一程序的 `avg_kernel_us` 可能与 `msprof` 报告的 Task Duration 不同，这是计时边界不同造成的正常现象。多次运行也可能有小幅波动，应优先比较同一环境、同一参数和同一口径下的结果。

### 常见错误

- 把字段名中的 `kernel` 理解为“已经排除了所有 Host 开销”；
- 将一次 `msprof` Task Duration 与 Host 多轮平均值直接相减后归因于算子实现；
- 只记录耗时，不记录 `N`、`blockDim`、`tileCount`、`buffer_num` 和计时口径。

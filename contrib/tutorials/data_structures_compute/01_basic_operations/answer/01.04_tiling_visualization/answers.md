# 01.04 Tiling 可视化交互实验参考解析

本目录中的 `tiling_record.py` 是编程实践的参考实现；其输出应与 Notebook 中 `derive_tiling(28672, 4, 7, 3, 6)` 的结果一致。

建议先在纸上完成元素个数、元素下标和 Byte 数的换算，再核对本页。以下区间均采用左闭右开写法 `[start, stop)`。

## 1. 参数推导

题目参数为 `N=28672`、`blockDim=4`、`tileCount=7`，数据类型为 `float32`，因此每个元素占 4 Byte。

### 推导

先按逻辑 Block 均分全局数组：

```text
blockLength = N / blockDim
            = 28672 / 4
            = 7168 个元素
```

再在每个 Block 内均分真实的 GM Tile：

```text
tileLength = blockLength / tileCount
           = 7168 / 7
           = 1024 个元素

tileBytes = tileLength × sizeof(float)
          = 1024 × 4
          = 4096 Byte
```

### 结论

- `blockLength=7168` 个元素；
- `tileLength=1024` 个元素；
- `tileBytes=4096` Byte；
- `4096 % 32 == 0`，满足本实验连续 `DataCopy` 的 32 Byte 对齐要求。

### 实验现象

把这组参数传给 01.04 中的 `derive_tiling`，函数应正常返回，不会抛出异常。后续将它作为合法用例运行时，4 个 Block 各处理 7168 个元素，每个 Block 都执行 7 次 Tile 搬入、计算和搬出。

### 常见错误

- 直接计算 `N / tileCount`。Tile 是在“每个 Block 内”继续切分，因此必须先求 `blockLength`；
- 把 `tileLength=1024` 写成 1024 Byte。这里的单位是“元素”，对应的传输量才是 4096 Byte；
- 把 `tileCount=7` 当作队列有 7 个缓冲槽。本实验中它表示每核真实的 GM Tile 数，与 `TQue` 的缓冲槽位数无关。

## 2. Block 3、Tile 6 的全局区间

### 推导

Block 和 Tile 的编号都从 0 开始。Block 3 的起点为：

```text
blockOffset = 3 × blockLength
            = 3 × 7168
            = 21504
```

Tile 6 在本 Block 内的起点为：

```text
tileOffset = 6 × tileLength
           = 6 × 1024
           = 6144
```

因此它的全局起点和终点分别为：

```text
globalStart = blockOffset + tileOffset
            = 21504 + 6144
            = 27648

globalStop = globalStart + tileLength
           = 27648 + 1024
           = 28672
```

### 结论

Block 3、Tile 6 的全局起点是 `27648`，处理区间是 `[27648, 28672)`。区间内共有 `28672-27648=1024` 个元素，最后一个有效下标是 `28671`。

### 实验现象

这是最后一个 Block 的最后一个 Tile，因此右端点恰好等于 `N`。若遍历条件写成 `i < 28672`，不会越界；若误写为 `i <= 28672`，会多访问一个元素。

### 常见错误

- 把 Block 3、Tile 6 理解成“第 3 个 Block、第 6 个 Tile”，进而使用 2 和 5 计算；
- 只写核内偏移 `6144`，遗漏 Block 起点 `21504`；
- 把左闭右开区间误写为 `[27648, 28671)`，导致少处理一个元素。

## 3. 三组非法参数

### 推导与结论

| 参数 | 首个不满足的条件 | 计算 | 拒绝原因 |
| --- | --- | --- | --- |
| `N=16385, blockDim=8, tileCount=8` | `N % blockDim == 0` | `16385 % 8 = 1` | 无法把所有元素均匀分给 8 个逻辑 Block |
| `N=16384, blockDim=8, tileCount=7` | `blockLength % tileCount == 0` | `blockLength=2048`，`2048 % 7 = 4` | 每个 Block 无法均匀切成 7 个 Tile |
| `N=8200, blockDim=8, tileCount=1` | `tileBytes % 32 == 0` | `blockLength=tileLength=1025`，`tileBytes=4100`，`4100 % 32 = 4` | 单次连续 `DataCopy` 的长度未按 32 Byte 对齐 |

### 实验现象

依次调用 `derive_partition` 时，三组参数会分别在三道检查处抛出 `ValueError`。在完整 Host 程序中，它们也应在 Kernel 启动前被拒绝，而不是先运行再依赖精度检查发现问题。

### 常见错误

- 只检查 `N` 能否被 `blockDim` 整除，忽略核内 Tile 的整除关系；
- 看到 `8200 % 8 == 0` 就认为第三组合法，未继续把 `tileLength` 从元素换算成 Byte；
- 为了“让参数通过”而向下取整。当前实验没有尾核和尾 Tile 处理，取整会造成元素遗漏。

## 4. `blockDim` 的含义

### 推导

`blockDim` 是本次 Kernel 启动时指定的逻辑 Block 数。Kernel 内部通过 `GetBlockNum()` 获得该数量，通过 `GetBlockIdx()` 获得当前逻辑 Block 的编号，并据此计算自己的 GM 区间。

### 结论

`blockDim=8` 只能说明本次启动创建了 8 个逻辑 Block，不能据此断言设备“总共有 8 个物理核”。逻辑 Block 如何映射和调度到设备计算资源，由运行时和硬件决定。

### 实验现象

默认参数 `N=16384`、`blockDim=8`、`tileCount=8` 下：

- `blockLength=2048`；
- `tileLength=256`；
- `tileBytes=1024`；
- Block 2、Tile 3 的起点为 `2×2048+3×256=4864`，区间为 `[4864, 5120)`。

无论设备报告多少物理计算资源，上述下标推导都只由本次启动参数决定。

### 常见错误

- 用设备物理核总数替代 `blockDim` 参与地址计算；
- 漏掉 `GetBlockIdx()`，使所有逻辑 Block 都从全局下标 0 开始；
- 把默认 `tileCount=8` 与默认单缓冲的 1 个队列槽位混为一谈。

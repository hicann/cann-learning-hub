# 01.05 独立实践参考解析

请先独立完成三个 TODO 并通过完整自检，再核对本页。可直接查看同目录的 [`student_compute.h`](./student_compute.h)，但更重要的是理解三个表达式的单位和调用位置。

## 1. `StudentBlockOffset`：核间偏移

### 推导

每个逻辑 Block 连续处理 `blockLength` 个元素，`blockIdx` 从 0 开始，因此当前 Block 的全局起始元素下标为：

```text
blockOffset = blockIdx × blockLength
```

默认配置 `N=16384`、`blockDim=8` 时，`blockLength=2048`。例如 Block 2 的起点是 `2×2048=4096`。

### 结论

参考实现为：

```cpp
__aicore__ inline uint32_t StudentBlockOffset(
    uint32_t blockLength, uint32_t blockIdx)
{
    return blockLength * blockIdx;
}
```

返回值的单位是“`float32` 元素下标”，不是 Byte。调用处会用它平移 `x`、`y` 和 `z` 的 GM 指针。

### 实验现象

正确实现后，不同逻辑 Block 绑定到互不重叠的 GM 区间。保留占位返回值 0 时，所有 Block 都处理首段数据，完整精度用例无法通过。

### 常见错误

- 额外乘以 `sizeof(float)`。C++ 指针加法已经按元素类型缩放；
- 写成 `blockLength + blockIdx`；
- 直接使用设备物理核数量，而不是调用处传入的逻辑 `blockIdx`。

## 2. `StudentTileOffset`：核内偏移

### 推导

`progress` 是当前 Tile 在本 Block 内的序号，`tileLength` 是每个 Tile 的元素数，因此：

```text
tileOffset = progress × tileLength
```

默认配置中 `tileLength=256`。Tile 3 的核内起点为 `3×256=768`；若位于 Block 2，则全局起点为 `4096+768=4864`，区间为 `[4864, 5120)`。

### 结论

参考实现为：

```cpp
__aicore__ inline uint32_t StudentTileOffset(
    int32_t progress, uint32_t tileLength)
{
    return static_cast<uint32_t>(progress) * tileLength;
}
```

它只返回“本核数据段内”的元素偏移。Block 偏移已经在 `GlobalTensor::SetGlobalBuffer` 时处理，不能在这里重复叠加。

### 实验现象

正确实现后，`Process` 的第 `p` 次迭代处理本 Block 的第 `p` 个真实 GM Tile。若始终返回 0，每次循环都重复处理第 0 个 Tile，其余 Tile 保持未计算状态。

### 常见错误

- 使用 `progress + tileLength`；
- 再次加上 `blockOffset`，造成地址偏移重复；
- 把 `progress` 与队列槽位编号混淆。`progress` 可从 0 增长到 `tileCount-1`，即使默认队列只有 1 个槽位也不改变。

## 3. `STUDENT_COMPUTE`：Vector 计算

### 推导

`CopyIn` 已把当前 Tile 的 `x`、`y` 从 GM 搬入输入队列；`Compute` 出队得到两个输入 `LocalTensor<float>`，并为输出申请 `LocalTensor<float>`。题目要求逐元素计算：

```text
z[i] = x[i] + y[i], 0 <= i < len
```

因此应调用接受两个输入张量的 Vector 加法 API，而不是使用标量循环，也不是给 `x` 加常数 0。

### 结论

参考实现为：

```cpp
#define STUDENT_COMPUTE(z, x, y, len) \
    AscendC::Add((z), (x), (y), (len))
```

其中 `x` 和 `y` 来自输入队列出队，`z` 是输出队列新申请的 LocalTensor，`len` 是当前 Tile 的 `tileLength`。

### 实验现象

占位实现 `AscendC::Adds(z, x, 0.0F, len)` 的结果等价于复制 `x`，完全没有使用 `y`。替换为 `AscendC::Add` 后，CPU Golden 对全部元素的比较应通过。

### 常见错误

- 继续使用 `Adds`；该 API 是“张量加标量”，与本题双输入张量相加不符；
- 交换输出和输入参数位置；
- 用 `for` 循环逐元素相加，违背本实验使用 Ascend C Vector API 的要求；
- 传入 `blockLength` 而不是当前 LocalTensor 对应的 `tileLength`。

## 4. 完整参考代码

三个关键点合在一起如下：

```cpp
#pragma once

__aicore__ inline uint32_t StudentBlockOffset(
    uint32_t blockLength, uint32_t blockIdx)
{
    return blockLength * blockIdx;
}

__aicore__ inline uint32_t StudentTileOffset(
    int32_t progress, uint32_t tileLength)
{
    return static_cast<uint32_t>(progress) * tileLength;
}

#define STUDENT_COMPUTE(z, x, y, len) \
    AscendC::Add((z), (x), (y), (len))
```

实践工程中的 `vector_add_student.asc` 是独立的学生起始工程，不依赖 `src/demo`。只需补全 `student_compute.h`，不应把演示工程整体复制过来替代练习。

## 5. 完整自检结果如何阅读

自检依次确认：

1. 三个学生接口是否使用了要求的参数和 `AscendC::Add`；
2. 工程能否从干净目录完成构建；
3. 三组非法 Shape 是否在 Kernel 启动前按预期拒绝；
4. 三组合法 Shape 是否满足输出字段与精度阈值 `max_abs_error<=1e-6`。

合法用例为：

| `N` | `blockDim` | `tileCount` | 关键意义 |
| ---: | ---: | ---: | :--- |
| 16384 | 8 | 8 | 默认配置 |
| 28672 | 4 | 7 | 验证合法的奇数 `tileCount`，防止把 Tile 数与缓冲槽位绑定 |
| 65536 | 8 | 8 | 验证更大规模下的数据覆盖 |

全部完成后应看到三项接口检查通过、干净构建成功、非法参数 3/3、合法精度用例 3/3，以及末尾的 `SELF_CHECK PASS`。实机验证中三个合法用例的最大绝对误差均为 0。

starter 能够通过编译，但接口检查不会通过，因此完整自检会先跳过合法输入精度用例。这说明“能够编译”并不等于“计算正确”。如果完成 TODO 后仍显示 `SELF_CHECK FAIL`，应从输出中最早的 `[FAIL]` 开始排查。

### 常见错误

- 只看末尾结果，不阅读最早出现的失败项；
- 在旧构建目录上反复编译，没有确认干净构建是否通过；
- 直接修改自检脚本或测试用例，而不是修复 `student_compute.h`；
- 只查看数组开头的少量元素，误判完整向量已经正确覆盖。

## 6. 完成核对

- `student_compute.h` 中只补全要求的三个关键点；
- 三项接口检查与干净构建均通过；
- 三类非法 Shape 均被 Host 正确拒绝；
- 三组合法 Shape 的精度检查全部通过；
- 自检末尾显示 `SELF_CHECK PASS`；
- 能说明 Block 偏移、Tile 偏移和 Vector 加法分别解决什么问题。

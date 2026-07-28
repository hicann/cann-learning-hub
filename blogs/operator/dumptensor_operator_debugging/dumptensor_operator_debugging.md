# 使用DumpTensor定位算子计算结果异常

在 CPU 程序里，结果不对可以 `printf` 逐步打印排查；但在 NPU 算子开发中，片上数据对外完全不可见，算子跑完只能看到最终输出，中间过程无从观察，出了问题只能靠推断。

`DumpTensor` 补上了这个缺口。它让算子在运行时将任意存储位置（GM、UB、L1）的 Tensor 内容打印出来，在数据流动的每一个环节建立可观测点，把"黑盒"调试变为"白盒"调试。

本文通过一个真实的 Bug 案例，完整演示 `DumpTensor` 的用法和调试思路。

## API 说明

### 基本接口

```
// 打印 LocalTensor
AscendC::DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize);

// 打印 GlobalTensor
AscendC::DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize);

```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| tensor | LocalTensor / GlobalTensor | 要打印的 Tensor |
| desc | uint32_t | 自定义标识符，用于区分不同打印点，建议按顺序编号 |
| dumpSize | uint32_t | 打印的元素个数 |

### 矩阵形态输出

当 Tensor 表示二维矩阵时，可以传入形状信息，输出更直观：

```
uint32_t shape[] = {3, 3};
AscendC::ShapeInfo shapeInfo(2, shape);  // 2维，形状为 3x3
AscendC::DumpTensor(tensor, 10, 9, shapeInfo);

```

输出效果：

```
DumpTensor: desc=10, addr=0, data_type=float16, position=UB, dump_size=9
[[1.000000, 2.000000, 3.000000],
 [4.000000, 5.000000, 6.000000],
 [7.000000, 8.000000, 9.000000]]

```

### 打印局部数据

`DumpTensor` 支持从 Tensor 的任意偏移位置开始打印：

```
// 从第 64 个元素开始，打印 16 个元素
AscendC::DumpTensor(xLocal[64], 0, 16);

```

以上是 `DumpTensor` 的核心用法。完整的参数说明和调用示例可以查阅以下文档：

- DumpTensor API 文档https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0192.html

- asc-tools：Kernel 调试数据展示样例https://gitcode.com/cann/asc-tools/tree/master/examples/01_show_kernel_debug_data

- 《Ascend C 算子开发文档手册》https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html

## 调试思路：沿数据流逐环节向前推进

添加打印点的核心思路是：沿数据流动方向，从最前端（输入数据）开始逐步向后验证，找到第一个输出异常的环节，问题就在那里。

```
输入数据 → [打印点1] → 搬入 (CopyIn) → [打印点2] → 计算 (Compute) → [打印点3] → 回写 (CopyOut) → [打印点4] → 输出数据

```

先在打印点1确认原始输入正确，再移动到打印点2验证搬运结果，以此类推——第一个出现异常的打印点，就是出问题的那一步。一旦定位到问题环节，就不需要再往后排查了。

## 实战案例：Add 算子输出后半段数据错误

### 问题描述

开发一个 Add 算子，对两个向量逐元素相加。算子采用 Tiling 方式分块处理数据，通过循环调用 `Process(progress)` 完成每一块的搬运和计算，`progress` 是当前迭代的块索引（从 0 开始）。本例共分两次迭代，每次处理 8 个元素：

- 输入 `x`（block 0 负责的部分）：`[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]`

- 输入 `y`：`[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`

- 期望输出：`[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]`

- 实际输出：`[2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9]`

后半段数据明显是第一次迭代结果的重复。但光看代码，问题可能出在输入数据、搬运逻辑或计算逻辑中，不好判断。

### 问题代码

`Process` 是算子的主循环，按 Tile 数依次调用 CopyIn → Compute → CopyOut，`progress` 就是循环变量 `i`，表示当前处理的是第几个分块：

```
__aicore__ inline void Process()
{
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

```

三个子函数各司其职，均以 `progress` 为参数计算当前分块在 GM 中的偏移地址：

```
__aicore__ inline void CopyIn(int32_t progress)
{
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm, this->tileLength);  // ← 每次都从头搬
    AscendC::DataCopy(yLocal, yGm, this->tileLength);  // ← 每次都从头搬
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}

__aicore__ inline void Compute(int32_t progress)
{
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
    AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
    outQueueZ.EnQue<float>(zLocal);
}

__aicore__ inline void CopyOut(int32_t progress)
{
    AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
    outQueueZ.FreeTensor(zLocal);
}

```

### 加入打印点，逐步排查

第一步：验证 GM 侧原始输入数据

在 `Process` 函数的循环之前打印：

```
AscendC::DumpTensor(xGm, 1, 8);
AscendC::DumpTensor(yGm, 2, 8);

```

输出：

```
DumpTensor: desc=1, position=GM, data_type=float32, dump_size=8
[1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000]  ✓ (第一个 tile 的数据)

DumpTensor: desc=2, position=GM, data_type=float32, dump_size=8
[1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]  ✓

```

原始输入正确，问题不在数据源头。

第二步：验证搬运后的 UB 数据

在 `CopyIn` 的 `EnQue` 之前打印。由于同一行代码会在每次迭代中执行，需要让不同迭代的输出有所区分——将 `progress` 叠加到 `desc` 上即可：第 0 次迭代输出 `desc=10`，第 1 次迭代输出 `desc=11`，以此类推。

```
// progress=0 时 desc=10，progress=1 时 desc=11，依次类推
AscendC::DumpTensor(xLocal, 10 + progress, this->tileLength);

```

第一次迭代（progress=0）输出：

```
DumpTensor: desc=10, position=UB, dump_size=8
[1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000]  ✓

```

第二次迭代（progress=1）输出：

```
DumpTensor: desc=11, position=UB, dump_size=8
[1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000]  ✗  期望是 [9, 10, 11, 12, 13, 14, 15, 16]

```

问题找到了：第二次迭代搬入的数据和第一次完全相同，说明 `CopyIn` 没有按 `progress` 推进地址偏移，每次都在从 GM 的起始位置读取。

### 修复

```
__aicore__ inline void CopyIn(int32_t progress)
{
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
    // 修复：用 progress 计算正确的 GM 偏移地址
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}

```

修复后，第二次迭代打印出 `[9, 10, 11, 12, 13, 14, 15, 16]`，最终输出正确变为 `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]`。

tips：调试结束后记得去除DumpTensor，否则会影响算子性能。

## 小结

Ascend C 新提供的 DumpTensor 工具，进一步增强了算子开发中的问题定位能力：当算子出现精度异常时，开发者可以直接打印算子内部各阶段的 Tensor 数据，快速判断问题究竟发生在输入处理、计算过程还是结果写回等具体环节，从而显著缩短排查路径、提升调试效率。 Ascend C 也在持续围绕易用性进行迭代升级，通过简化工程复杂度、增强调测能力、优化问题定位体验，并引入 AI 辅助编程等手段，不断降低算子开发和维护成本，助力开发者更高效地完成算子实现、验证与优化，持续提升开发效率与体验。

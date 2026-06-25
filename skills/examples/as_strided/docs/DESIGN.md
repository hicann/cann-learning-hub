# as_strided 算子设计与实施文档

---

## 0. 概述

### 0.0 需求类型判断

**判断标准**：用户未明确指定具体 shape，要求支持 float16/float32/int32 多种数据类型和 1-4 维任意 stride，属于**通用**需求。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | as_strided |
| 算子类别 | Conversion（Index/Gather） |
| 需求类型 | 通用 |
| 支持数据类型 | float16, float32, int32 |
| 支持芯片 | ascend910b |
| 特殊约束 | stride 可为正/负/零；storage_offset >= 0；支持非 32 字节对齐场景 |

---

## 1. 算子设计

### 1.1 数学公式

```
输入:  input_x - 任意形状, dtype ∈ {float16, float32, int32}
属性:  size[n]    - 输出张量各维度大小
       stride[n]  - 输出张量各维度步长（可为正/负/零）
       storage_offset - 起始偏移量（>= 0）
输出:  output - shape = size, dtype 与 input_x 相同

数学公式:
output[i_0, i_1, ..., i_{n-1}] = input_x[storage_offset + Σ(i_j × stride_j)]

其中 i_j ∈ [0, size_j)，j = 0, 1, ..., n-1
```

**本质**：将输出展平为 1D 后，每个输出元素从输入 tensor 的任意位置（由 stride 和 offset 决定）gather 而来。这是一个典型的 **非连续索引读取** 操作。

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 |
|---------|---------|---------|---------|
| 输入数据 GM→UB 搬运 | DataCopyPad | blockCount=1, blockLen=inputBytes | GM 连续 → UB 连续 |
| 按 offset 表收集元素 | Gather | dst, src, srcOffset, srcBaseOffset=0, count | UB 内按字节偏移收集 |
| 输出数据 UB→GM 搬运 | DataCopyPad | blockCount=1, blockLen=outputBytes | UB 连续 → GM 连续 |
| 偏移表 GM→UB 搬运 | DataCopyPad | blockCount=1, blockLen=offsetTileBytes | Workspace → UB 连续 |
| 单元素 GM→UB 搬运(Path B) | DataCopyPad | blockCount=1, blockLen=sizeof(T) | GM 单元素 → UB |

#### 1.2.1 API 语义验证

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 |
|-----|---------|---------|---------|---------|-----|
| DataCopyPad (GM→UB) | GM 连续存储，可能非 32B 对齐 | 全量输入搬运到 UB | `DataCopyPad(dst, src, DataCopyExtParams, DataCopyPadExtParams)` | dst 起始地址 32B 对齐；blockLen 单位为字节 | ✅ |
| Gather (UB→UB) | UB 内源数据连续存储；offset 表为 uint32_t 字节偏移 | 按 offset 表从源收集到目的 | `Gather<T>(dst, src, srcOffset, srcBaseOffset, count)` | dst/src/srcOffset 起始地址 32B 对齐；T 支持 half/float/int32_t；offset 单位为字节；srcBaseOffset 对齐到元素大小 | ✅ |
| DataCopyPad (UB→GM) | UB 连续存储，可能非 32B 对齐 | 输出搬运到 GM | `DataCopyPad(dst, src, DataCopyExtParams)` | src 起始地址 32B 对齐；blockLen 单位为字节；GM 端无对齐约束 | ✅ |
| DataCopyPad (Workspace→UB) | Workspace 中预计算的 offset 表连续存储 | 偏移表搬运到 UB | `DataCopyPad(dst, src, DataCopyExtParams, DataCopyPadExtParams)` | dst 起始地址 32B 对齐；blockLen 单位为字节 | ✅ |

**验证清单**：
- [x] 1. DataCopyPad GM→UB：数据布局为 GM 连续存储，支持非对齐搬运，满足需求
- [x] 2. Gather：源数据和 offset 表均在 UB，T 支持 half/float/int32_t，offset 为字节偏移，满足需求
- [x] 3. DataCopyPad UB→GM：支持非对齐搬运，GM 端自动丢弃 dummy 数据，满足需求
- [x] 4. DataCopyPad Workspace→UB：偏移表在 workspace 中连续存储，DataCopyPad 搬运到 UB，满足需求
- [x] 5. 已查阅官方文档，API 适用场景正确
- [x] 6. 已记录验证过程

**Gather API 关键约束确认**：
- 支持数据类型：half, float, int32_t（ascend910b 属于 A3 系列，均支持）✅
- srcOffset 类型：LocalTensor\<uint32_t\>，单位为字节 ✅
- srcBaseOffset：uint32_t，单位为字节，需对齐到元素大小 ✅
- count：处理元素个数 ✅
- 地址对齐：dst/src/srcOffset 起始地址需 32 字节对齐 ✅
- 偏移地址约束：偏移地址后不能超出 UB 大小数据的范围 ✅（Host 预计算时做防御性 clamp 保证）

### 1.3 数据流

**Path A（输入全量入 UB，主路径）**：

```
Host 预计算: 对每个输出元素计算 srcIdx，clamp 到 [0, inputTotalElements-1]，
            生成 byte offset 表存入 workspace

输入 input_x (Global Tensor)
    ↓ DataCopyPad (全量搬运, 仅一次)
输入 input_x (Local Tensor, VECIN)
    ↓ [per tile loop:]
    偏移表 workspace[offsetStart..offsetStart+tileSize] (Global Tensor)
    ↓ DataCopyPad (搬运当前 tile 的偏移表)
    偏移表 offsetLocal (Local Tensor, VECCALC)
    ↓ Gather (按 offset 表从 inputLocal 收集)
输出 output (Local Tensor, VECOUT, Double Buffer)
    ↓ DataCopyPad (非对齐写回)
输出 output (Global Tensor)
```

**Path B（输入过大无法全量入 UB，回退路径）**：

```
Host 预计算: 对每个输出元素计算 srcIdx，clamp 到 [0, inputTotalElements-1]，
            生成 srcIdx 表存入 workspace

[per tile loop:]
    偏移表 workspace[srcIdxStart..srcIdxStart+tileSize] (Global Tensor)
    ↓ DataCopyPad (搬运当前 tile 的 srcIdx 表)
    srcIdx 表 srcIdxLocal (Local Tensor, VECCALC)
    ↓ [per element in tile:]
    输入 input_x[srcIdx[i]] (Global Tensor)
        ↓ DataCopyPad (单元素搬运, blockLen=sizeof(T))
    临时 tmpLocal (Local Tensor, VECIN, 32B 对齐)
        ↓ GetValue + SetValue 写入输出位置
输出 output (Local Tensor, VECOUT)
    ↓ DataCopyPad (非对齐写回)
输出 output (Global Tensor)
```

### 1.4 核心计算步骤

**核心计算步骤**（Host 侧预计算 + Kernel 侧执行）：

```
Host 侧预计算（一次性，结果存入 workspace）：
  1. 对每个输出元素 f ∈ [0, totalOutputElements)：
     a. 展平输出索引 - 将一维平坦索引 f 反推为多维索引 (i_0,...,i_{n-1})
     b. 计算源索引 - srcIdx = storage_offset + Σ(i_j × stride_j)
     c. 防御性 clamp - srcIdx = clamp(srcIdx, 0, inputTotalElements - 1)
     d. Path A: 生成 byte offset 表 offset[f] = srcIdx × sizeof(T)
        Path B: 生成 srcIdx 表 srcIdxTable[f] = srcIdx

Kernel 侧执行（per tile）：
  Path A:
    1. DataCopyPad 从 workspace 搬运当前 tile 的 offset 表到 UB
    2. Gather 按偏移表从 inputLocal 收集到 outputLocal
    3. DataCopyPad 写回 GM
  Path B:
    1. DataCopyPad 从 workspace 搬运当前 tile 的 srcIdx 表到 UB
    2. 逐元素 DataCopyPad 从 GM[inputGlobal[srcIdx[i]]] 读取到 tmpLocal
    3. GetValue + SetValue 写入 outputLocal 对应位置
    4. DataCopyPad 写回 GM
```

**分支差异对比**：

| 操作 | Path A（输入入UB） | Path B（输入不入UB） |
|------|-------------------|---------------------|
| 输入数据准备 | 全量 DataCopyPad GM→UB（一次） | 逐元素 DataCopyPad GM→UB |
| 偏移/索引表 | Host 预计算 byte offset → workspace → DataCopyPad 到 UB | Host 预计算 srcIdx → workspace → DataCopyPad 到 UB |
| 核心收集操作 | Gather API（硬件指令） | GetValue + SetValue 逐元素写入 |
| 输出写回 | DataCopyPad UB→GM（批量） | DataCopyPad UB→GM（批量） |
| 性能特征 | 高效（1次大DMA + N_tile次小DMA(offset表) + N_tile次Gather + N_tile次DMA） | 较慢（N_tile次小DMA(srcIdx表) + N次小DMA(元素) + N_tile次DMA） |

**关键设计要点**：

1. **路径选择**：Host 侧根据 `inputTotalBytes + offsetTableTileBytes + outputTileBytes ≤ UBBudget` 判断走 Path A 还是 Path B
2. **索引预计算**：Host 侧预计算所有输出元素的源索引/偏移表，存入 workspace。避免 Kernel 内大量 SetValue 标量操作和除法/取模运算
3. **防御性 clamp**：Host 预计算时对 srcIdx 做 clamp 到 [0, inputTotalElements-1]，防止负 stride 或越界参数导致 Gather 偏移越界（硬件异常）
4. **零 stride 处理**：stride=0 时，该维度所有位置读取同一元素，Gather 自然支持（多个 offset 值相同）
5. **非对齐处理**：CopyIn/CopyOut 统一使用 DataCopyPad，避免对齐判断分支
6. **Double Buffer**：Path A 的 outQueueDst 开启 Double Buffer，使当前 tile 的 CopyOut 与下一 tile 的 Gather+CopyIn(offset) 并行

**参数使用规则**：

| 参数位置 | 用有效长度 | 用对齐长度 |
|---------|-----------|-----------|
| DataCopyPad blockLen / Gather count | ✓ | ✗ |
| UB 数据偏移 / Buffer 大小 | ✗ | ✓ |

### 1.5 内存管理(Buffer 规划)

**UB 容量**：ascend910b (DAV_2201) 实际 UB 为 **192KB (196608 bytes)**（依据 CANN 9.0.0 `kernel_utils_constants.h` 中 `__NPU_ARCH__ == 2201` 定义）。

> **注意**：`GetUBSizeInBytes()` 不支持 A3 系列（含 ascend910b），因此使用编译时常量。

**Path A Buffer 规划**：

| Buffer 名称 | 用途 | 大小计算 | TPosition | Double Buffer |
|------------|------|---------|-----------|--------------|
| inQueueSrc | 输入数据（全量） | AlignUp(inputTotalElements × sizeof(T), 32) | VECIN | 否（全量一次 CopyIn，无并行收益） |
| tmpQueueOffset | Gather 偏移表 | AlignUp(tileSize × sizeof(uint32_t), 32) | VECCALC | 否 |
| outQueueDst | 输出数据 | AlignUp(tileSize × sizeof(T), 32) | VECOUT | **是**（CopyOut ∥ Compute 并行） |

**Path A 总 UB 使用量**：

```
ubUsed = AlignUp(inputTotalElements × sizeof(T), 32)    // 输入全量
       + AlignUp(tileSize × 4, 32)                        // offset 表
       + AlignUp(tileSize × sizeof(T), 32) × 2            // 输出 tile (Double Buffer)
```

约束：`ubUsed ≤ 192KB (196608 bytes)`

**Path B Buffer 规划**：

| Buffer 名称 | 用途 | 大小计算 | TPosition | Double Buffer |
|------------|------|---------|-----------|--------------|
| inQueueTmp | 单元素临时读取 | 32 字节（最小对齐单元） | VECIN | 否 |
| tmpQueueSrcIdx | srcIdx 表 | AlignUp(tileSize × sizeof(uint32_t), 32) | VECCALC | 否 |
| outQueueDst | 输出数据 | AlignUp(tileSize × sizeof(T), 32) | VECOUT | 否 |

**Path B 总 UB 使用量**：

```
ubUsed = 32                                              // 单元素临时
       + AlignUp(tileSize × 4, 32)                        // srcIdx 表
       + AlignUp(tileSize × sizeof(T), 32)                // 输出 tile
```

**偏移表存储规划**：

| 用途 | 大小 | 存储位置 | 说明 |
|------|------|---------|------|
| Path A: byte offset 表 | totalOutputElements × sizeof(uint32_t) | Tiling Data（追加） | 优先追加到 tiling data；容量不足时回退到 Kernel 内增量计算 |
| Path B: srcIdx 表 | totalOutputElements × sizeof(uint32_t) | Tiling Data（追加） | 优先追加到 tiling data；容量不足时回退到 Kernel 内增量计算 |

> **注意**：TilingFunc 无法直接写入 workspace（CANN 9.0 gert::TilingContext 不提供 GetWorkspaceData 接口），因此偏移表通过 `GetRawTilingData()->Append()` 追加到 tiling data 二进制码流中。Kernel 通过 `tiling + sizeof(AsStridedTilingData)` 计算偏移表地址，使用 DataCopyPad 搬运到 UB。当 tiling data 容量不足时，Kernel 回退到增量索引计算（混合进制计数器，仅用加法/减法/比较，无除法/取模）。

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 输出元素一维展平后按元素范围切分 |
| 单核任务量 | `coreElements = CeilDiv(totalOutputElements, blockDim)` |
| 使用的核数 | `blockDim = min(可用 AI Core 数, totalOutputElements)`，强制动态计算 |
| 负载均衡方式 | 均分输出元素，尾核处理余量 |

**多核切分公式**（Kernel 内动态计算，不存入 TilingData）：

```cpp
uint32_t blockIdx = GetBlockIdx();
uint32_t coreElements = CeilDiv(totalOutputElements, blockDim);
uint32_t coreOffset = blockIdx * coreElements;
// 尾核调整
if (coreOffset + coreElements > totalOutputElements) {
    coreElements = totalOutputElements - coreOffset;
}
```

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192KB (ascend910b, DAV_2201) |
| 单次处理数据量 | tileSize 个输出元素 |
| 是否需要分 chunk | 是，输出元素按 tile 分批处理 |
| chunk 大小计算公式 | 见下方 |

**tileSize 计算**：

```cpp
constexpr uint32_t UB_SIZE = 196608;  // 192KB (DAV_2201)
constexpr uint32_t RESERVED = 8192;   // 预留 8KB
uint32_t ubBudget = UB_SIZE - RESERVED;  // 188416

if (pathA) {
    // Path A: 输入全量在 UB，tileSize 受 offset 表 + 输出 tile(Double Buffer) 约束
    uint32_t inputBytes = AlignUp(inputTotalElements * sizeof(T), 32);
    uint32_t remainBudget = ubBudget - inputBytes;
    // offset 表: tileSize * 4, 输出: tileSize * sizeof(T) * 2 (Double Buffer)
    uint32_t perTileBytes = 4 + sizeof(T) * 2;  // 每元素占用
    tileSize = (remainBudget / perTileBytes) & ~0x7;  // 向下对齐到 8 元素
    tileSize = max(tileSize, 8);  // 至少 8 元素
} else {
    // Path B: 无输入全量，tileSize 受 srcIdx 表 + 输出 tile 约束
    uint32_t remainBudget = ubBudget - 32;  // 减去单元素临时 buffer
    // srcIdx 表: tileSize * 4, 输出: tileSize * sizeof(T)
    uint32_t perTileBytes = 4 + sizeof(T);  // 每元素占用
    tileSize = (remainBudget / perTileBytes) & ~0x7;
    tileSize = max(tileSize, 8);
}
```

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 | TilingKey 区分：DT_FLOAT16(0)、DT_FLOAT(1)、DT_INT32(2)，Kernel 模板化 |
| 输入大小（Path A/B） | Host 判断 inputTotalBytes 是否可入 UB，设置 pathFlag |
| 大 shape | 多核切分 + UB tile 循环处理 |
| 小 shape | 单核或少量核即可处理，tile 可能只需 1 轮 |
| 对齐 | CopyIn/CopyOut 统一使用 DataCopyPad，无需区分对齐/非对齐 |
| 非对齐 | DataCopyPad 自动处理，blockLen 使用有效字节长度 |
| 正 stride | 正常索引，offset 为正值 |
| 负 stride | 源索引可能为负，Host 预计算时 clamp 到 [0, inputTotalElements-1]，防止 Gather 偏移越界 |
| 零 stride | 该维度所有位置映射同一源元素，Gather 多个 offset 相同，自然支持 |
| 越界访问 | 防御性 clamp：srcIdx < 0 或 srcIdx >= inputTotalElements 时 clamp 到 0，输出值虽错但不触发硬件异常 |

### 2.4 类别特有设计

#### 2.4.1 Path A：输入全量入 UB + Gather

**适用场景**：`inputTotalBytes + offsetTableTileBytes + outputTileBytes × 2 ≤ UBBudget`

**Init 伪代码**：

```cpp
// GlobalBuffer 设置（1D 展平视图）
inputGlobal.SetGlobalBuffer((__gm__ T*)inputGM);
outputGlobal.SetGlobalBuffer((__gm__ T*)outputGM);
offsetGlobal.SetGlobalBuffer((__gm__ uint32_t*)workspaceGM);  // workspace 中预计算的 offset 表

// 全量输入搬运（仅执行一次）
LocalTensor<T> inputLocal = inQueueSrc.AllocTensor<T>();
DataCopyExtParams inParams{1, (uint32_t)(inputTotalElements * sizeof(T)), 0, 0, 0};
DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
DataCopyPad(inputLocal, inputGlobal, inParams, padParams);
inQueueSrc.EnQue(inputLocal);
```

**Compute 核心流程伪代码**：

```cpp
// Path A: 输入已在 UB (inputLocal)，按 tile 处理输出
LocalTensor<T> inputLocal = inQueueSrc.DeQue<T>();  // 全量输入，整个 Process 复用

for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
    uint32_t tileStart = coreOffset + tile * tileSize;
    uint32_t curTileSize = min(tileSize, coreElements - tile * tileSize);

    // 1. 从 workspace 搬运当前 tile 的 Gather byte offset 表
    LocalTensor<uint32_t> offsetLocal = tmpQueueOffset.AllocTensor<uint32_t>();
    DataCopyExtParams offParams{1, curTileSize * sizeof(uint32_t), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> offPadParams{false, 0, 0, 0};
    DataCopyPad(offsetLocal, offsetGlobal[tileStart], offParams, offPadParams);
    tmpQueueOffset.EnQue(offsetLocal);

    // 2. Gather 收集
    LocalTensor<uint32_t> offLocal = tmpQueueOffset.DeQue<uint32_t>();
    LocalTensor<T> outputLocal = outQueueDst.AllocTensor<T>();
    Gather(outputLocal, inputLocal, offLocal, 0, curTileSize);
    outQueueDst.EnQue(outputLocal);
    tmpQueueOffset.FreeTensor(offLocal);

    // 3. 写回 GM
    LocalTensor<T> outLocal = outQueueDst.DeQue<T>();
    DataCopyExtParams outParams{1, curTileSize * sizeof(T), 0, 0, 0};
    DataCopyPad(outputGlobal[tileStart], outLocal, outParams);
    outQueueDst.FreeTensor(outLocal);
}

inQueueSrc.FreeTensor(inputLocal);
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 | Double Buffer |
|------------|------|---------|--------------|
| inQueueSrc | 输入全量数据 | AlignUp(inputTotalElements × sizeof(T), 32) | 否 |
| tmpQueueOffset | Gather 偏移表（每 tile） | AlignUp(tileSize × 4, 32) | 否 |
| outQueueDst | 输出 tile | AlignUp(tileSize × sizeof(T), 32) | **是** |

#### 2.4.2 Path B：输入过大 + 逐元素 DataCopyPad

**适用场景**：`inputTotalBytes + offsetTableTileBytes + outputTileBytes × 2 > UBBudget`

**Init 伪代码**：

```cpp
// GlobalBuffer 设置（1D 展平视图）
inputGlobal.SetGlobalBuffer((__gm__ T*)inputGM);
outputGlobal.SetGlobalBuffer((__gm__ T*)outputGM);
srcIdxGlobal.SetGlobalBuffer((__gm__ uint32_t*)workspaceGM);  // workspace 中预计算的 srcIdx 表
```

**Compute 核心流程伪代码**：

```cpp
// Path B: 输入在 GM，逐元素读取，按 tile 累积输出
LocalTensor<T> outputLocal = outQueueDst.AllocTensor<T>();

for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
    uint32_t tileStart = coreOffset + tile * tileSize;
    uint32_t curTileSize = min(tileSize, coreElements - tile * tileSize);

    // 1. 从 workspace 搬运当前 tile 的 srcIdx 表
    LocalTensor<uint32_t> srcIdxLocal = tmpQueueSrcIdx.AllocTensor<uint32_t>();
    DataCopyExtParams idxParams{1, curTileSize * sizeof(uint32_t), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> idxPadParams{false, 0, 0, 0};
    DataCopyPad(srcIdxLocal, srcIdxGlobal[tileStart], idxParams, idxPadParams);
    tmpQueueSrcIdx.EnQue(srcIdxLocal);

    LocalTensor<uint32_t> idxLocal = tmpQueueSrcIdx.DeQue<uint32_t>();

    // 2. 逐元素从 GM 读取并写入输出 UB
    for (uint32_t i = 0; i < curTileSize; i++) {
        uint32_t srcIdx = idxLocal.GetValue(i);  // 从预计算的 srcIdx 表读取
        // 从 input GM 读取 1 个元素到临时 UB
        LocalTensor<T> tmpLocal = inQueueTmp.AllocTensor<T>();
        DataCopyExtParams inParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(tmpLocal, inputGlobal[srcIdx], inParams, padParams);
        inQueueTmp.EnQue(tmpLocal);

        // 从临时 UB 写入输出 UB 对应位置
        LocalTensor<T> valLocal = inQueueTmp.DeQue<T>();
        T val = valLocal.GetValue(0);
        outputLocal.SetValue(i, val);
        inQueueTmp.FreeTensor(valLocal);
    }

    tmpQueueSrcIdx.FreeTensor(idxLocal);

    // 3. 写回 GM
    outQueueDst.EnQue(outputLocal);
    LocalTensor<T> outLocal = outQueueDst.DeQue<T>();
    DataCopyExtParams outParams{1, curTileSize * sizeof(T), 0, 0, 0};
    DataCopyPad(outputGlobal[tileStart], outLocal, outParams);
    outQueueDst.FreeTensor(outLocal);
    outputLocal = outQueueDst.AllocTensor<T>();
}
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| inQueueTmp | 单元素临时读取 | 32 字节 |
| tmpQueueSrcIdx | srcIdx 表（每 tile） | AlignUp(tileSize × 4, 32) |
| outQueueDst | 输出 tile | AlignUp(tileSize × sizeof(T), 32) |

**注意**：
- Path B 中 GetValue/SetValue 用于 UB 内部数据搬运（LocalTensor），符合 API 使用约束，但属于已知反模式（标量操作，吞吐受限）。Path B 仅作为大输入场景的回退方案，性能不作为验收标准。
- Path B 的 srcIdx 表从 workspace DataCopyPad 搬运，消除了 Kernel 内的除法/取模索引计算。

### 2.5 TilingData 结构设计

```cpp
struct AsStridedTilingData {
    uint32_t totalOutputElements;   // 输出总元素数 = product(size)
    uint32_t tileSize;              // 每 tile 处理的元素数
    uint32_t inputTotalElements;    // 输入总元素数
    uint32_t ndim;                  // 维度数 (1-4)
    uint32_t size[4];               // 输出各维度大小
    int32_t stride[4];              // 输出各维度步长（可为负）
    int32_t storageOffset;          // 存储偏移量
    uint32_t dimStride[4];          // 各维度累积步长（Host 预计算 offset 时使用）
    uint32_t pathFlag;              // 0=PathA, 1=PathB
    uint32_t blockDim;              // 使用的核数（Kernel 内通过 GetBlockIdx() 计算各核偏移）
    uint32_t offsetTableInTiling;   // 1=偏移表追加在tiling data中, 0=kernel内增量计算
};
```

**设计说明**：
- `coreElements` 和 `coreOffset` 已移除：直调模型中 TilingData 由 Host 写入一次、所有核共享读取，无法存储每核不同的值。Kernel 内通过 `GetBlockIdx()` 和 `blockDim` 动态计算。
- `dimStride` 保留：Host 侧预计算 offset 表时需要使用，不传入 Kernel。
- `offsetTableInTiling`：标识偏移表是否成功追加到 tiling data。1 表示偏移表在 tiling data 中（Kernel 用 DataCopyPad 搬运），0 表示容量不足需 Kernel 内增量计算。

**dimStride 计算公式**（Host 侧预计算，用于偏移表预计算）：

```cpp
// dimStride[d] = size[d+1] * size[d+2] * ... * size[ndim-1]
// 例如 3D: dimStride[0] = size[1]*size[2], dimStride[1] = size[2], dimStride[2] = 1
dimStride[ndim - 1] = 1;
for (int d = ndim - 2; d >= 0; d--) {
    dimStride[d] = dimStride[d + 1] * size[d + 1];
}
```

**Host 侧偏移表预计算**：

```cpp
// 预计算所有输出元素的偏移表，存入 workspace
uint32_t* offsetTable = workspace;  // Path A: byte offset 表; Path B: srcIdx 表
for (uint32_t f = 0; f < totalOutputElements; f++) {
    // 从平坦索引反推多维索引
    int32_t srcIdx = storageOffset;
    uint32_t remaining = f;
    for (int d = 0; d < ndim; d++) {
        uint32_t i_d = remaining / dimStride[d];
        remaining = remaining % dimStride[d];
        srcIdx += (int32_t)i_d * stride[d];
    }
    // 防御性 clamp：防止负 stride 或越界参数导致 Gather 偏移越界
    if (srcIdx < 0 || (uint32_t)srcIdx >= inputTotalElements) {
        srcIdx = 0;  // clamp 到合法索引，输出值虽错但不触发硬件异常
    }
    if (pathA) {
        offsetTable[f] = (uint32_t)srcIdx * sizeof(T);  // byte offset for Gather
    } else {
        offsetTable[f] = (uint32_t)srcIdx;  // element index for DataCopyPad
    }
}
```

### 2.6 Host 侧 Tiling 流程

```cpp
// 1. 获取属性
auto size = op.GetAttrListInt("input_size");
auto stride = op.GetAttrListInt("input_stride");
auto storageOffset = op.GetAttrInt("input_storage_offset");

// 2. 计算输出总元素数
uint32_t totalOutputElements = 1;
for (auto s : size) totalOutputElements *= s;

// 3. 计算 dimStride
// (如 2.5 节公式)

// 4. 路径判断
uint32_t inputTotalElements = op.GetInputShape(0).TotalCount();
uint32_t inputBytes = inputTotalElements * sizeof(T);
uint32_t ubBudget = 196608 - 8192;  // 192KB - 8KB reserved = 188416
uint32_t tileSize = ...;  // 根据 2.2 节公式计算
uint32_t offsetTileBytes = AlignUp(tileSize * 4, 32);
uint32_t outputTileBytes = AlignUp(tileSize * sizeof(T), 32);
bool pathA = (inputBytes + offsetTileBytes + outputTileBytes * 2 <= ubBudget);  // Double Buffer

// 5. 多核切分
uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
uint32_t blockDim = min(coreNum, totalOutputElements);

// 6. 预计算偏移表并追加到 tiling data
// (如 2.5 节 Host 侧偏移表预计算公式)
// 检查 tiling data 容量，若足够则 Append 偏移表并设置 offsetTableInTiling=1
// 若容量不足则设置 offsetTableInTiling=0（Kernel 内增量计算）

// 7. 填充 TilingData
tiling.totalOutputElements = totalOutputElements;
tiling.tileSize = tileSize;
tiling.inputTotalElements = inputTotalElements;
tiling.ndim = size.size();
tiling.blockDim = blockDim;
// ... 其余字段
```

---

## 3. 确认清单
- [x] 多核切分策略已确定（输出元素一维展平，均分到各核，Kernel 内动态计算 coreOffset）
- [x] UB 切分策略已确定（Path A: 输入全量+偏移表+输出tile×2; Path B: 临时+srcIdx表+输出tile）
- [x] Buffer 规划已完成（Path A/B 各自的 Buffer 布局和大小，Double Buffer 策略）
- [x] 分支场景已覆盖（dtype×3, Path A/B, 正/负/零 stride, 对齐/非对齐, 越界 clamp）
- [x] 类别特有设计已完成（Gather 核心流程 + 逐元素回退流程 + Host 预计算偏移表）
- [x] Workspace 规划已完成（偏移表/srcIdx 表预计算存储）
- [x] 防御性处理已设计（负 stride/越界 srcIdx 的 clamp 策略）

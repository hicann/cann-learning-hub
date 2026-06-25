# Ascend C API 最佳实践速查

从 `ascendc-api-best-practices` skill 整合的核心内容。

---

## ⛔️ API 黑名单

**禁止在生产代码中使用**：

| API | 禁止原因 | 替代方案 |
|-----|---------|---------|
| `GlobalTensor::SetValue()` | 效率极低，单元素逐个写入 | `DataCopyPad` |
| `GlobalTensor::GetValue()` | 效率极低，单元素逐个读取 | `DataCopyPad` |

**仅允许调试时使用**：
```cpp
// ✅ 调试：单点验证
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));
```

---

## DataCopy / DataCopyPad 选择规则

**原则：优先使用 DataCopyPad**

| 场景 | API | 原因 |
|-----|-----|------|
| **非对齐或不确定对齐** | `DataCopyPad` | 自动处理对齐/非对齐，避免边界 bug |
| **数据量严格 32 字节对齐** | `DataCopy` 或 `DataCopyPad` | 确定对齐时 DataCopy 可用，DataCopyPad 更安全 |

### 32 字节对齐要求

| 数据类型 | 对齐元素数 | 最小对齐字节数 |
|---------|-----------|--------------|
| half (2 bytes) | 16 | 32 |
| float (4 bytes) | 8 | 32 |
| int32_t (4 bytes) | 8 | 32 |

### DataCopyPad 使用示例

```cpp
// CopyIn（GM → UB）
AscendC::DataCopyParams copyParams{1, cols * sizeof(float), 0, 0};
AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);

// CopyOut（UB → GM）
AscendC::DataCopyPad(yGm, yLocal, copyParams);
```

---

## TBuf vs TQue 选择

| 场景 | 推荐类型 | 说明 |
|-----|---------|------|
| MTE2/MTE3 搬运缓冲区 | `TQue<VECIN/VECOUT>` | 需要与 Vector 并行，需要 EnQue/DeQue |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | 不涉及 MTE 搬运，用 `Get<T>()` 获取 |
| Double Buffer | `TQue` + `InitBuffer(que, 2, size)` | 在 InitBuffer 中设置 num=2 开启 |

### TQue 正确用法

```cpp
// 模板 depth=1 即可，Double Buffer 在 InitBuffer 的 num 参数中设置
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
pipe->InitBuffer(inQueueX, 2, bufferSize);  // num=2 开启 Double Buffer

AscendC::LocalTensor<half> x = inQueueX.AllocTensor<half>();
AscendC::DataCopyPad(x, xGm, {1, size * sizeof(half), 0, 0}, {false, 0, 0, 0});
inQueueX.EnQue(x);
// ...
AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
inQueueX.FreeTensor(xLocal);
```

### TBuf 正确用法

```cpp
// TBuf：纯计算缓冲区
AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
pipe->InitBuffer(workBuf, bufferSize);

// ✅ 使用 Get<T>() 获取 Tensor，无需释放
AscendC::LocalTensor<float> work = workBuf.Get<float>();
// ... 计算逻辑 ...
// 无需 FreeTensor
```

---

## Double Buffer 流水线并行

### 核心认知

**Double Buffer 不是"用2块内存计算"，而是"用2块内存做搬入/搬出，使 MTE2/MTE3 与 Vector 计算并行"。**

本质：**内存搬运与计算并行，掩盖搬运延迟**。

### 实现原则

| Buffer 类型 | InitBuffer num | 说明 |
|------------|----------------|------|
| `TQue<VECIN>` (MTE2 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TQue<VECOUT>` (MTE3 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TBuf<VECCALC>` (纯计算) | - | TBuf 不涉及 MTE 搬运 |

### TQue Buffer 数量限制

| 产品系列 | eventID 数量 | 最大 TQue 数量 |
|---------|-------------|---------------|
| Atlas A2/A3 系列 | 8 | 8 |

**注意**：
- 不开启 Double Buffer（num=1）：最多可申请 8 个 TQue
- 开启 Double Buffer（num=2）：每个 TQue 占用 2 个 buffer，最多只能申请 4 个 TQue

---

## repeatTime 参数限制

**核心问题**：repeatTime 为 uint8_t 时最大值 255，超过会溢出导致计算错误

### 受影响的 API

| API 类别 | API 名称 | 参数限制 |
|---------|---------|---------|
| 二元运算 | Sub, Add, Mul, Div, Max, Min | `repeatTime` ≤ 255 |
| 一元运算 | Exp, Log, Sqrt, Abs, Neg | `repeatTime` ≤ 255 |
| 标量运算 | Muls, Adds, Divs | `repeatTime` ≤ 255 |

### 解决方案

**Host 侧限制 R_max（推荐）**：
```cpp
constexpr uint32_t MAX_REPEAT_TIMES = 255;
uint32_t R_max = (UB_SIZE - overheadBytes) / bytesPerRow;
R_max = std::min(R_max, MAX_REPEAT_TIMES);  // 确保 R_max <= 255
```

**Kernel 侧分批处理**：
```cpp
constexpr uint32_t MAX_REPEAT = 255;
uint32_t processedRows = 0;
while (processedRows < rowCount) {
    uint32_t batchRepeat = std::min(rowCount - processedRows, MAX_REPEAT);
    // ... 使用 batchRepeat 调用 API
    processedRows += batchRepeat;
}
```

---

## 常见错误速查

| 错误现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| 数据错误（非对齐场景） | 使用 DataCopy 处理非对齐数据 | 改用 DataCopyPad |
| R=256 时计算错误 | repeatTime 溢出 | 限制 R_max ≤ 255 或分批处理 |
| 编译报错 "half precision operation not allowed" | FP16 精度问题 | 转换为 FP32 计算 |

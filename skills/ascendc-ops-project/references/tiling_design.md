# Ascend C 算子 Tiling 设计速查

从 `ascendc-tiling-design` skill 整合的核心内容。

---

## 算子分类体系

| 类别 | 特征 | 典型算子 | Tiling 复杂度 |
|------|------|---------|--------------|
| **Elementwise 逐元素类** | 输入输出Shape相同，逐元素独立计算 | Sin, Cos, Abs, Exp | ⭐ 简单 |
| **Reduction 归约类** | 沿轴归约（含索引跟踪变体） | ReduceSum, Softmax, LayerNorm, ArgMax | ⭐⭐⭐ 复杂 |
| **Broadcast 广播类** | 输入Shape不同，需广播对齐 | Add, Mul, Sub | ⭐⭐ 中等 |
| **Conversion 数据转换类** | 改变布局/形状，合并/拆分张量 | Transpose, Concat, Split | ⭐⭐ 中等 |
| **MatMul 矩阵乘类** | 矩阵乘法，高计算密度，用Cube单元 | MatMul, BatchMatMul | ⭐⭐⭐ 复杂 |
| **Convolution 卷积类** | 空间卷积，滑动窗口计算 | Conv2D, DepthwiseConv | ⭐⭐⭐ 复杂 |

---

## 通用设计要素（所有类别必须）

### 1. 多核切分策略

**核心问题**：任务如何分配给多个 AI Core？

**设计要点**：
- 负载均衡：每个核处理的任务量尽量相等
- 数据局部性：相邻数据尽量分配给同一核
- 粒度适中：tile 不能太小（调度开销大），不能太大（并行度低）

**输出**：
- [ ] 总任务切分方式（按哪个维度切）
- [ ] 每个 AI Core 处理的任务量
- [ ] 使用的 AI Core 数量

### 2. UB 切分策略

**核心问题**：单次能处理多少数据？

**设计要点**：
- UB 容量限制（A2/A3: 192KB, A5: 248KB）
- 单次处理数据量
- 是否需要分 chunk 处理

**输出**：
- [ ] 单次处理的数据量
- [ ] 是否需要分 chunk
- [ ] chunk 大小计算公式

### 3. Buffer 规划

**核心问题**：需要哪些 buffer？各多大？

**设计要点**：
- 输入 buffer（inQueue）
- 输出 buffer（outQueue）
- 中间计算 buffer（tmpBuf, workBuf 等）
- Double Buffer 优化

**输出**：
- [ ] Buffer 列表及用途
- [ ] 各 Buffer 大小计算公式
- [ ] 总 UB 使用量

### 4. 分支场景覆盖

**核心问题**：需要处理哪些不同场景？

**常见分支维度**：
- 数据类型：FP32 / FP16 / BF16 / INT8
- Shape 大小：大 shape / 小 shape
- 数据对齐：32字节对齐 / 非对齐
- 边界情况：最小值 / 最大值 / 特殊值

**输出**：
- [ ] 分支决策条件
- [ ] 各分支的处理策略
- [ ] 边界测试用例

---

## EleWise 类算子 Tiling 设计

### 场景判定

EleWise（Elementwise）：输入输出 Shape 相同，逐元素独立计算，无跨元素依赖。

```
给定: N 个输入 shape + M 个输出 shape

所有输入输出 shape 完全相同？
  ├─ YES → EleWise，展平为 dim0，1D 线性处理
  └─ NO  → Broadcast
```

### 通用规则

- **多核对齐**：元素数对齐到 512 的倍数（ELEM_ALIGN_FACTOR），每个核至少 4KB 数据
- **UB 对齐**：按 256B 对齐，确保 Vector 指令效率
- **区分首/尾 block**：尾 block 数据量可能小于首 block，循环次数和 tail 大小不同

### Tiling 参数计算模板

```cpp
// Host 侧 Tiling 计算
constexpr uint64_t UB_SIZE = 192 * 1024;  // A2/A3 UB 大小
constexpr uint32_t TYPE_SIZE = sizeof(T);
constexpr uint32_t ELEM_ALIGN_FACTOR = 512;

// 1. 计算总元素数
uint32_t totalLength = 1;
for (int i = 0; i < rank; i++) {
    totalLength *= shape_dims.GetDim(i);
}

// 2. 多核切分
uint32_t coreNum = platform.GetCoreNum();
uint32_t totalLengthAligned = ((totalLength + ELEM_ALIGN_FACTOR - 1) / ELEM_ALIGN_FACTOR) * ELEM_ALIGN_FACTOR;
uint32_t avgLengthPerCore = totalLengthAligned / coreNum;
uint32_t tailLength = totalLengthAligned % coreNum;

// 3. UB 切分
// Double Buffer: in*2 + out*2 = 4 个 buffer
uint32_t tileSize = (UB_SIZE / 4) / TYPE_SIZE;
tileSize = (tileSize / ELEM_ALIGN_FACTOR) * ELEM_ALIGN_FACTOR;  // 对齐

// 4. 设置 Tiling 数据
tiling->totalLength = totalLength;
tiling->tileSize = tileSize;
tiling->blockDim = coreNum;
```

---

## Reduction 类算子 Tiling 设计（简要）

### 核心挑战

- **多轴归约**：需要处理 axis 参数，可能归约多个维度
- **中间结果**：需要存储中间归约结果，Buffer 规划复杂
- **数值稳定性**：如 Softmax 需要先减最大值，LayerNorm 需要均值和方差

### 常见模式

| 模式 | 适用场景 | 特点 |
|------|---------|------|
| **逐行处理** | Softmax, LayerNorm | 每行独立计算，可并行 |
| **分块归约** | 大 shape ReduceSum | 分块累积，最后合并 |
| **Welford 算法** | LayerNorm | 单次遍历计算均值和方差 |

---

## Broadcast 类算子 Tiling 设计（简要）

### 核心挑战

- **广播对齐**：不同 shape 的输入需要广播对齐
- **内存访问模式**：小张量需要重复读取，可能需要特殊优化

### 常见模式

| 模式 | 适用场景 | 特点 |
|------|---------|------|
| **标量广播** | Add(x, scalar) | 标量复制到 UB，使用 Adds/Muls |
| **向量广播** | Add([N], [1]) | 小向量复制，使用 ARA 模式 |
| **多维广播** | Add([B,N,C], [1,N,1]) | 需要计算广播 stride |

---

## Tiling 数据结构模板

```cpp
// op_kernel/op_tiling.h
struct TilingData {
    // 通用参数
    uint32_t totalLength;    // 总元素数
    uint32_t tileSize;       // 单次处理元素数
    uint32_t blockDim;       // 核数
    int32_t dtype;           // 数据类型
    
    // EleWise 特有
    // 无额外参数
    
    // Reduction 特有
    uint32_t axis;           // 归约轴
    uint32_t axisLength;     // 归约轴长度
    uint32_t outerLength;    // 外层循环次数
    uint32_t innerLength;    // 内层循环次数
    
    // Broadcast 特有
    uint32_t broadcastDim;   // 广播维度
    uint32_t broadcastSize;  // 广播大小
};
```

---

## 检查清单

### Tiling 设计完成检查

- [ ] 确定算子类别（EleWise/Reduction/Broadcast/...）
- [ ] 设计多核切分策略
- [ ] 设计 UB 切分策略
- [ ] 规划 Buffer（输入/输出/中间）
- [ ] 考虑 Double Buffer 优化
- [ ] 覆盖分支场景（dtype/shape/对齐）
- [ ] 设计边界测试用例

### Host 侧实现检查

- [ ] 正确计算 totalLength
- [ ] 正确设置 blockDim
- [ ] 正确计算 tileSize
- [ ] 正确处理尾核/尾块
- [ ] 正确获取属性和 ValueDepend
- [ ] 正确设置 workspace 大小

### Kernel 侧实现检查

- [ ] 正确使用 blockIdx_ 计算偏移
- [ ] 正确处理尾核/尾块
- [ ] 正确使用 DataCopyPad 处理非对齐
- [ ] 正确使用 Buffer（EnQue/DeQue/FreeTensor）
- [ ] 正确处理多数据类型（模板）

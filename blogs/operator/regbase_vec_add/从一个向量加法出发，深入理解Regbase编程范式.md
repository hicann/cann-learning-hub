# 从一个向量加法出发，深入理解Regbase编程范式

大家好，欢迎来到Ascend C深度学习编程的深度探索之旅。在昇腾AI芯片的开发生态中，Ascend C作为面向AI Core的高性能编程语言，为我们提供了从高层抽象到底层控制的完整编程体系。而在这个体系中，Regbase编程范式就像是一把精密的手术刀——它让我们能够直接操作硬件寄存器，以最细粒度控制数据的流动和计算过程。

在深度学习算子开发中，我们常常面临这样的抉择：是使用封装完善的高层API快速交付，还是深入底层挖掘每一丝性能潜力？对于大多数场景，高层API已经足够优秀；但在追求极致性能的关键路径上，Regbase编程范式就成为了我们的秘密武器。

本文将从硬件内存架构出发，通过一个实际的向量累加案例，逐层拆解Regbase编程的核心原理和最佳实践。无论你是刚接触Ascend C的新手，还是希望深入底层优化的资深开发者，相信这篇文章都能给你带来新的启发。

## 一、为什么需要Regbase？

在AscendC的编程体系中，有两种主要的编程范式：

- **Membase（基于内存）**：使用LocalTensor，API封装程度高，开发便捷
- **Regbase（基于寄存器）**：直接操作向量寄存器，灵活性最大，性能最优

Regbase矢量计算API是专门为RegBase架构设计的，它允许我们直接操作芯片中Vector计算单元的寄存器。与基础API相比，Regbase的核心差异在于：

| 特性 | 基础API | Regbase API |
|------|---------|-------------|
| 输入输出 | LocalTensor | Reg矢量计算寄存器 |
| 控制权 | 封装好的流程 | 用户自主控制数据搬运和计算 |
| 灵活性 | 中等 | 极高 |

## 二、SIMD Vector内存架构深度解析

在开始写代码之前，我们必须理解硬件的内存层级，这是Regbase编程的基础。

### 2.1 内存层级结构

```
┌─────────────────────────────────────────┐
│      Global Memory (GM) 全局内存         │
│           容量大，访问慢                  │
└────────────────┬────────────────────────┘
                 │ DataCopy
                 ↓
┌─────────────────────────────────────────┐
│    Unified Buffer (UB) 统一缓冲区        │
│         单核共享，中等速度                │
└────────────────┬────────────────────────┘
                 │ Load/Store
                 ↓
┌─────────────────────────────────────────┐
│      VF Reg 向量寄存器文件               │
│    SIMD私有内存，最接近计算单元           │
│  ┌───────────────────────────────────┐  │
│  │ RegTensor1 寄存器张量              │  │
│  │ RegTensor2 寄存器张量              │  │
│  │ MaskReg 掩码寄存器                 │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 2.2 关键硬件约束（非常重要！）

**规则一：不能直接从GM加载到寄存器**

SIMD架构不支持从Global Memory直接加载数据到Reg矢量计算寄存器。数据流向必须是：
```
GM → UB → VF Reg → 计算 → VF Reg → UB → GM
```

**规则二：所有VF Reg共享同一个UB**

在一个AI Core内，所有的向量运算单元共享同一个Unified Buffer。这意味着我们需要合理规划UB的使用。

**规则三：寄存器是私有内存**

每个向量运算单元都有自己私有的VF Reg，这为数据并行提供了天然的支持。

## 三、Regbase编程模型

### 3.1 向量函数（Vector Function）

Regbase编程的核心是向量函数，用`__simd_vf__`标记：

```cpp
template <typename T>
__simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, ...)
{
    // 这里面的代码会被发送到向量运算单元执行
    // 使用Reg矢量计算API操作寄存器
}
```

`__simd_vf__`是一个编译器指令，它告诉编译器：
- 这个函数要在向量运算单元上执行
- 函数内部可以使用Reg矢量计算API
- 调用者需要从AI Core的标量控制流调用这个函数

### 3.2 Regbase vs Membase 调用层级对比

让我们通过一个对比图来理解两种范式的差异：

**Membase调用层级：**
```
核函数 → LocalTensor → 基础API（自动处理搬运） → 计算
```

**Regbase调用层级：**
```
核函数 → UB指针 → Load（手动搬运到寄存器） → 计算 → Store（手动搬回UB）
```

这就是Regbase灵活性的来源——你控制每一步。

## 四、深入案例：矩阵逐行累加

现在我们用一个实际案例来展示Regbase编程的完整流程。我们要实现的功能是：

输入一个64×64的矩阵，沿着行方向做逐行累加：
```
y[0] = x[0]
y[1] = x[1] + y[0]
y[2] = x[2] + y[1]
...
y[i] = x[i] + y[i-1]
```

### 4.1 整体架构设计

我们采用经典的三段式结构：

```cpp
__aicore__ inline void Process()
{
    CopyIn();   // 1. GM → UB
    Compute();  // 2. UB → Reg → 计算 → Reg → UB
    CopyOut();  // 3. UB → GM
}
```

### 4.2 CopyIn：数据从GM搬到UB

```cpp
__aicore__ inline void CopyIn()
{
    // 从队列分配LocalTensor（对应UB内存）
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

    // 执行Global Memory到Unified Buffer的搬运
    AscendC::DataCopy(xLocal, xGm, count);

    // 入队，供后续Compute使用
    inQueueX.EnQue<T>(xLocal);
}
```

**关键点：**
- `DataCopy`是异步的，但队列机制会帮我们处理同步
- 使用队列可以方便地扩展为流水并行

### 4.3 Compute：核心计算的深度解析

这是最精彩的部分，让我们逐层拆解：

```cpp
__aicore__ inline void Compute()
{
    // 1. 取出输入Tensor
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    // 2. 先把x复制到y（y[0] = x[0]）
    AscendC::DataCopy(yLocal, xLocal, inner * outter);

    // 3. 获取UB的物理地址——这是连接UB和寄存器的桥梁
    __ubuf__ T* dstAddr = reinterpret_cast<__ubuf__ T*>(yLocal.GetPhyAddr());

    // 4. 计算一次向量操作能处理的数据量
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(T);

    // 5. 计算需要多少次迭代
    uint16_t repeatTimes = AscendC::CeilDivision(inner, oneRepeatSize);

    // 6. 调用真正的Regbase计算
    AddVF(dstAddr, inner, outter, repeatTimes, oneRepeatSize);

    outQueueY.EnQue<T>(yLocal);
}
```

**关于oneRepeatSize的深入理解：**

`AscendC::GetVecLen()`返回向量单元一次能处理的字节数。对于float32：
- 假设向量长度是256字节
- oneRepeatSize = 256 / 4 = 64个float
- 也就是说，一条向量指令能处理64个float的加法

这就是SIMD（单指令多数据）的威力！

### 4.4 AddVF：Regbase编程的精髓

现在让我们深入最核心的向量函数：

```cpp
template <typename T>
__simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, uint32_t inner, uint32_t outter,
                              uint32_t repeatTimes, uint32_t oneRepeatSize)
{
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 步骤一：创建掩码寄存器
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // ALL模式表示所有元素都参与计算
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    // 掩码的作用：在向量计算中，有时候我们只需要处理部分元素
    // 比如最后一次repeat可能不足oneRepeatSize个元素
    // 这时就需要用掩码来mask掉超出范围的元素

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 步骤二：声明寄存器张量
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    AscendC::Reg::RegTensor<T> srcReg;  // 存放第i行
    AscendC::Reg::RegTensor<T> dstReg;  // 存放第i+1行，也是结果

    // RegTensor是对向量寄存器的抽象
    // 它不是一个单一的寄存器，而是一组寄存器的组合
    // 能够容纳oneRepeatSize个T类型的元素

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 步骤三：外层循环（遍历行）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // 注意这里是 outter - 1，因为我们处理i和i+1两行
    for (uint16_t i = 0; i < outter - 1; ++i) {

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // 步骤四：内层循环（遍历一行中的数据块）
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        for (uint16_t j = 0; j < repeatTimes; ++j) {

            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            // 步骤五：Load - 从UB加载到寄存器
            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            // 加载第i行的数据到srcReg
            AscendC::Reg::LoadAlign(srcReg, dstAddr + i * inner + j * oneRepeatSize);

            // 加载第i+1行的数据到dstReg
            AscendC::Reg::LoadAlign(dstReg, dstAddr + (i + 1) * inner + j * oneRepeatSize);

            // LoadAlign要求地址对齐到向量边界
            // 这就是我们前面计算oneRepeatSize并按此切分数据的原因

            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            // 步骤六：Compute - 向量加法
            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            // 执行：dstReg = dstReg + srcReg
            AscendC::Reg::Add(dstReg, dstReg, srcReg, mask);

            // 这是一条向量指令！
            // 在一个时钟周期内，oneRepeatSize个元素同时完成加法
            // 这就是SIMD架构的核心优势

            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            // 步骤七：Store - 从寄存器存回UB
            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            AscendC::Reg::StoreAlign(dstAddr + (i + 1) * inner + j * oneRepeatSize, dstReg, mask);

            // 现在dstReg中的结果已经写回UB了
            // 下一次循环的i+1就是这里的i+1

            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            // 步骤八：Memory Barrier - 内存屏障
            // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE,
                                       AscendC::Reg::MemType::VEC_LOAD>();

            // ⚠️ 这是最关键的一行代码！

            // 为什么需要屏障？
            // 因为下一次循环的i = 当前i + 1
            // 那时的srcReg就是现在的dstReg
            // 我们必须确保Store完成后，才能执行下一次Load

            // 这个屏障的含义：
            // "所有在屏障之前的VEC_STORE操作完成后，
            //  才能执行屏障之后的VEC_LOAD操作"

            // 如果没有这行代码，会产生数据竞争，结果错误！
        }
    }
}
```

## 五、流水线同步控制深度解析

在上面的代码中，我们使用了`LocalMemBar`，这是Regbase编程中同步机制的一种。让我们系统地理解一下同步控制。

### 5.1 为什么需要同步？

现代处理器都是乱序执行的，硬件为了优化性能会重排指令。但在有数据依赖的情况下，我们必须保证某些指令在另一些指令之前完成。

在我们的案例中，数据依赖是：
```
Store（第i+1行结果）→ Load（第i+1行作为下一次的src）
```

### 5.2 LocalMemBar的工作原理

```cpp
LocalMemBar<BeforeType, AfterType>()
```

这个屏障保证：
- 所有类型为`BeforeType`的指令在屏障之前完成
- 然后才能执行类型为`AfterType`的指令

常用的内存类型：
- `VEC_STORE`：向量存储指令
- `VEC_LOAD`：向量加载指令
- `VEC_CMP`：向量比较指令

## 六、Reg矢量计算寄存器详解

### 6.1 MaskReg（掩码寄存器）

掩码寄存器用于控制向量计算中哪些元素参与计算。

```cpp
// 常用掩码模式
MaskPattern::ALL          // 所有元素都参与
MaskPattern::SEQUENCE     // 前N个元素参与
MaskPattern::EVERY_OTHER  // 隔一个参与一个
```

### 6.2 RegTensor（寄存器张量）

RegTensor是一个抽象概念，它代表一组向量寄存器，能够容纳多个数据元素。

```cpp
AscendC::Reg::RegTensor<T> reg;
```

RegTensor的容量由`GetVecLen()`和数据类型T共同决定。

## 七、Regbase编程的最佳实践

### 7.1 数据对齐

始终使用`LoadAlign`和`StoreAlign`，确保地址对齐到向量边界。对齐访问比非对齐访问快得多。

### 7.2 合理使用掩码

不要总是用`ALL`模式。在循环的最后一次迭代中，如果数据不足一个向量长度，应该用合适的掩码来避免越界访问。

### 7.3 减少内存屏障的使用

内存屏障会影响性能，因为它会暂停流水线。只在真正有数据依赖的地方使用屏障。

### 7.4 充分利用向量长度

一次向量操作处理的数据越多，计算效率越高。尽可能按`oneRepeatSize`来组织数据。

## 八、总结与回顾

今天我们从硬件架构出发，深入探讨了Regbase编程范式。让我们总结一下核心要点：

### 核心知识点回顾

1. **内存层级**：GM → UB → VF Reg，理解这个流是基础
2. **向量函数**：用`__simd_vf__`标记，在向量单元执行
3. **Regbase四步曲**：CreateMask → Load → Compute → Store
4. **内存屏障**：在有数据依赖时使用，保证正确性
5. **数据对齐**：对齐访问发挥最佳性能

### Regbase的适用场景

- 需要极致性能的关键路径
- 基础API无法满足的特殊计算模式
- 需要精细控制指令调度的场景
- 对数据布局有特殊要求的算法

Regbase编程虽然更底层，但正是这种对硬件的直接控制能力，让我们能够充分挖掘Ascend芯片的计算潜力。希望这篇深度解析能帮助你更好地理解和使用Regbase！

---

**参考资料：**
- CANN社区版 9.0.0-beta.2 开发文档
- Ascend C算子开发指南 - Reg矢量计算编程
- 本文代码案例参考：https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/00_introduction/04_vector_reg/vector_add

（本文基于CANN asc-devkit样例整理，支持Ascend 950PR/Ascend 950DT产品）

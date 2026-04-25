# 深入理解NDDMA多维数据搬运：昇腾算子开发性能优化利器

---

## 🔥 引言：你还在手动循环搬运数据吗？

在Ascend C算子开发过程中，你是不是经常遇到这些痛点：
- 卷积前需要对特征图Padding，自己写循环填充效率低
- 矩阵转置需要手动计算偏移地址，代码又长又容易出错
- 多维数据切片、广播需要多层循环，算力全浪费在数据搬运上了

今天给大家介绍Ascend C新一代芯片的黑科技：**NDDMA（N-Dimensional DMA，多维直接内存访问）**，它能在数据搬运过程中**硬件自动完成Padding/Transpose/Broadcast/Slice等多种变换**，只用一次API调用就能完成原来几十行代码的工作，性能还提升数倍！

读完本文你将收获：
1. ✅ 理解NDDMA的核心原理和优势
2. ✅ 掌握5种典型NDDMA使用场景
3. ✅ 看懂NDDMA的参数配置方法
4. ✅ 学会在自己的算子中集成NDDMA
5. ✅ 拿到NDDMA最佳实践清单

---

## 🧱 基础概念铺垫：什么是NDDMA？

### NDDMA定义
NDDMA（N-Dimensional DMA）是**昇腾新一代芯片（Atlas 350加速卡及后续产品）提供的硬件加速多维数据搬运功能**，相比于传统的一维DMA搬运，它支持灵活配置每个维度的步长（Stride）和Padding参数，在搬运数据的同时自动完成各种数据变换。

### 官方支持情况
| 产品 | 是否支持NDDMA |
| ---- | ---- |
| Atlas 350 加速卡 | ✅ 完全支持 |
| Atlas A3/A2系列 | ❌ 暂不支持 |
| 老款Atlas推理/训练系列 | ❌ 暂不支持 |

### NDDMA vs 传统DataCopy对比
| 对比项 | 传统一维DataCopy | NDDMA多维DMA |
|--------|-----------------|--------------|
| 支持维度 | 只能一维连续搬运 | 支持N维（最高6维） |
| 数据变换 | 搬完需要软件循环处理 | 搬运+变换一步完成，硬件加速 |
| 代码量 | 需要写多层循环计算地址 | 配置参数，一次API调用完成 |
| 性能 | 软件循环开销大，CPU占用高 | 硬件自动处理，效率提升3~5倍 |
| 适用场景 | 简单连续数据搬运 | 复杂多维数据变换场景 |

---

## 🎯 典型应用场景：5大功能一网打尽

我们以官方样例 `data_copy_gm2ub_nddma` 为例，看看NDDMA能实现哪些常见数据操作：

### 场景1：Padding填充
**需求**：输入 `[16, 32]` 矩阵，四周填充0，输出 `[32, 64]`
```cpp
// 参数配置
AscendC::NdDmaLoopInfo<2> loopInfo{
  {1, 32},      // 每个维度的step
  {1, 64},      // 目的维度step
  {32, 16},     // 源数据总长度
  {15, 13},     // 左/上padding
  {17, 3}       // 右/下padding
};
AscendC::NdDmaParams<float, 2> params{loopInfo, 0}; // padding值为0
AscendC::DataCopy<float, 2>(xLocal, xGm, params);
```
**应用场景**：卷积层输入特征图边界填充，保证输出尺寸不变。

### 场景2：Nearest Padding最近邻填充
**需求**：输入 `[28, 15]` 矩阵，填充到 `[32, 32]`，填充区域使用边界数据而不是0
```cpp
AscendC::NdDmaLoopInfo<2> loopInfo{{1, 15}, {1, 32}, {15, 28}, {11, 3}, {6, 1}};
AscendC::NdDmaParams<float, 2> params{loopInfo, 0};
static constexpr AscendC::NdDmaConfig dmaConfig = {true}; // 开启最近邻填充
AscendC::DataCopy<float, 2, dmaConfig>(xLocal, xGm, params);
```
**应用场景**：图像预处理、对边界精度要求高的算子。

### 场景3：Transpose转置
**需求**：输入 `[16, 64]` 矩阵，转置为 `[64, 16]`
```cpp
// 关键：交换源和目的的stride配置
AscendC::NdDmaLoopInfo<2> loopInfo{{1, 64}, {16, 1}, {64, 16}, {0, 0}, {0, 0}};
AscendC::NdDmaParams<float, 2> params{loopInfo, 0};
AscendC::DataCopy<float, 2>(xLocal, xGm, params);
```
**应用场景**：矩阵乘法前维度转换、Transformer Attention层QKV维度交换。

### 场景4：Broadcast广播
**需求**：输入 `[1, 16]` 行向量，广播为 `[3, 16]` 矩阵
```cpp
// 关键：将被广播维度的源step设置为0，重复读取同一行
AscendC::NdDmaLoopInfo<2> loopInfo{{1, 0}, {1, 16}, {16, 3}, {0, 0}, {0, 0}};
AscendC::NdDmaParams<float, 2> params{loopInfo, 0};
AscendC::DataCopy<float, 2>(xLocal, xGm, params);
```
**应用场景**：偏置添加、归一化层参数广播、向量矩阵运算。

### 场景5：Slice切片
**需求**：从 `[32, 64]` 大矩阵中截取左上角 `[16, 16]` 子块
```cpp
AscendC::NdDmaLoopInfo<2> loopInfo{{1, 64}, {1, 16}, {16, 16}, {0, 0}, {0, 0}};
AscendC::NdDmaParams<float, 2> params{loopInfo, 0};
AscendC::DataCopy<float, 2>(xLocal, xGm, params);
```
**应用场景**：特征图分块计算、大张量切片处理。

---

## ⚙️ 核心代码深度解析：参数怎么配？

NDDMA的核心是 `NdDmaLoopInfo` 这个参数结构体，我们以二维为例拆解每个字段的含义：

```cpp
// 二维NDDMA参数配置
AscendC::NdDmaLoopInfo<2> loopInfo{
  {srcStep0, srcStep1},    // 源数据每个维度的步长（每走一步跳过多少元素）
  {dstStep0, dstStep1},    // 目的数据每个维度的步长
  {srcLen0, srcLen1},      // 源数据每个维度的总长度
  {padLeft0, padLeft1},    // 每个维度左边/上边的padding长度
  {padRight0, padRight1}   // 每个维度右边/下边的padding长度
};
```

### 核心原理：通过步长（Step）实现各种变换
不同操作的本质就是对步长的不同配置：
- **转置**：交换源和目的的步长配置
- **广播**：被广播维度的源步长设为0（重复读取同一位置）
- **切片**：设置源长度为需要的切片大小
- **Padding**：配置左右上下padding参数

---

## 📝 完整实现流程：从输入到输出

我们以完整的官方样例实现为例，看看NDDMA在算子中的集成流程：

### 第一步：定义场景参数
```cpp
#if SCENARIO_NUM == 1
    constexpr uint32_t SRC_TOTAL_LENGTH = 16 * 32;
    constexpr uint32_t DST_TOTAL_LENGTH = 32 * 64;
// 其他场景的参数定义...
#endif
```

### 第二步：核函数实现
```cpp
template <typename T>
class KernelDataCopy {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t xCountIn, uint32_t yCountIn, AscendC::TPipe* pipeIn) {
        // 初始化全局内存指针和队列
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe->InitBuffer(inQueueX, 1, xCount * sizeof(T));
        pipe->InitBuffer(outQueueY, 1, yCount * sizeof(T));
    }

    __aicore__ inline void Process() {
        CopyIn();  // 用NDDMA从GM搬运到UB，同时完成变换
        CopyOut(); // 从UB搬运回GM
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        // 根据不同场景配置NDDMA参数
        if constexpr (scenarioNum == 1) {
            AscendC::NdDmaLoopInfo<2> loopInfo{{1, 32}, {1, 64}, {32, 16}, {15, 13}, {17, 3}};
            AscendC::NdDmaParams<T, 2> params{loopInfo, 0};
            AscendC::NdDmaDci(); // 刷新cache
            AscendC::DataCopy<T, 2>(xLocal, xGm, params);
        }
        // 其他场景的参数配置...
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut() {
        // 结果从UB搬运回GM
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::DataCopy(yLocal, xLocal, yCount);
        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
        
        AscendC::LocalTensor<T> yOutLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm, yOutLocal, yCount);
        outQueueY.FreeTensor(yOutLocal);
    }
};
```

### 第三步：核函数调用
```cpp
__global__ __vector__ void datacopy_custom(GM_ADDR x, GM_ADDR y) {
    AscendC::TPipe pipe;
    KernelDataCopy<float> op;
    op.Init(x, y, SRC_TOTAL_LENGTH, DST_TOTAL_LENGTH, &pipe);
    op.Process();
}
```

---

## 💡 最佳实践总结：避坑指南

### 使用NDDMA的注意事项：
1. **✅ 芯片兼容性检查**：首先确认目标芯片是否支持NDDMA，目前只有Atlas 350及后续产品支持
2. **✅ 维度对齐要求**：不同数据类型有不同的对齐要求，float类型建议对齐到32字节
3. **✅ 合理配置队列深度**：根据数据量大小配置合理的队列深度，避免内存溢出
4. **✅ 调用NdDmaDci刷新cache**：每次NDDMA调用前建议刷新cache，保证数据一致性
5. **✅ 优先使用硬件完成变换**：凡是能通过NDDMA完成的操作，不要用软件循环实现
6. **❌ 不要在NDDMA后立即读取数据**：需要等待DMA搬运完成后再操作数据
7. **❌ 不要配置超过硬件支持的维度数**：目前最高支持6维配置

### 性能优化技巧：
- 尽量将多个连续的小搬运合并成一次NDDMA调用，减少DMA启动开销
- 合理配置Stride参数，尽量让内存访问连续，提高缓存命中率
- 对于复杂场景，可以将多次变换合并到一次NDDMA搬运中完成

---

## 🔚 总结与回顾

今天我们系统学习了昇腾NDDMA多维数据搬运技术：
1. **NDDMA是什么**：硬件加速的多维数据搬运接口，支持搬运+变换一步完成
2. **5大典型场景**：Padding/Nearest Padding/Transpose/Broadcast/Slice
3. **参数配置方法**：核心是NdDmaLoopInfo结构体的5组参数
4. **完整实现流程**：从参数定义到核函数集成的全流程
5. **最佳实践**：使用注意事项和性能优化技巧

NDDMA是昇腾算子开发中提升性能的利器，尤其是对于数据密集型算子，用好NDDMA可以大幅减少数据搬运开销，把算力真正用在计算上。

---

## 📚 参考资料
1. [CANN官方文档：多维数据搬运 (ISASI)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_00131.html)
2. 昇腾官方样例仓库：[data_copy_gm2ub_nddma](https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/02_features/03_basic_api/00_data_movement/data_copy_gm2ub_nddma/README.md)
3. Ascend C算子开发指南

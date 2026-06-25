# 算子开发的守护神：深度解析msSanitizer异常检测工具

大家好，欢迎来到Ascend C算子开发工具深度探索系列。在算子开发过程中，你是否遇到过这样的问题：代码看起来逻辑正确，但一运行就出现神秘的精度错误、内存访问异常、或者莫名其妙的同步失败？

在昇腾AI芯片的开发生态中，msSanitizer（MindStudio Sanitizer）就是我们的"异常检测守护神"。它是一款专门为单算子开发场景设计的工具，能够在早期阶段发现并修复各类异常问题，确保算子的质量和稳定性。

本文将从msSanitizer的核心功能出发，通过一个实际的Add算子案例，深入讲解如何使用这个工具来检测和定位各类异常问题。无论你是刚接触Ascend C的新手，还是希望提升开发效率的资深开发者，相信这篇文章都能给你带来新的启发。

## 一、为什么需要msSanitizer？

在Ascend C算子开发中，我们面临着几类常见的异常问题：

| 异常类型 | 问题表现 | 定位难度 |
|---------|---------|---------|
| 内存类问题 | 非法读写、内存泄漏、非对齐访问 | ⭐⭐⭐⭐⭐ |
| 数据竞争问题 | 流水间/核间竞争导致精度异常 | ⭐⭐⭐⭐ |
| 未初始化问题 | 脏数据读取导致结果错误 | ⭐⭐⭐ |
| 同步问题 | 同步指令未配对导致后续算子失败 | ⭐⭐⭐⭐ |

这些问题往往具有隐蔽性——代码编译通过，甚至在某些情况下运行正常，但在特定条件下就会出现问题。msSanitizer的出现，就是为了帮助我们在早期发现这些问题。

### msSanitizer的四大核心能力

msSanitizer提供四类核心异常检测功能：

1. **内存检测** - 定位非法读写、多核踩踏、非对齐访问、内存泄漏、非法释放等问题
2. **竞争检测** - 发现流水间竞争、流水内竞争、核间竞争等数据竞争问题
3. **未初始化检测** - 检测内存未初始化导致的脏数据读取
4. **同步检测** - 定位前序算子中未配对的同步指令问题

### 使用前提

在使用msSanitizer之前，建议先通过`msOpST`工具在真实硬件环境中完成算子功能测试，确保基本功能正确后，再使用msSanitizer进行深度异常检测。

## 二、msSanitizer使用指南

### 2.1 基本使用流程

1. **环境准备**：参考算子开发工具文档中的"环境准备"章节
2. **编译算子**：使用标准的Ascend C编译流程
3. **生成测试数据**：运行gen_data.py生成输入数据
4. **使用msSanitizer运行**：根据需要选择不同的检测模式
5. **分析报错信息**：根据msSanitizer的输出来定位问题

### 2.2 常用命令汇总

```bash
# 基本检测（非法读写 / 非对齐访问）
mssanitizer ./demo

# 开启内存泄漏检查
mssanitizer ./demo --leak-check=yes

# 开启分配内存未使用检查
mssanitizer ./demo --check-unused-memory=yes

# 竞争检测
mssanitizer ./demo --tool=racecheck

# 未初始化检测
mssanitizer ./demo --tool=initcheck
```

### 2.3 如何读懂msSanitizer报错信息

msSanitizer的报错信息格式是统一的，学会解读这格式就能快速定位问题：

```
====== ERROR: <错误类型> of size <大小>
======    at <地址> on <内存类型> in <算子名>
======    in block <块信息> on device <设备号>
```

**关键信息解读：**
- **错误类型**：illegal write/misaligned access/LeakCheck等，告诉你是什么问题
- **大小**：异常访问的字节数
- **地址**：异常发生的具体内存地址
- **内存类型**：UB（Unified Buffer统一缓冲区）/ GM（Global Memory全局内存）
- **算子名**：哪个算子出的问题
- **块信息**：在哪个AI Core/AIV块
- **设备号**：在哪个设备上

### 2.4 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

### 2.5 使用限制

msSanitizer存在两类不支持的检测场景：
- ❌ 不支持对多线程算子进行检测
- ❌ 不支持对使用了掩码的向量类计算指令进行检测

## 三、最佳实践：如何高效使用msSanitizer

### 3.1 检测顺序建议

建议按照以下顺序进行检测，这样能帮你系统性地排查问题：

1. **先测基本功能**：用msOpST确保算子功能正确
2. **再测内存检测**：发现非法读写、对齐问题等基础问题
3. **然后测泄漏检测**：确保没有内存泄漏
4. **最后测竞争和初始化**：发现更隐蔽的并行问题

### 3.2 常见问题排查清单

| 问题现象 | 可能原因 | 建议检测工具 |
|---------|---------|-------------|
| 运行崩溃 | 非法读写、非对齐访问 | 基础mssanitizer |
| 结果错误 | 数据竞争、未初始化 | --tool=racecheck/initcheck |
| 内存持续增长 | 内存泄漏 | --leak-check=yes |
| 时好时坏 | 数据竞争 | --tool=racecheck |

## 四、msSanitizer内存检测深度解析

现在我们通过一个实际的Add算子案例，来看看msSanitizer如何帮助我们检测各类内存异常。

### 4.1 案例背景：Add算子实现

我们要实现的是一个固定shape为72×4096的Add算子，计算公式很简单：

```python
z = x + y
```

其中：
- x：输入，形状为[72, 4096]，数据类型为float
- y：输入，形状为[72, 4096]，数据类型为float
- z：输出，形状为[72, 4096]，数据类型为float

Add算子的经典三段式实现流程：
1. **CopyIn**：将Global Memory上的xGm和yGm搬运到Local Memory
2. **Compute**：对xLocal和yLocal执行加法操作
3. **CopyOut**：将结果从zLocal搬运至Global Memory

### 4.2 异常场景一：非法读写

**问题描述**：访问了未分配的内存区域

**复现方法**：在DataCopy中使用错误的搬运大小

```cpp
// 1. correct datacopy
AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);    

// 2. illegal read of xGm (TILE_LENGTH*2)
// AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH * 2);
```

**问题分析**：LocalTensor xLocal分配的大小为TILE_LENGTH，但搬运时错误地填为TILE_LENGTH * 2，大于xLocal分配的大小，因此触发非法读写。

**msSanitizer报错**：
```
====== ERROR: illegal write of size 16384
======    at 0x0 on UB in add_custom
======    in block aiv(0-7) on device 0
```

按照我们前面讲的解读方式：
- `illegal write of size 16384`：检测到16384字节的非法写入
- `at 0x0 on UB`：异常发生在Unified Buffer的0x0地址
- `in add_custom`：在add_custom算子中
- `in block aiv(0-7)`：在AIV块0-7中

### 4.3 异常场景二：非对齐访问

**问题描述**：内存访问未满足字节对齐要求

**复现方法**：使用未对齐的UB侧地址

```cpp
// 1. correct datacopy
AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);    

// 3. misaligned access of xLocal (should be 32Byte aligned)
// AscendC::DataCopy(xLocal[5], xGm[i * TILE_LENGTH], TILE_LENGTH);
```

**问题分析**：DataCopy GM->UB的搬运中，UB侧地址需要满足32B对齐要求，但xLocal[5]的地址是5 * sizeof(float) = 20字节，不满足32B对齐，因此触发非对齐访问异常。

**msSanitizer报错**：
```
====== ERROR: misaligned access of size 32
======    at 0x14 on UB in add_custom
======    in block aiv(0-7) on device 0
```

**硬件背景知识**：
在昇腾AI芯片中，Unified Buffer的访问通常需要满足32字节对齐。非对齐访问会导致性能下降，甚至硬件异常。msSanitizer能够精准检测这类问题。

### 4.4 异常场景三：内存泄漏

**问题描述**：申请内存使用后未释放，导致程序运行过程中内存占用持续增加

**复现方法**：注释掉内存释放代码

```cpp
// 1. correct free for memory. If deleted, it will trigger memory leak check.  
aclrtFree(tilingDevice);
```

**问题分析**：注释掉aclrtFree后，tilingDevice在使用后未释放，因此触发内存泄漏检测。

**注意**：调用mssanitizer时需要传入`--leak-check=yes`来开启分配内存泄露检查。

**msSanitizer报错**：
```
====== ERROR: LeakCheck: detected memory leaks

======    Direct leak of 64 byte(s)
======      at 0x12c0c0013000 on GM
======      allocated in :0 (serialNo:0)
```

### 4.5 异常场景四：分配内存未使用

**问题描述**：对内存分配后未使用，导致资源浪费

**复现方法**：分配超过实际需要的内存

```cpp
// 1. correct malloc for memory
aclrtMalloc((void **)(&inputDevice[i]), inputsInfo[i].length, ACL_MEM_MALLOC_HUGE_FIRST);

// 2. needed inputsInfo[i].length, but malloc length * 5, therefore trigger unused memory
// aclrtMalloc((void **)(&inputDevice[i]), inputsInfo[i].length * 5, ACL_MEM_MALLOC_HUGE_FIRST);
```

**问题分析**：inputDevice[i]实际需要分配的大小为inputsInfo[i].length，但分配了inputsInfo[i].length * 5，其中有inputsInfo[i].length * 4未使用，因此触发分配内存未使用检测。

**注意**：调用mssanitizer时需要传入`--check-unused-memory=yes`来开启分配内存未使用检查。

**msSanitizer报错**：
```
====== WARNING: Unused memory of 4718624 byte(s)
======    at 0x12c041200000 on GM
======    code in :0 (serialNo:260)
```

## 五、竞争检测：发现并行计算中的隐蔽问题

在多核或流水并行的场景中，数据竞争是一个常见但难以定位的问题。msSanitizer的竞争检测功能能够帮助我们发现这类问题。

### 5.1 什么是数据竞争？

数据竞争发生在：
- 多个执行单元（不同流水线、不同核）同时访问同一块内存
- 至少有一个访问是写操作
- 没有正确的同步机制来保证访问顺序

在Add算子的流水并行实现中，我们使用SetFlag和WaitFlag来保证MTE2 GM->UB的搬运和Vector计算Add的时序。

### 5.2 异常场景：流水间竞争

**问题描述**：删除同步指令，导致先计算后搬运数据

**复现方法**：注释掉SetFlag和WaitFlag

```cpp
// dependency of PIPE_MTE2 & PIPE_V caused by xLocal/yLocal in one single loop
// If SetFlag and WaitFlag are deleted, will trigger RAW 
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
```

**问题分析**：SetFlag和WaitFlag用来保证MTE2搬运和Vector计算的时序。删除后可能会导致先计算Add再搬运数据，导致读取到旧数据（RAW hazard），因此触发竞争检测。

**注意**：调用mssanitizer时需要传入`--tool=racecheck`来开启竞争检测。

**msSanitizer报错**：
```
====== ERROR: Potential RAW hazard detected at UB in add_custom on device 0:
======    PIPE_MTE2 Write at RAW()+0x4000 in block 0 (aiv) on device 0 at pc current 0xd08 (serialNo:25)
======    xxxxx
======    PIPE_V Read at RAW()+0x4000 in block 0 (aiv) on device 0 at pc current 0x1578 (serialNo:28)
======    xxxxx
```

**如何理解RAW hazard**：
- RAW = Read After Write（写后读）
- PIPE_MTE2在地址0x4000处写入
- PIPE_V在同一地址处读取
- 但没有正确的同步保证写完成后再读

## 六、未初始化检测：防止脏数据引发的神秘错误

内存申请后未初始化就直接读取，是导致算子精度异常的常见原因之一。这类问题特别隐蔽，因为未初始化内存的值是不确定的——有时候恰好是0，结果看起来正常；有时候是随机值，结果就错误了。

### 6.1 异常场景：未初始化读取

**问题描述**：内存申请后为未初始化状态，未对内存进行写入，直接读取未初始化的值

**复现方法**：注释掉GlobalBuffer的初始化

```cpp
// correct initialize of zGm. 
// If deleted, it will trigger uninitialized read
zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
```

**问题分析**：对于device侧的zGm在使用前未初始化，因此触发未初始化检测。

**注意**：调用mssanitizer时需要传入`--tool=initcheck`来开启未初始化检测。

**msSanitizer报错**：
```
====== ERROR: uninitialized read of size 1179648
======    at 0x12c041600000 on GM
```

## 七、总结与回顾

今天我们深入探讨了msSanitizer异常检测工具，这是Ascend C算子开发中的"守护神"。让我们总结一下核心要点：

### 核心知识点回顾

1. **四大检测能力**：内存检测、竞争检测、未初始化检测、同步检测
2. **如何读报错**：掌握统一报错格式，快速定位问题位置
3. **内存检测**：非法读写、非对齐访问、内存泄漏、未使用内存
4. **竞争检测**：发现流水间、流水内、核间的数据竞争
5. **未初始化检测**：防止脏数据读取导致的神秘错误
6. **命令选项**：根据需要选择不同的检测模式

### msSanitizer的价值

- **早期发现问题**：在开发阶段就发现问题，避免线上故障
- **精准定位**：提供详细的报错信息，帮助快速定位
- **全面覆盖**：覆盖内存、竞争、初始化等各类异常
- **易于使用**：简单的命令行接口，上手快速

msSanitizer虽然只是一个工具，但正是这样的工具，让我们的算子开发过程更加稳健和高效。希望这篇深度解析能帮助你更好地使用msSanitizer，写出更高质量的Ascend C算子！

---

**参考资料：**
- CANN社区版 9.0.0-beta.2 开发文档 - 异常检测（msSanitizer）
- 本文代码案例参考：https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/01_utilities/05_sanitizer/msSanitizer

（本文基于CANN asc-devkit样例整理，支持Atlas A3/A2训练和推理系列产品）

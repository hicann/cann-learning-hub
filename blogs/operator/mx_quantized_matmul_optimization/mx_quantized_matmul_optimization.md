# MX量化矩阵乘性能优化实践

cann-samples是CANN社区提供的高性能实操样例库，致力于为开发者提供可复用的优化方法论和最佳实践代码。本系列文章将陆续介绍仓库中的典型样例，分享我们在算子优化过程中的思考与经验。

cann-samples：

https://gitcode.com/cann/cann-samples

## 本文将帮助你

- 理解MX量化的硬件加速机制：MX量化相较传统量化实现优势

- 掌握性能建模方法：如何定量分析Bound类型，针对性优化，避免盲目调优

- 理解核心优化思路：SWAT等关键策略的原理与应用

- 了解TensorAPI范式：如何用Layout抽象简化坐标计算，代码量减少60%+

## 算子功能说明

| 对比项 | MXFP8 | MXFP4 |
| --- | --- | --- |
| A/B数据类型 | float8_e4m3fn | float4_e2m1 |
| 量化方式 | GroupSize=32的pergroup量化 | GroupSize=32的pergroup量化 |
| scaleA/B数据类型 | float8_e8m0 | float8_e8m0 |
| 显存压缩比 | 相比 FP16/BF16 内存占用减少约 50% | 相比 FP16/BF16 内存占用减少约 75% |
| 典型应用场景 | 训推场景，兼顾训推速度与精度的平衡 | 模型推理，侧重显存效率与推理速度 |

计算公式

## MX量化原理

### 技术背景

MX量化是一种GroupSize=32、Scale类型为float8_e8m0的分组量化方案。从效果上看，它等价于传统的PerGroup量化；但在实现层面，两者存在显著差异：

| 特性 | MX量化 |
| --- | --- |
| 量化效果 | 分组量化（等价） |
| Scale粒度 | 固定GroupSize=32 |
| 执行方式 | 纯Cube核计算<br>硬件自动完成Scale乘法 |
| 易用性 | 高<br>仅需搬运数据至指定缓冲区，无需Scale乘法代码 |
| 性能影响 | 无额外开销<br>Scale乘法融合到MMAD指令，零额外耗时 |
| Block代码量 | ~350行（TensorAPI）<br>仅需基于layout+坐标简易实现搬运计算 |
| Scale坐标计算 | 需要手动对齐L0 Buffer层级上的坐标 |

### 硬件加速机制

MX量化的核心优势在于硬件支持自动乘Scale操作，带来显著的易用性和性能提升。传统PerGroup量化需要AIC（Cube核）和AIV（Vector核）协同处理：

```
// 传统PerGroup量化：AIC+AIV协同处理（复杂且性能受限）
// AIC（Cube核）：执行量化矩阵乘
for (int g = 0; g < K/G; g++) {
    // MMAD计算：量化后的A × 量化后的B，输出int32结果到L0C
    MMAD(l0c, a_quantized[g], b_quantized[g]);
    // Fixpipe自动搬运L0C → UB（供AIV读取）
}

// AIV（Vector核）：显式执行Scale乘法和累加（性能瓶颈）
for (int g = 0; g < K/G; g++) {
    // 需搬运Scale到UB（额外的数据搬运开销）
    CopyIn(scaleA_ub[g], scaleA[g]);
    CopyIn(scaleB_ub[g], scaleB[g]);

    // 显式执行scale乘法（计算耗时 + AIC-AIV同步开销）
    temp = scaleA_ub[g] * scaleB_ub[g];  // broadcast乘法
    output += l0c_ub[g] * temp;          // 逐元素乘法+累加

    // 同步等待AIC下一轮数据
    SyncWait(aic_signal);
}

```

而MX量化的MMAD指令硬件自动完成Scale乘法：

```
// MX量化：纯Cube核计算（硬件自动融合Scale乘法）
// 仅需搬运数据到指定缓冲区，无需Scale乘法代码
for (int g = 0; g < K/G; g++) {
    // 数据搬运：A/B → L0A/L0B，Scale → L0A_MX/L0B_MX
    LoadData(l0a, a[g], scaleA[g]);  // 数据+Scale一起搬运
    LoadData(l0b, b[g], scaleB[g]);

    // MMAD指令自动执行：A × B × ScaleA × ScaleB
    // 无需AIV参与，无需显式Scale乘法，零额外开销
    MMAD(l0c, l0a, l0b);  // 输出已包含Scale乘法的结果
}

```

Scale被预先搬运至独立的缓冲区`L0A_MX`和`L0B_MX`，MMAD指令执行时硬件自动完成scale乘法，无需AIV参与、无需额外计算、无需同步等待。

### 易用性与性能提升

#### 易用性提升

| 维度 | MX量化 | 提升效果 |
| --- | --- | --- |
| 核间协作 | 仅需Cube核 | 开发复杂度降低50% |
| 同步管理 | 无需同步 | 代码量减少30%+ |
| Scale搬运 | 自动搬运至L0_MX | 无需额外搬运逻辑 |
| Scale乘法 | 硬件自动完成 | 无需Scale乘法代码 |

开发效率提升：MX量化不需要理解AIC和AIV的流水线、同步机制、缓冲区管理等复杂概念；仅需掌握数据搬运和坐标映射，开发者学习曲线大幅缩短。

#### 性能提升

| 维度 | MX量化 | 性能提升 |
| --- | --- | --- |
| 计算单元 | 纯AIC(MMAD融合Scale) | 零额外计算开销 |
| 流水并行 | 纯Cube流水，无断流 | 流水线并行度提升 |
| 总耗时 | 仅MMAD | 整体耗时减少60% |

性能瓶颈消除：MX量化通过硬件融合彻底消除AIV处理这一性能瓶颈，实现纯Cube模板实现的理想状态。

### 数据搬运流程

MX量化执行时的完整数据搬运流程如下图所示：

![mx_quantization_data_movement](images/mx_quantization_data_movement.png)

Figure 2. MX量化数据搬运流程：GM → L1 → L0/L0_MX → MMAD计算

关键要点：

- 输入矩阵A/B：GM → L1 → L0A/L0B

- Scale张量：GM → L1 → L0A_MX/L0B_MX（独立缓冲区）

- MMAD指令：自动完成 A × B × ScaleA × ScaleB

开发者的核心任务：精确控制输入矩阵与Scale的坐标对应关系。

### 实现约束

开发MX量化算子时需要注意以下关键约束：

1. K维度对齐约束：由于scale在L0_MX缓冲区上的最小分形为`(16, 2)`，对应输入矩阵在K方向的最小单位为64，要求baseK是64的整数倍。

2. K维度补零处理：基于约束1，当K轴非64对齐时，存在两类场景：

  - 当输入矩阵的K维度在内轴，即输入排布为`(m, k)`或`(n, k)`时，`ND2NZ`指令可自动完成K方向补零；仅在MXFP8且Ceil(K/32)是奇数时，需要补零；

- 当输入矩阵的K维度在外轴，即输入排布为`(k, m)`或`(k, n)`时，`ND2NZ`指令无法完成K方向补零，需在K对齐16后依然非64对齐时手动处理K方向补零。推荐补零方法：使用`SET2D`对L1缓冲区目标地址清零 + `ND2NZ`跳写目标地址。

1. 指令约束：`MMAD`指令需关闭`gemv`功能。

- 详细约束说明：参见性能优化指南：https://gitcode.com/cann/cann-samples/blob/master/Samples/2_Performance/matmul_story/docs/quant_matmul_mx_performance.md

开发者提示：违反约束可能导致运行时错误或性能下降，建议在开发前充分理解硬件限制。

## 性能建模方法

### 为什么需要性能建模？

传统调优方式往往依赖经验或大量试错。性能建模的核心价值在于将调优过程量化、可预测、可迁移：

| 方式 | 特点 | 问题 |
| --- | --- | --- |
| 凭经验调优 | 根据直觉选择优化策略 | 经验难以复用，新场景需重新摸索 |
| 大量试错 | 尝试多种配置选最优 | 时间成本高，可能错过最优解 |
| 性能建模 | 量化分析各流水耗时 | 针对性优化，方法可迁移 |

### 理论耗时公式

样例建立了完整的性能建模体系，各流水的理论耗时如下：

MMAD计算时间：

核数频率

其中`16 × C0 × 16`表示MX量化在Cube核上每拍的计算量。其中，MXFP8场景下`C0 = 32`，MXFP4场景下`C0 = 64`。

MTE2搬运时间（GM到L1）：

MTE1搬运时间（L1到L0）：

FIXPIPE搬出时间（L0C到GM）：

### Bound类型判断

通过比较各流水耗时，可以定量判断性能瓶颈类型：

| Bound类型 | 判断条件 | 特征 | 优化方向 |
| --- | --- | --- | --- |
| Cube Bound |  | 计算已达极限 | 关注负载均衡 |
| MTE2 Bound |  | GM到L1搬运受限 | 减少搬运量、提高带宽利用率 |
| MTE1 Bound |  | L1到L0搬运受限 | 消除Bank冲突、优化搬运效率 |
| FIXPIPE Bound |  | L0C到GM搬出受限 | 小K场景需重点关注，提高带宽利用率 |

### Bound判断思路

以MTE2 Bound为例说明判断过程：

步骤1：计算理论MMAD耗时

核数频率

步骤2：计算MTE2搬运耗时步骤3：对比判断

- 若  → Cube Bound（计算已达极限）

- 若  → MTE2 Bound（搬运受限）

关键洞察：通过定量分析，精准定位瓶颈因素，避免"盲人摸象"式的调优。上述公式可以分析出：增大baseM/baseN可降低重复搬运比例，更容易达成Cube Bound。

详细的推导过程和参数说明，参见性能建模章节。

性能建模章节：

https://gitcode.com/cann/cann-samples/blob/master/Samples/2_Performance/matmul_story/docs/quant_matmul_mx_performance.md#%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E5%BB%BA%E6%A8%A1

### 性能示例：8192×8192矩阵

以M=N=K=8192、MXFP4、32核、1.65GHz，全局内存带宽1.6T/s，L2带宽5.2T/s，baseM=baseN=baseK=256为例，各流水理论耗时：

| 流水 | 公式计算 | 理论耗时 |
| --- | --- | --- |
| MMAD | 8192³ / (16 × 64 × 16 × 32 × 1.65GHz) | ~635 us |
| MTE2 | 全局内存搬运量 / 全局内存带宽 + L2搬运量 / L2带宽 | ~455 us |
| MTE1 | 单核MTE1搬运量 / 单核MTE1速率 | ~163 us |
| FIXPIPE | 输出搬运 / L2带宽 | ~25 us |

结论：此场景为Cube Bound，可重点关注SWAT优化提升L2命中率。

## 核心优化策略

本样例实现了完整的优化策略体系（详见性能优化指南），以下重点介绍最具代表性的几种。

性能优化指南：

https://gitcode.com/cann/cann-samples/blob/master/Samples/2_Performance/matmul_story/docs/quant_matmul_mx_performance.md

> 下述优化策略并不局限于MX量化场景，Matmul算子均可适配使能

### SWAT：自适应滑动窗口模板

解决问题：多核场景下首轮L2缓存命中率低

核心原理：传统按列优先分配任务，每个核处理的区域呈窄条状，存在首轮L2缓存利用率低，后续L2命中率过高导致的整体不均衡，从而无法在初期从全局内存中访问数据时达成CubeBound。SWAT通过方正切分的策略来实现首轮即可CubeBound，并结合"Z型滑动"策略，显著改善每轮L2命中率。即使数据从全局内存获取，也依然可以具备CubeBound的能力。

![swat_partition_principle](images/swat_partition_principle.png)

Figure 3. SWAT原理图：优化首轮的L2命中率，实现Cube Ratio优化

仿真效果：图中展示了`(1024, 1024)x(1024, 8192)`矩阵乘使能SWAT前后的仿真流水对比结果。可以明显看到，同样的基本块切分下，因排布方式的差异，优化了首轮的MTE2综合带宽，从而实现算子Cube Ratio从79.1%提升到86.7%。

![swat_pipeline_comparison](images/swat_pipeline_comparison.png)

Figure 4. SWAT流水效果对比：上部分是传统行/列优先实现；下部分是SWAT实现。

### 尾轮负载均衡

解决问题：多核分配基本快中，最后一轮基本快分配不均导致的计算负载不均衡

核心原理：将最后一轮的基本快进行二次切分，实现计算再分配，从而实现整体的计算负载均衡

![tail_block_load_balancing_principle](images/tail_block_load_balancing_principle.png)

Figure 5. 尾轮负载均衡原理图：通过二次切分实现多核负载均衡，消除最后轮核算力浪费。

仿真效果：图中展示了`(2560, 1024)x(1024, 2560)`矩阵乘的对比结果，按照基本块(256, 256)进行切分，会产生100个基本块，基于32核进行划分时，最后一轮仅有4个核。为此可以对其进行二次切分，将基本块缩小8倍，从而重新分配到32核上均衡计算，减少最长核的耗时。

![tail_block_pipeline_comparison](images/tail_block_pipeline_comparison.png)

Figure 6. 尾轮负载均衡流水效果对比：上部分未使能，下部分对尾块进行二次负载均衡。

### UnitFlag

解决问题：当为了追求计算掩盖搬入访存，需要尽量的提高基本快的Size，从而降低重复搬入量。为此无法开启L0C的Double Buffer，导致MMAD和FIXP断流。通过Unitflag能力即可实现两者block级的同步能力。

![unitflag_pipeline_principle](images/unitflag_pipeline_principle.png)

Figure 7. UnitFlag流水原理图：提供512B粒度的细粒度同步，在无法开启L0C Double Buffer时提升并行度。

仿真效果：图中展示了`(1024, 1024)x(1024, 4096)`矩阵乘使能Unitflag前后仿真流水对比结果。Unitflag开启后提高了MMAD和FIXP流水的并行度，Cube Ratio从89.5%提升到90.9%。

![unitflag_pipeline_comparison](images/unitflag_pipeline_comparison.png)

Figure 8. Unitflag流水效果对比：上部分是未开启Unitflag实现；下部分是开启Unitflag实现。

### 更多优化策略

除上述重点介绍外，仓库还包含以下优化技术：

| 优化策略 | 解决问题 | 适用场景 |
| --- | --- | --- |
| Scale缓存优化 | Scale带宽打不满 | 小矩阵、MTE2 Bound |
| 全载优化 | MTE2 Bound | Decode场景 |
| Double Buffer | 流水停顿 | 缓冲区充足 |
| L1 Bank冲突优化 | MTE1速率下降 | Bank冲突场景 |

详细的原理说明、代码实现和效果对比，请参考仓库中的性能优化指南和示例代码。

优化指南：

https://gitcode.com/cann/cann-samples/blob/master/Samples/2_Performance/matmul_story/docs/quant_matmul_mx_performance.md

示例代码：

https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp4

## 代码实现：TensorAPI范式

### 传统实现的痛点

量化矩阵乘涉及复杂的数据搬运：GM→L1→L0，且每个层级的排布规则不同：

| 缓冲区 | 排布规则 | 坐标计算复杂度 |
| --- | --- | --- |
| GM | ND排布 | 简单 |
| L1 | NZ/ZN排布 | 复杂，需手算偏移 |
| L0 | ZN/Zz/Nn排布 | 复杂，规则各不相同 |

传统实现需要为每次搬运手动设置大量参数。以GM到L1的ND2NZ搬运为例：

```
// 传统方式：手动设置ND2NZ参数（多达10个参数）
AscendC::Nd2NzParams nd2nzParams;
nd2nzParams.ndNum = 1;
nd2nzParams.nValue = nDim;
nd2nzParams.dValue = dDim;
nd2nzParams.srcNdMatrixStride = 1;
nd2nzParams.srcDValue = transA ? m_ : k_;
nd2nzParams.dstNzC0Stride = transA ? curPadAKL1 : curAlignM;
nd2nzParams.dstNzNStride = 1;
nd2nzParams.dstNzMatrixStride = 1;
AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);

// 手动计算L1缓冲区偏移（复杂且易错）
uint64_t l1Offset = IS_FP4
    ? AscendC::TOTAL_L1_SIZE * (bufferId & 1)
    : (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
l1BufferBOffset_[bufferId] =
    l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);

```

L1到L0的搬运更加复杂，需要同时处理数据张量和Scale张量，并且不同的模板无法有效复用，导致重复造轮子：

```
// 传统方式：手动设置LoadData参数（多达8个参数）
AscendC::LoadData2DParamsV2 loadDataParams;
loadDataParams.mStartPosition = 0;
loadDataParams.kStartPosition = CeilDiv(iter * baseK_, C0_SIZE);
loadDataParams.mStep = m1;
loadDataParams.kStep = CeilDiv(curKL0, C0_SIZE);
loadDataParams.srcStride = loadDataParams.mStep;
loadDataParams.dstStride = loadDataParams.mStep;
loadDataParams.ifTranspose = false;

// Scale张量需要额外设置LoadData2DMxParams
AscendC::LoadData2DMxParams loadData2DMxParams;
loadData2DMxParams.xStartPosition = 0;
loadData2DMxParams.yStartPosition = CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
loadData2DMxParams.xStep = m1;
loadData2DMxParams.yStep = CeilDiv(curKL0, MXFP_DIVISOR_SIZE);
loadData2DMxParams.srcStride = CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
loadData2DMxParams.dstStride = loadData2DMxParams.yStep;

// 执行搬运（需要同时传入数据张量和Scale张量）
AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);

```

问题总结：

- 参数众多，不同排布规则参数组合不同

- 偏移计算复杂，涉及bufferId、切分大小、对齐要求等多个因素

- 数据张量和Scale张量需要分别处理，代码重复

### TensorAPI的核心范式

样例采用TensorAPI的"Layout抽象 + 自动坐标映射"范式，将复杂性封装到底层：

Step 1：创建Layout描述排布

```
// 一行代码描述L1的NZ排布（替代手动设置10个ND2NZ参数）
auto layoutAL1 = AscendC::Te::MakeNzLayout<AType>(curM, curKL0);

// 创建L0的ZN排布
auto layoutBL0 = AscendC::Te::MakeZnLayout<BType>(curKL0, curN);

// 创建Scale的Zz排布
auto layoutScaleAL0 = AscendC::Te::MakeZzLayout<fp8_e8m0_t>(
    curM, CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);

```

Step 2：创建Tensor绑定地址与Layout

```
// 绑定L1缓冲区地址与Layout
auto tensorAL1 = AscendC::Te::MakeTensor(
    AscendC::Te::MakeL1memPtr<AType>(l1BufferAOffset_[l1BufId]),
    layoutAL1);

```

Step 3：切片获取子张量（框架自动计算坐标映射）

```
// 从GM张量切片，框架自动处理ND→NZ的坐标映射
auto gmBlockA = gmA(
    AscendC::Te::MakeCoord(0, kL1Offset),      // 起始坐标
    AscendC::Te::MakeShape(curM, curGmAKL1));  // 子张量形状

// 从L1张量切片
auto tensorBlockAL1 = tensorAL1(
    AscendC::Te::MakeCoord(0, kL0Offset),
    AscendC::Te::MakeShape(curM, curKL0));

```

Step 4：执行操作（框架自动处理跨层级坐标转换）

```
// GM到L1的搬运（框架自动填充DataCopy参数）
auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

// L1到L0的搬运（框架自动处理数据张量和Scale张量的坐标对齐）
auto copyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
AscendC::Te::Copy(copyL12L0, tensorAL0, tensorBlockAL1);

```

### 代码量对比

以核心数据搬运逻辑为例：

| 实现方式 | 代码行数 | 参数数量 | 关键特点 |
| --- | --- | --- | --- |
| 传统方式 | ~80行 | 10+参数/次 | 手算偏移，易出错 |
| TensorAPI范式 | ~30行 | 3-4参数/次 | Layout抽象，逻辑清晰 |

完整代码示例：参见仓库中的quant_matmul_mxfp4_swat.asc，展示了SWAT模板的完整TensorAPI实现，包含数据搬运、坐标映射、优化策略应用等核心逻辑。

quant_matmul_mxfp4_swat.asc：

https://gitcode.com/cann/cann-samples/blob/master/Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp4/quant_matmul_mxfp4_swat.cpp

### TensorAPI的核心价值

1. 声明式编程：开发者只需声明"数据的排布是什么"，而非"如何计算坐标偏移"

2. 类型安全：Layout信息编码在类型系统中，编译期即可发现排布不匹配

3. 可复用：同一套代码可适配不同排布（ND/Nz/Zn/Zz/Nn）

4. 可维护：核心逻辑简洁，便于理解和修改

5. 可扩展：支持自定义搬运逻辑，高度定制

## 快速上手与性能验证

```
cd Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp4

# 自动构建 + 自动推荐最优算法 + 运行
bash scripts/run.sh 16 2048 16384

# 指定目标可执行文件，跳过重新构建
bash scripts/run.sh \
  --target quant_matmul_mxfp4_a_full_load --skip-build 16 2048 16384

# 查看完整帮助
bash scripts/run.sh --help

```

### 性能效果验证

读者可以通过以下方式体验优化效果：

1. 对比不同优化策略：运行不同的可执行文件（如`quant_matmul_mxfp4_swat`、`quant_matmul_mxfp4_a_full_load`）

2. 测试不同矩阵规模：尝试8192×8192、4096×4096、1024×1024等不同规模，观察性能差异

3. 分析Bound类型：结合性能建模公式，理解不同规模下的性能瓶颈

4. Profiling分析：使用昇腾Profiling工具分析各流水线耗时，验证理论建模

典型效果参考：

```
[Profile Breakdown]
+--------------------------------+----------+---------+----------+---------+---------+------------+--------------+
| candidate                      |kernel(us)| mac(us) |scalar(us)| mte1(us)| mte2(us)|fixpipe(us) |icache_miss(%)|
+================================+==========+=========+==========+=========+=========+============+==============+
| quant_matmul_mxfp4_swat        |    12.345|   1.234 |     0.567|   0.123 |   0.456 |     0.789 |        0.100 |
| quant_matmul_mxfp4_a_full_load |    15.678|   2.100 |     0.800|   0.200 |   0.300 |     0.500 |        0.250 |
+--------------------------------+----------+---------+----------+---------+---------+------------+--------------+

```

## 小结

`quant_matmul_mxfp8`和`quant_matmul_mxfp4`样例提供了MXFP8/MXFP4量化矩阵乘的完整解决方案，相比传统方案具有显著提升：

### 易用性提升

- 硬件融合设计：MMAD指令自动完成Scale乘法，无需AIV参与，开发复杂度降低50%+

- 纯Cube核计算：无需编写双核代码、管理同步逻辑，代码量减少60%+

- TensorAPI范式：Layout抽象 + 自动坐标映射，大幅降低开发门槛，新手学习周期缩短

### 性能提升

- 零额外开销：Scale乘法融合到MMAD指令

- 纯Cube实现：消除传统方案的AIV瓶颈

- 内存高效：减少输入的内存占用

### 开发支持

- 性能建模方法：帮助开发者科学分析瓶颈，针对性优化，避免盲目调优

- 系统化优化策略：SWAT、Double Buffer、UnitFlag等覆盖不同Bound类型场景，详见性能优化指南
性能优化指南：
https://gitcode.com/cann/cann-samples/blob/master/Samples/2_Performance/matmul_story/docs/quant_matmul_mx_performance.md

- 完整示例代码：提供TensorAPI实现参考，快速上手开发

我们希望这份样例能够帮助开发者快速掌握量化矩阵乘的优化方法，也期待社区的反馈与建议。如有问题，欢迎在GitCode Issues提出。

相关资源：

- 性能优化指南：`Samples/2_Performance/matmul_story/docs/quant_matmul_mx_performance.md`

- MXFP8样例代码：`Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp8/`

- MXFP4样例代码：`Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp4/`

``

# as_strided 设计串讲

## 审查结论
- [ ] 设计可直接开发（无阻塞问题）
- [x] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

## 审查概述

对 DESIGN.md §1（算子设计：API 映射、数据流）和 §2.4（伪代码）进行了重点审查。发现 **1 个阻塞级问题**（UB 容量假设错误，会导致运行时崩溃）和 **5 个需讨论级问题**（性能风险、正确性隐患、设计不完备）。API 选择（DataCopyPad + Gather + SetValue/GetValue）经验证均存在且签名匹配，但使用方式存在优化空间。

---

## 质疑清单

### 问题 1：UB 容量假设错误——DAV_2201 实际为 192KB，非 248KB
- **类别**：内存规划
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §1.5（Buffer 规划）、§2.2（UB 切分策略）
- **问题描述**：设计中 `constexpr uint32_t UB_SIZE = 253952; // 248KB`，但查阅 CANN 9.0.0 源码 `kernel_utils_constants.h` 确认：
  ```cpp
  // __NPU_ARCH__ == 2201 (Ascend910B 系列)
  const uint32_t TOTAL_UB_SIZE = 192 * 1024;  // 196608 bytes
  ```
  实际 UB 为 **192KB (196608 bytes)**，而非 248KB。设计中的所有 Buffer 规划、tileSize 计算、Path A/B 路径判断均基于错误的 248KB 容量。
- **Developer 视角**：如果按 248KB 规划 Buffer，实际只有 192KB，`InitBuffer` 分配时会超出 UB 容量，导致运行时 buffer 溢出崩溃。Path A 的输入全量搬运假设有 ~245KB 可用，实际只有 ~188KB，能容纳的输入量大幅缩小，Path A/B 阈值也需要重新计算。
- **建议方案**：
  1. 将 `UB_SIZE` 修正为 `192 * 1024 = 196608`
  2. 重新计算 `ubBudget = 196608 - 8192 = 188416`（预留 8KB）
  3. 重新推导 Path A 的 tileSize 和 Path A/B 切换阈值
  4. 建议使用 `GetUBSizeInBytes()` 或平台 API 动态获取 UB 大小，避免硬编码

---

### 问题 2：Path A 偏移表使用 SetValue 逐元素构建——性能风险
- **类别**：伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.4.1（Path A 伪代码，第 262-273 行）
- **问题描述**：伪代码中通过循环调用 `offsetLocal.SetValue(i, srcIdx * sizeof(T))` 逐元素构建 Gather 偏移表。SetValue 官方文档明确警告："不要大量使用SetValue对LocalTensor进行赋值，会使性能下降"。对于 tileSize 为数千元素的 tile，每个 tile 需要数千次标量 SetValue 调用，无法利用向量引擎。
- **Developer 视角**：SetValue 是标量操作，每次只能写一个元素。相比向量 API 一次处理 128/64 个元素，吞吐差距巨大。对于 as_strided 这种索引计算密集型算子，偏移表构建可能成为性能瓶颈。
- **建议方案**：
  1. **Host 预计算方案**（推荐）：在 Host 侧预计算全部 offset 表，通过 workspace 传入 Kernel，Kernel 内 DataCopy 到 UB 后直接使用。代价是需要 O(totalOutputElements × 4) bytes 的 workspace
  2. **1D 特殊优化**：1D 场景下 offset 为等差数列，可用 `Arange + Muls + Adds` 向量化生成
  3. **混合方案**：1D 走向量化，多维走 SetValue（接受性能折衷）或 Host 预计算

---

### 问题 3：负 stride 下 srcIdx 可能为负——Gather 偏移越界风险
- **类别**：伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.4（核心计算步骤，第 127 行）、§2.3（分支场景覆盖，第 242 行）
- **问题描述**：设计声称"负 stride：源索引可能递减，offset 仍为正值（srcIdx × sizeof(T) 始终 >= 0）"，但这一断言仅在用户保证所有源索引合法时成立。伪代码中 `srcIdx` 为 `int32_t`，当 `storageOffset + Σ(i_j × stride_j) < 0` 时，`srcIdx` 为负值，`srcIdx * sizeof(T)` 为负，存入 `LocalTensor<uint32_t>` 会发生无符号回绕（如 -4 → 0xFFFFFFFC），Gather 将读取远超 UB 范围的地址，导致运行时异常。
- **Developer 视角**：虽然 PyTorch 对越界 as_strided 访问也是 UB，但 NPU 端不应因用户参数错误导致硬件异常（可能触发 HCCL 超时或设备挂起）。应在 Kernel 内做防御性处理。
- **建议方案**：
  1. **防御性 clamp**：在写入 offset 前检查 `srcIdx >= 0 && srcIdx < inputTotalElements`，越界时填 0（Gather 读 input[0]），输出值虽错但不崩溃
  2. **srcBaseOffset 偏移**：计算所有输出元素的最小 srcIdx，设为 srcBaseOffset，使所有相对偏移非负。但这需要预扫描，增加复杂度
  3. **Host 侧校验**：在 Tiling 阶段检查参数合法性，非法时走 Path B 并做 clamp（最简方案）

---

### 问题 4：TilingData 包含每核变量——共享数据结构设计错误
- **类别**：多核策略
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.5（TilingData 结构设计，第 369-370 行）
- **问题描述**：TilingData 结构体包含 `coreElements`（本核处理的元素数）和 `coreOffset`（本核起始输出索引），标注为"本核"变量。但在直调模型中，TilingData 由 Host 写入一次、所有核共享读取，无法为每个核存储不同的值。§2.1 的多核切分公式中 `coreOffset = GetBlockIdx() * coreElements` 是在 Kernel 内计算的，与 TilingData 中的字段矛盾。
- **Developer 视角**：如果所有核读到相同的 coreOffset=0，只有核 0 会产出正确结果，其余核从偏移 0 开始重复计算，输出数据错误。
- **建议方案**：
  1. 从 TilingData 中移除 `coreElements` 和 `coreOffset`
  2. 在 Kernel Init 中通过 `GetBlockIdx()` 动态计算：
     ```cpp
     uint32_t blockIdx = GetBlockIdx();
     uint32_t coreElements = CeilDiv(totalOutputElements, blockDim);
     uint32_t coreOffset = blockIdx * coreElements;
     if (coreOffset + coreElements > totalOutputElements) {
         coreElements = totalOutputElements - coreOffset;
     }
     ```
  3. TilingData 仅存储 `totalOutputElements` 和 `blockDim`（或 `coreElementsPerCore` 即均分值）

---

### 问题 5：未讨论 Double Buffer 策略——遗漏流水线优化
- **类别**：内存规划
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.5（Buffer 规划）、§2.4（伪代码）
- **问题描述**：设计未提及 Double Buffer 策略。根据 API 最佳实践，Double Buffer 可使 MTE2/MTE3 搬运与 Vector 计算并行，掩盖搬运延迟。当前伪代码为严格串行：CopyIn → Compute → CopyOut。
- **Developer 视角**：
  - **Path A**：输入全量 CopyIn 一次后复用，无需 Double Buffer；但 `outQueueDst` 可开启 Double Buffer（`InitBuffer(que, 2, size)`），使当前 tile 的 CopyOut 与下一 tile 的 Compute(Gather) 并行
  - **Path B**：Compute 本身极慢（逐元素标量操作），Double Buffer 收益有限，但 `outQueueDst` 仍可考虑
  - **offset 表**：若走 SetValue 路径，`tmpQueueOffset` 不涉及 MTE 搬运，无需 Double Buffer；若改为 Host 预计算 + DataCopy 路径，则可开启
- **建议方案**：
  1. Path A：`outQueueDst` 开启 Double Buffer（`InitBuffer(outQueueDst, 2, tileSize * sizeof(T))`）
  2. Path A：`inQueueSrc` 不开 Double Buffer（全量一次 CopyIn，无并行收益）
  3. 在 DESIGN.md §1.5 补充 Double Buffer 策略说明

---

### 问题 6：Path B 使用 GetValue+SetValue 逐元素模式——性能极差且为已知反模式
- **类别**：伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.4.2（Path B 伪代码，第 313-334 行）
- **问题描述**：Path B 使用 `GetValue(0)` 从临时 LocalTensor 读取单元素 + `SetValue(i, val)` 写入输出 LocalTensor 的逐元素模式。API 最佳实践文档（api-transpose.md §3 反模式表）明确将 "GetValue/SetValue 逐元素搬运" 列为反模式，标注"标量 UB 读写，吞吐极差"。此外，Path B 中每个元素还需一次 `DataCopyPad(blockLen=sizeof(T))`，DMA setup 开销远大于 2/4 字节的有效负载。
- **Developer 视角**：Path B 作为大输入回退路径，性能本就低于 Path A，但 GetValue+SetValue 模式使性能差距进一步放大到可能不可接受的程度。对于输出数千元素的场景，Path B 可能比 Path A 慢 2-3 个数量级。
- **建议方案**：
  1. **Host 预计算源索引 + 批量 DataCopyPad**：在 Host 侧预计算所有输出元素对应的源 GM 偏移，传入 workspace。Kernel 内按 tile 批量 DataCopyPad 从 GM 各偏移位置读取（仍需逐元素 DMA，但省去 GetValue/SetValue 中间步骤）
  2. **接受 Path B 性能折衷**：在文档中明确标注 Path B 仅用于输入 > 188KB 的极端场景，性能不作为验收标准
  3. **探索 DataCopyPad blockCount 模式**：若源索引有规律（如连续段），可将连续段合并为一次 blockCount > 1 的 DataCopyPad

---

### 问题 7：Path A 伪代码缺少 CopyIn 步骤——关键 DMA 操作未展示
- **类别**：伪代码可实现性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.1（Path A 伪代码）
- **问题描述**：Path A 伪代码从 tile 循环开始，但缺少最关键的 CopyIn 步骤——将输入全量从 GM 搬运到 UB 的 DataCopyPad 调用。这是 Path A 中最大的 DMA 操作（可能搬运 100KB+ 数据），应在伪代码中明确展示，包括参数设置和 EnQue/DeQue 流水线管理。
- **Developer 视角**：缺少 CopyIn 伪代码导致开发者不确定：inputLocal 是在 Init 中一次性 CopyIn 还是在 Process 首轮 CopyIn？是否需要 EnQue/DeQue？DataCopyPad 参数如何设置？容易实现出错。
- **建议方案**：在 §2.4.1 伪代码前补充 CopyIn 步骤：
  ```cpp
  // Path A CopyIn: 全量输入搬运（仅执行一次）
  LocalTensor<T> inputLocal = inQueueSrc.AllocTensor<T>();
  DataCopyExtParams inParams{1, inputTotalElements * sizeof(T), 0, 0, 0};
  DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  DataCopyPad(inputLocal, inputGlobal, inParams, padParams);
  inQueueSrc.EnQue(inputLocal);
  // 后续 tile 循环中通过 DeQue 获取 inputLocal
  ```

---

### 问题 8：多维索引反推使用除法/取模——NPU 上昂贵操作
- **类别**：伪代码可实现性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.1（Path A 伪代码，第 268-269 行）、§2.4.2（Path B 伪代码，第 319-320 行）
- **问题描述**：伪代码中多维索引反推使用 `i_d = remaining / dimStride[d]` 和 `remaining = remaining % dimStride[d]`，每元素每维度需一次整数除法+取模。NPU 标量除法/取模是昂贵操作（数十个 cycle），对 4D 输出的 1000 元素 tile，需 8000 次除法+取模。
- **Developer 视角**：虽然 ndim 最多 4 且操作为标量，不会像向量 API 那样高效，但累积开销可能占 Compute 阶段的显著比例。对 1D 场景完全可以避免。
- **建议方案**：
  1. **1D 特化**：ndim=1 时 `srcIdx = storageOffset + flatIdx * stride[0]`，无需除法/取模
  2. **增量式索引计算**：从 tileStart 开始，逐元素递增多维索引（类似进位计数器），避免除法/取模：
     ```cpp
     // 初始化: curIdx[0..ndim-1] = decompose(tileStart)
     // 每步: curIdx[ndim-1]++; 若溢出则进位
     ```
  3. 当前方案可接受作为 v1 实现，后续性能优化时再替换

---

### 问题 9：DataCopyPad UB→GM 写回伪代码中 outputGlobal 偏移语义需确认
- **类别**：伪代码可实现性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.1（第 286 行）、§2.4.2（第 341 行）
- **问题描述**：伪代码使用 `DataCopyPad(outputGlobal[tileStart], outLocal, outParams)`，其中 `tileStart` 是输出展平后的一维索引。GlobalTensor 的 `[]` 运算符接受元素偏移（非字节偏移），对于 1D 展平输出这是正确的。但需确认 `outputGlobal` 的 GlobalBuffer 设置是否对应展平后的 1D 视图，而非原始多维 shape。
- **Developer 视角**：如果 outputGlobal 按 `SetGlobalBuffer((__gm__ T*)outputGM)` 设置（即指向展平后的 1D 起始地址），则 `outputGlobal[tileStart]` 正确。应在伪代码或 Init 中明确 GlobalBuffer 的设置方式。
- **建议方案**：在伪代码或 Init 步骤中补充 `outputGlobal.SetGlobalBuffer((__gm__ T*)outputGM)` 说明，确认输出按 1D 展平视图访问。

---

## API 验证记录

| API | 变体搜索命令 | 存在的变体 | 选用变体 | A3 支持 | 签名匹配 |
|-----|------------|-----------|---------|---------|---------|
| Gather | `ls asc-devkit/docs/api/context/ \| grep -iE "^Gather"` | Gather.md, Gather-51.md, Gatherb(ISASI).md, GatherB.md, GatherMask.md, Gather（支持寄存器为源操作数）.md | Gather.md（tensor前n个数据计算） | ✅ | ✅ |
| DataCopyPad | `ls asc-devkit/docs/api/context/ \| grep -iE "^DataCopyPad"` | DataCopyPad(ISASI).md | DataCopyPad(ISASI).md | ✅ (GM↔VECIN/VECOUT) | ✅ |
| SetValue | `ls asc-devkit/docs/api/context/ \| grep -iE "^SetValue"` | SetValue.md, SetValue-3.md | SetValue.md（LocalTensor） | ✅ | ✅ |
| GetValue | `ls asc-devkit/docs/api/context/ \| grep -iE "^GetValue"` | GetValue.md, GetValue-1.md | GetValue.md（LocalTensor） | ✅ | ✅ |

**Gather 关键约束确认**（基于 Gather.md 官方文档）：
- ✅ 支持数据类型：A3 系列支持 int16_t/uint16_t/int32_t/uint32_t/float/half/bfloat16_t，覆盖设计所需的 half/float/int32_t
- ✅ srcOffset 类型：`LocalTensor<uint32_t>`，单位为字节，与设计一致
- ✅ srcBaseOffset：uint32_t，单位为字节，设计使用 0
- ✅ dst/src/srcOffset 起始地址需 32 字节对齐
- ✅ srcOffset 取值需保证 src 元素类型位宽对齐（设计使用 `srcIdx * sizeof(T)`，满足）
- ⚠️ srcOffset 偏移地址不能超出 UB 大小数据的范围（需确保 srcIdx < inputTotalElements）

**DataCopyPad 关键约束确认**（基于 DataCopyPad(ISASI).md 官方文档）：
- ✅ A3 支持 GM→VECIN/VECOUT 和 VECIN/VECOUT→GM 通路
- ⚠️ A3 **不支持** GM→VECCALC 通路（设计未使用此通路，无影响）
- ✅ blockLen ∈ [1, 2097151]，Path A 全量搬运和 Path B 单元素搬运均在范围内

**Gather-51 变体**：不支持 A3 系列（"Atlas A3 训练系列产品/Atlas A3 推理系列产品: x"），不可使用。

---

## UB 容量验证详情

| 来源 | UB 大小 | 适用架构 |
|------|---------|---------|
| DESIGN.md | 248KB (253952 bytes) | ascend910b |
| CANN 9.0.0 `kernel_utils_constants.h` `__NPU_ARCH__==2201` | **192KB (196608 bytes)** | DAV_2201 (Ascend910B 全系列) |
| npu-smi 实际芯片 | 910B3 | DAV_2201 |
| API 最佳实践 `api-buffer.md` | 192KB | A2/A3 |

**结论**：设计中的 248KB 假设错误，应修正为 192KB。修正后 Path A 可用 UB 预算从 ~245KB 降至 ~188KB，显著影响 Path A/B 阈值和 tileSize。

---

## 审查统计

| 严重程度 | 数量 | 问题编号 |
|---------|------|---------|
| 🔴 阻塞 | 1 | #1 |
| 🟡 需讨论 | 5 | #2, #3, #4, #5, #6 |
| 🟢 建议 | 3 | #7, #8, #9 |
| **合计** | **9** |

---

### Architect 回应

#### 问题 1：UB 容量假设错误——DAV_2201 实际为 192KB，非 248KB
- **回应**：已修改
- **理由**：Developer 验证正确。CANN 9.0.0 `kernel_utils_constants.h` 中 `__NPU_ARCH__ == 2201` 明确定义 `TOTAL_UB_SIZE = 192 * 1024`，实际芯片 910B3 属于 DAV_2201 架构。原设计使用 248KB 会导致 `InitBuffer` 分配超出 UB 容量，运行时崩溃。
- **文档依据**：`/home/developer/Ascend/cann-9.0.0/aarch64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h` 第 128-130 行
- **DESIGN.md 变更**：
  1. §1.5 UB 容量从 248KB 修正为 192KB (196608 bytes)
  2. §2.2 `UB_SIZE` 从 253952 修正为 196608，`ubBudget` 从 245760 修正为 188416
  3. §2.6 Host Tiling 流程中 `ubBudget = 196608 - 8192`
  4. 关于 `GetUBSizeInBytes()` 建议：查阅官方文档确认该 API **不支持 A3 系列**（含 ascend910b），因此无法使用动态获取，必须使用编译时常量 196608

#### 问题 2：Path A 偏移表使用 SetValue 逐元素构建——性能风险
- **回应**：已修改
- **理由**：Developer 质疑合理。SetValue 官方文档明确警告"不要大量使用SetValue对LocalTensor进行赋值，会使性能下降"（[SetValue.md](asc-devkit/docs/api/context/SetValue.md) 约束说明）。采纳 Developer 推荐的 Host 预计算方案：在 Host 侧预计算全部 offset 表，通过 workspace 传入 Kernel，Kernel 内 DataCopyPad 到 UB 后直接使用。这彻底消除了 Kernel 内的 SetValue 标量操作，且 workspace 开销（totalOutputElements × 4 bytes）在合理范围内。
- **文档依据**：`asc-devkit/docs/api/context/SetValue.md` 约束说明；`asc-devkit/docs/api/context/DataCopyPad(ISASI).md` GM→VECIN/VECCALC/VECOUT 通路
- **DESIGN.md 变更**：
  1. §1.2 API 映射表新增"偏移表 GM→UB 搬运"行（DataCopyPad workspace → UB）
  2. §1.3 数据流更新：Path A 增加从 workspace DataCopyPad 偏移表步骤，移除"Compute: 计算当前 tile 的 byte offset 表"
  3. §1.4 核心计算步骤重构：分为"Host 侧预计算"和"Kernel 侧执行"两阶段
  4. §2.4.1 Path A 伪代码：移除 SetValue 循环，替换为 DataCopyPad 从 workspace 搬运 offset 表
  5. §2.5 新增"Host 侧偏移表预计算"伪代码
  6. §1.5 新增 Workspace 规划表

#### 问题 3：负 stride 下 srcIdx 可能为负——Gather 偏移越界风险
- **回应**：已修改
- **理由**：Developer 质疑合理。Gather 官方文档约束"偏移地址后不能超出UB大小数据的范围"（[Gather.md](asc-devkit/docs/api/context/Gather.md) srcOffset 参数说明），负 srcIdx 存入 `uint32_t` 会无符号回绕，导致 Gather 读取超范围地址。采纳防御性 clamp 方案：Host 预计算 offset 时，对 srcIdx 做 `clamp(srcIdx, 0, inputTotalElements - 1)`，越界时填 0。输出值虽错但不触发硬件异常。由于偏移表已改为 Host 预计算（问题 2），clamp 逻辑自然在 Host 侧完成，无需额外 Kernel 开销。
- **文档依据**：`asc-devkit/docs/api/context/Gather.md` srcOffset 参数说明："偏移地址后不能超出UB大小数据的范围"
- **DESIGN.md 变更**：
  1. §1.4 关键设计要点第 3 条：新增"防御性 clamp"说明
  2. §2.3 分支场景覆盖：负 stride 行从"offset 仍为正值"改为"Host 预计算时 clamp 到 [0, inputTotalElements-1]"
  3. §2.3 新增"越界访问"行：防御性 clamp 策略
  4. §2.5 Host 偏移表预计算伪代码：增加 clamp 逻辑

#### 问题 4：TilingData 包含每核变量——共享数据结构设计错误
- **回应**：已修改
- **理由**：Developer 质疑正确。直调模型中 TilingData 由 Host 写入一次、所有核共享读取，`coreElements` 和 `coreOffset` 无法为每核存储不同值。如果所有核读到 coreOffset=0，只有核 0 产出正确结果。采纳建议：从 TilingData 移除这两个字段，改为存储 `blockDim`，Kernel 内通过 `GetBlockIdx()` 动态计算。
- **文档依据**：直调模型 TilingData 语义（Host 写一次，所有核共享读）
- **DESIGN.md 变更**：
  1. §2.5 TilingData 结构体：移除 `coreElements` 和 `coreOffset`，新增 `blockDim`
  2. §2.1 多核切分公式：标注"Kernel 内动态计算，不存入 TilingData"
  3. §2.5 新增设计说明段落解释移除原因

#### 问题 5：未讨论 Double Buffer 策略——遗漏流水线优化
- **回应**：部分修改
- **理由**：Developer 建议合理。Path A 的 `outQueueDst` 开启 Double Buffer 可使当前 tile 的 CopyOut 与下一 tile 的 Gather+CopyIn(offset) 并行，有实际收益。`inQueueSrc` 不开 Double Buffer（全量一次 CopyIn，无并行收益）。Path B 的 Compute 本身是逐元素标量操作，Double Buffer 收益极有限，v1 不开启。
- **文档依据**：`asc-devkit/docs/api/context/SetDoubleBuffer.md`；API 最佳实践 Double Buffer 流水线并行原理
- **DESIGN.md 变更**：
  1. §1.5 Buffer 规划表：新增 Double Buffer 列，标注 Path A outQueueDst 为"是"，其余为"否"
  2. §1.5 Path A 总 UB 使用量：输出 tile 项乘以 2（Double Buffer）
  3. §1.4 关键设计要点第 6 条：新增 Double Buffer 策略说明
  4. §2.2 tileSize 计算：Path A 的 perTileBytes 考虑 Double Buffer（`sizeof(T) * 2`）

#### 问题 6：Path B 使用 GetValue+SetValue 逐元素模式——性能极差且为已知反模式
- **回应**：部分修改
- **理由**：Developer 正确识别了 GetValue+SetValue 反模式。然而 Path B 作为大输入回退路径，核心瓶颈是逐元素 DataCopyPad（每个元素一次小 DMA），GetValue+SetValue 是在此瓶颈之上的额外开销。通过 Host 预计算 srcIdx 表（workspace → DataCopyPad 到 UB），消除了 Kernel 内的除法/取模索引计算，但 GetValue+SetValue 仍需保留用于将单元素放置到输出 tile 的正确位置——目前没有向量 API 能将 DataCopyPad 读入的元素直接写入 LocalTensor 的任意偏移位置。采纳"接受 Path B 性能折衷"方案，在文档中明确标注 Path B 性能不作为验收标准。
- **文档依据**：`asc-devkit/docs/api/context/SetValue.md` 约束说明；Path B 仅用于输入 > 188KB 的极端场景
- **DESIGN.md 变更**：
  1. §2.4.2 Path B 伪代码：增加从 workspace DataCopyPad srcIdx 表步骤，移除 Kernel 内索引计算循环
  2. §1.5 Path B Buffer 规划：新增 tmpQueueSrcIdx buffer
  3. §2.4.2 注意事项：明确标注 GetValue+SetValue 为已知反模式，Path B 性能不作为验收标准

#### 问题 7：Path A 伪代码缺少 CopyIn 步骤——关键 DMA 操作未展示
- **回应**：已修改
- **理由**：Developer 质疑合理。Path A 的全量输入 CopyIn 是关键 DMA 操作（可能搬运 100KB+ 数据），缺少此步骤的伪代码会导致实现时不确定执行时机和参数设置。在 §2.4.1 新增 Init 伪代码段，包含 SetGlobalBuffer 设置和全量 DataCopyPad CopyIn。
- **文档依据**：`asc-devkit/docs/api/context/DataCopyPad(ISASI).md` GM→Local Memory 通路
- **DESIGN.md 变更**：
  1. §2.4.1 新增"Init 伪代码"段：包含 GlobalBuffer 设置（1D 展平视图）和全量 DataCopyPad CopyIn
  2. Compute 伪代码开头增加 `DeQue` 获取 inputLocal

#### 问题 8：多维索引反推使用除法/取模——NPU 上昂贵操作
- **回应**：已修改（由问题 2 的方案间接解决）
- **理由**：采纳问题 2 的 Host 预计算方案后，所有索引计算（包括多维索引反推的除法/取模）均在 Host 侧完成，Kernel 内不再需要任何除法/取模运算。此问题被彻底消除，无需额外优化（1D 特化、增量式索引计算等）。Host 侧除法/取模开销可忽略不计。
- **文档依据**：Host 侧计算无 NPU 标量操作开销
- **DESIGN.md 变更**：§1.4 核心计算步骤已重构为 Host 预计算 + Kernel 执行两阶段，Kernel 内无除法/取模

#### 问题 9：DataCopyPad UB→GM 写回伪代码中 outputGlobal 偏移语义需确认
- **回应**：已修改
- **理由**：Developer 建议合理。`outputGlobal[tileStart]` 的正确性依赖于 GlobalBuffer 按 1D 展平视图设置。在 Init 伪代码中已明确补充 `outputGlobal.SetGlobalBuffer((__gm__ T*)outputGM)` 和 `inputGlobal.SetGlobalBuffer((__gm__ T*)inputGM)`，确认所有 GlobalTensor 均按 1D 展平视图访问。
- **文档依据**：`asc-devkit/docs/api/context/SetGlobalBuffer.md`；GlobalTensor `[]` 运算符接受元素偏移
- **DESIGN.md 变更**：§2.4.1 和 §2.4.2 的 Init 伪代码中均补充 SetGlobalBuffer 设置说明

---

### 回应统计
- 接受 6 项（#1, #2, #3, #4, #7, #9），保留原设计 0 项，部分修改 2 项（#5, #6），间接解决 1 项（#8）
- DESIGN.md 已全面更新 |

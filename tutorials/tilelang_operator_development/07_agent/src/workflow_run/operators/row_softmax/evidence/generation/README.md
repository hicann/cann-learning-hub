# 生成阶段材料阅读说明

本目录保存生成阶段的原始实现和执行材料。原始日志、JSON 与编译器生成源码保持原样，便于核对；本说明负责解释每份文件应该看什么，以及它能够支持怎样的结论。

建议按下面的顺序阅读：

1. 从 `initial/row_softmax.py` 看设计怎样落实到 TileLang 代码。
2. 从 `accuracy.log` 确认 pytest 是否真实执行。
3. 从 `result.json` 查看每个案例的数值指标。
4. 对后端生成过程感兴趣时，再查看 `generated_source/m64_n512.cpp` 和 `source_check.json`。

## 1. `initial/row_softmax.py`：初始算子实现

这份文件分为三部分：

- `_validate_shape`：检查 M、N 是否在支持范围内，并满足 4 行和 64 列的倍数要求。
- `build`：定义并编译 TileLang kernel。
- `run`：检查公开输入，创建新的输出 Tensor，再调用编译后的 kernel。

阅读 `build` 时，可以沿一行数据寻找设计与代码的对应关系：

- `T.Kernel(rows)` 创建 M 个逻辑任务，`bx` 表示当前任务编号。
- `T.copy(X[bx : bx + 1, :], x)` 将第 `bx` 行搬入 UB。
- 四个 `T.SimdVF` 依次完成最大值、指数、指数和与归一化。
- `T.copy(y, Y[bx : bx + 1, :])` 将结果写回对应输出行。

阅读 `run` 时，重点看输入约束是否落实，以及 `torch.empty(...)` 是否为输出创建了新的存储空间。

## 2. `accuracy.log`：pytest 原始执行日志

这是终端运行 pytest 时保存的原始文本。可以依次寻找：

- `collected 9 items`：pytest 实际收集到 9 个测试节点。
- `test_specified[...]`：方括号中是本次运行的 case ID。
- `TileLang begins/completes to compile`：该 shape 触发并完成了 kernel 编译。
- 每个案例后的 JSON 行：记录本次输出的误差指标。
- `PASSED` 和末尾的 `9 passed`：节点结果与整次运行汇总。

日志中的 `$COURSE_RUNTIME`、`$WORKSPACE` 是课程归档时替换的环境路径，不影响测试节点和结果。

## 3. `result.json`：机器可读的测试结果

这份 JSON 将日志中的关键信息整理成固定字段，便于 Agent 或脚本继续读取：

- `stage`、`status`：材料所属阶段和总体结论。
- `passed`、`failed`、`skipped`：实际测试数量。
- `case_id`、`seed`：案例身份及随机输入种子。
- `max_abs_error`：实际输出与参考结果之间最大的绝对差值。
- `max_normalized_error`：绝对误差除以该元素允许误差后的最大值；小于等于 1 才满足逐元素容差。
- `max_row_sum_error`：每行输出之和与 1 的最大差值。
- `files`：本轮算子和 pytest 文件的 SHA256，用来标识结果对应的文件版本。

结构化结果方便汇总，但不能单独证明测试真的执行过，因此还要与原始日志对应。

## 4. `generated_source/m64_n512.cpp`：后端生成代码样例

这是 `(M,N)=(64,512)` 对应的 AscendC 生成源码，不是人工编写的主实现。可以搜索以下标记：

- `__global__ __vector__`：设备侧 Vector kernel 入口。
- `block_idx * 512`：当前逻辑任务对应的行起点。
- `asc_copy_gm2ub_align`：输入从 GM 搬到 UB。
- 四个 `__simd_vf__` 函数：最大值、指数、指数和与归一化对应的向量计算区域。
- `asc_sync_notify`、`asc_sync_wait`：数据搬运与 Vector 计算之间的同步。
- `asc_copy_ub2gm_align`：结果从 UB 写回 GM。

该文件用于观察 TileLang 最终生成了什么，不应手工修改。修改它不会改变下一次由 TileLang 重新生成的代码。

## 5. `source_check.json`：九项源码的结构核对

这份 JSON 对九个指定案例分别记录：

- `source_sha256`：生成源码的内容哈希。
- `simdvf_definitions`：生成的 SimdVF 函数数量，本例预期为 4。
- `vector_entry`：是否存在 Vector kernel 入口。
- `gm_to_ub`、`ub_to_gm`：是否存在输入搬入与结果写回。
- `mte2_v_notify_wait`、`v_mte3_notify_wait`：是否存在对应流水之间的同步。
- `design_ub_bytes`：按设计计算的显式 UB 用量，公式为 `12 × N + 64` 字节。

相同 N 的案例可能具有相同源码哈希，因为 M 决定启动多少个逻辑任务，而每个任务内部执行的行 Softmax 代码没有变化。

## 这些材料不能单独证明什么

- 只看 `accuracy.log`，无法了解代码最终生成了怎样的后端结构。
- 只看生成 C++ 或 `source_check.json`，无法证明数值结果正确。
- 生成阶段的 9 个案例通过，只能建立可运行的初始版本，不能证明支持范围已经得到完整覆盖。

因此，课堂正文只用它们支持“初始实现已经实际运行”的判断；测试是否充分、断言是否可信，继续由 ST 阶段分析。

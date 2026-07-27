# 4.4 课后练习答案

## 选择题

**1. B** — Cube Only 模式通过 `@asc.jit(matmul_cube_only=True)` 装饰器启用。该参数告诉编译器核函数仅使用 Cube 计算单元，不涉及 Vector 计算。

**2. B** — Cube Only 模式下，kernel launch 的核数为 `tiling.used_core_num`（不除以 2）。因为纯 Cube 模式只使用 AIC 核，每个 AIC 核独立处理一个数据分片。

**3. B** — MIX 模式下 kernel launch 的核数为 `USE_CORE_NUM // 2`。因为每个 AI Core 包含 1 个 AIC 和 1 个 AIV，MIX 模式需要 AIC+AIV 成对协同工作，所以实际启动的核组数为总核数的一半。

**4. B** — `m_index = block_idx % m_single_blocks`，其中 `m_single_blocks = tiling.m.ceildiv(tiling.single_core_m)`。M 方向的分块数决定了每个核在 M 轴上的位置索引。

## 填空题

**5.** `asc.ascend_is_aic()` — 在 Cube Only 模式下，核函数内的计算逻辑需要用 `if asc.ascend_is_aic():` 包裹，确保仅在 AIC 核上执行 Cube 计算代码，AIV 核跳过。

**6.** 在 Tiling 配置中调用 `matmul_tiling.enable_bias(True)` 启用 Bias，在 kernel 中调用 `matmul.set_bias(bias_global)` 设置偏置张量。Cube Only 模式下 set_tail 需传入 3 个参数 `set_tail(tail_m, tail_n, tiling.k_a)`（同一API，MIX模式传入2个参数，tail_k使用默认值-1）。

**7.** 尾块处理的目的是处理矩阵维度不能被 `single_core_m` 或 `single_core_n` 整除的情况。当 `tail_m >= tiling.single_core_m` 时设为 `tiling.single_core_m`（满块），否则保留实际剩余值。这通过 `matmul.set_tail(tail_m, tail_n)` 传给高阶 API。

**8.** 当 `is_trans_a=True` 时，A 矩阵的偏移计算从 `m_index * tiling.k_a * tiling.single_core_m` 变为 `m_index * tiling.single_core_m`，因为转置后 M 轴在内存中的连续方向发生了变化。

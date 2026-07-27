# 4.2 课后练习答案

## 选择题

**1. B** — Cube 计算的数据流路径为 GM → L1(A1/B1) → L0A/L0B(A2/B2) → mmad → L0C(CO1) → fixpipe → GM。数据从 GM 经 L1 中转后到达 L0A/L0B，mmad 从 L0A/L0B 读取数据计算并输出到 L0C(CO1)，最后由 fixpipe 搬回 GM。

**2. B** — 在 fp16 数据类型下，Cube 矩阵的分形大小为 16×16。A2 和 B2 上的分形矩阵均为 16×16 个元素。

## 填空题

**3.** `asc.TPosition.A2` — asc.mmad 的左矩阵（fm 参数）必须位于 A2 存储位置，这是 Cube 硬件单元对输入数据的布局要求。

**4.** GM到L1的数据搬运使用 `asc.data_copy` 指令并传入 `asc.Nd2NzParams` 参数，该过程将GM中的 **ND** 格式数据转换为L1上的 **NZ** 格式。Nd2NzParams 通过 nd_num、n_value、d_value、src_nd_matrix_stride、src_d_value、dst_nz_c0_stride、dst_nz_n_stride、dst_nz_matrix_stride 共8个参数描述ND到NZ的格式转换。

**5.** data_copy（MTE2流水线）完成后、load_data（MTE1流水线）开始前，需要插入 `asc.set_flag` 和 `asc.wait_flag` 同步指令，对应的HardEvent为 `asc.HardEvent.MTE2_MTE1`。set_flag/wait_flag 必须成对出现，event_id 通过 `pipe.fetch_event_id()` 获取。

**6.** `asc.mmad` 的 `MmadParams` 中，`m` 对应左矩阵A的行数（M维度），`n` 对应右矩阵B的列数（N维度），`k` 对应左矩阵A的列数/右矩阵B的行数（K维度，即缩减维度）。当K维度较大需要分块累加时，`cmatrix_init_val` 在首次迭代应设为 `True`（L0C清零，执行 `C = A_0 × B_0`），后续迭代应设为 `False`（L0C保留上次结果，执行 `C += A_i × B_i`）。

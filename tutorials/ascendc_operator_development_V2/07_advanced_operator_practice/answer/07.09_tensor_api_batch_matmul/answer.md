# 07.09 Tensor API BatchMatmul 参考答案

## 课后实践参考答案

以下给出 `batch_matmul_practice_kernel.h` 中各 TODO 的参考填法（与官方 `batch_matmul_tensor_api` 样例的推荐写法一致）；完整可运行参考实现对应课程工程中的 `batch_matmul_answer_kernel.h`。

### TODO(1)：batch 维多核切分索引

```cpp
uint32_t single_core_b = B / block_num;
uint32_t single_core_res_b = B % block_num;
uint32_t actual_single_core_b = single_core_b + (single_core_res_b > block_index ? 1 : 0);
uint32_t batch_index_start = single_core_b * block_index + min(block_index, single_core_res_b);
uint32_t batch_index_end = actual_single_core_b + batch_index_start;
```

B 个 batch 均分给所有核心，余数 `B % 核数` 逐个分给编号靠前的核（前 `single_core_res_b` 个核多算一个）。编号相邻的核心访问相邻 GM 地址，对 L2Cache 友好。本节规格 B=128、32 核，恰好每核 4 个 batch，余数为 0。

### TODO(2)：GM Tensor 的 3D 布局

```cpp
auto gm_a = make_tensor(make_mem_ptr(a), make_frame_layout<nd_ext_layout_ptn>(B, M, K));
auto gm_b = make_tensor(make_mem_ptr(b), make_frame_layout<nd_ext_layout_ptn>(B, K, N));
auto gm_c = make_tensor(make_mem_ptr(c), make_frame_layout<nd_ext_layout_ptn>(B, M, N));
auto gm_bias = make_tensor(make_mem_ptr(bias), make_frame_layout<nd_ext_layout_ptn>(B, 1, N));
```

batch 维作为 Layout 的第一维，由 Layout 统一管理行列寻址——相比手工做 `x + b_core_idx * M * K` 基址偏移，3D 布局的写法让后续 slice/copy 与单矩阵完全一致，这是官方样例的推荐写法。A/B 均为 ND 存储、不做转置。

### TODO(3)：L1/L0 缓冲 Tensor（脚手架，无需修改）

L1A/L0A 用 `nz_layout_ptn`、L1B 用 `nz_layout_ptn`、L0B 用 `zn_layout_ptn`（Cube 核各级存储的分形格式要求，由 `copy_gm_to_l1`/`copy_l1_to_l0a`/`copy_l1_to_l0b` 搬运时自动完成格式转换）；L0C 为 float 类型（FP32 累加）；Bias 在 L1 与 BiasTable（`__biasbuf__`）均为 `nd_ext_layout_ptn`。各级张量第一维均为 batch 维。

### TODO(4)：GM→L1 批量搬运

```cpp
copy(
    copy_gm_to_l1_atom, l1_a_tensor,
    gm_a.slice(make_coord(l1_batch_index, make_coord(0, 0)), make_shape(l1_batch_size, make_shape(M, K))));
copy(
    copy_gm_to_l1_atom, l1_b_tensor,
    gm_b.slice(make_coord(l1_batch_index, make_coord(0, 0)), make_shape(l1_batch_size, make_shape(K, N))));
copy(
    copy_gm_to_l1_atom, l1_bias_tensor,
    gm_bias.slice(make_coord(l1_batch_index, make_coord(0, 0)), make_shape(l1_batch_size, make_shape(1, N))));
```

3D slice 的 `coord` 第一维是本批 batch 的起始索引（内层 `make_coord(0, 0)` 为矩阵内坐标），`shape` 第一维是本批 batch 数（内层为单个矩阵的行列）。一次 copy 搬入整批 `L1_BATCH_SIZE` 个 batch。

### TODO(5)：L1→L0A/L0B/BiasTable 搬运

```cpp
copy(
    copy_l1_to_l0a_atom, l0_a_tensor,
    l1_a_tensor.slice(make_coord(l0_batch_index, make_coord(0, 0)), make_shape(l0_batch_size, make_shape(M, K))));
copy(
    copy_l1_to_l0b_atom, l0_b_tensor,
    l1_b_tensor.slice(make_coord(l0_batch_index, make_coord(0, 0)), make_shape(l0_batch_size, make_shape(K, N))));
copy(
    copy_l1_to_bt_atom, l0_bias_tensor,
    l1_bias_tensor.slice(make_coord(l0_batch_index, make_coord(0, 0)), make_shape(l0_batch_size, make_shape(1, N))));
```

从 L1 批量张量中按 `l0_batch_index` 切出 L0 批次。Bias 走专用的 `copy_l1_to_biastable` 通路，搬运到 `__biasbuf__` 的 BiasTable 中供 mmad 使用。

### TODO(6)：逐 batch 的 5 参数 mmad

```cpp
for (uint32_t l0c_batch_index = 0; l0c_batch_index < l0_batch_size; l0c_batch_index++) {
    mmad(
        mmad_atom.with(mmad_params{
            static_cast<uint16_t>(M), static_cast<uint16_t>(N), static_cast<uint16_t>(K),
            unit_flag_mode::disable, true}),
        l0_c_tensor.slice(make_coord(l0c_batch_index, make_coord(0, 0)), make_shape(1, make_shape(M, N))),
        l0_a_tensor.slice(make_coord(l0c_batch_index, make_coord(0, 0)), make_shape(1, make_shape(M, K))),
        l0_b_tensor.slice(make_coord(l0c_batch_index, make_coord(0, 0)), make_shape(1, make_shape(K, N))),
        l0_bias_tensor.slice(make_coord(l0c_batch_index, make_coord(0, 0)), make_shape(1, make_shape(1, N))));
}
```

三个要点：一是硬件 mmad 指令不支持 batch 维运算，必须逐 batch 循环；二是第 5 个参数传入 Bias 张量，硬件在矩阵乘的同时随路累加 Bias（C = A × B + Bias）；三是 `init_with_zero = true`——每个 batch 的输出独立计算（单次 mmad 内部完成全部 K 累加），不存在 04.03 中跨 K block 的累加语义。

### TODO(7)：L0C→GM 批量搬出

```cpp
copy(
    copy_l0c_to_gm_atom,
    gm_c.slice(
        make_coord(l1_batch_index + l0_batch_index, make_coord(0, 0)),
        make_shape(l0_batch_size, make_shape(M, N))),
    l0_c_tensor);
```

GM 目标 batch 起点 = L1 批起点 + L0 批内偏移。一次 copy 搬出整批 `L0_BATCH_SIZE` 个结果矩阵，`copy_l0c_to_gm`（Fixpipe）在搬出过程中同时完成 FP32 → half 的类型转换。

### 运行参考实现

```bash
bash run.sh --case=answer
```

当 `verify_result.py` 输出 `test pass!` 时，说明参考实现正确。

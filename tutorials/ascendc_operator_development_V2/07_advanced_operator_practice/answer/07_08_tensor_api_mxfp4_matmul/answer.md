# 07.08 Tensor API MxFP4 高性能矩阵算子参考答案

## 课后练习

### 练习 1：MxFP4 Scale 共享与对齐

MxFP4 中 K 方向每 **32** 个元素共享一个 scale，scale_k 需对齐到 **2** Byte。

scale_k 的计算公式为 `align_even(ceil(K/32))`：先对 K 除以 32 向上取整，再对齐到 2 的倍数。对齐到 2 Byte 是因为硬件约束要求 scale 数据在 K 方向满足 2Byte 连续对齐。

### 练习 2：Mutex slot-lock 同步机制

Mutex slot-lock 中，**生产者**流水线调用 `asc_lock` 获取缓冲区写权限、`asc_unlock` 释放缓冲区并通知消费者；**消费者**流水线调用 `asc_lock` 获取缓冲区读权限、`asc_unlock` 释放缓冲区并通知生产者可覆盖。具体对应关系：

- L1 缓冲（slot 0/1, 2/3）：MTE2 是生产者（GM→L1 copy），MTE1 是消费者（L1→L0 copy）
- L0 缓冲（slot 4/5）：MTE1 是生产者（L1→L0 copy），M 是消费者（Mmad）
- L0C 缓冲（slot 6）：M 是生产者（Mmad 写入），FIX 是消费者（L0C→GM copy）

本例需要 7 个 slot：L1 Data Ping/Pong（slot 0/1）、L1 Scale Ping/Pong（slot 2/3）、L0 Ping/Pong（slot 4/5）、L0C（slot 6）。4 路输入 × Ping/Pong = 8 个 L1 缓冲区，加上 L0 的 A/B 各 Ping/Pong 共 4 个缓冲区（共享 slot 4/5）和 1 个 L0C，共 13 块缓冲区；但 A/B 生命周期一致绑定到 slot 0/1，ScaleA/ScaleB 生命周期一致绑定到 slot 2/3，L0 的 A/B 共享 slot 4/5，因此实际只需 7 个 slot。Mutex 的 slot-lock 模型适合表达这种差异化的缓冲生命周期（Data 和 Scale 的 chunk 大小不同），且无需预置即可使用。

### 练习 3：scale_factor_k 的作用

`scale_factor_k=4` 表示 ScaleA/ScaleB 在 K 方向的搬运范围是 A/B 的 4 倍。即 `scale_chunk_step = step_k × scale_factor_k = 2 × 4 = 8`，一次搬运覆盖 8 个 base_k 块对应的 scale 范围。

Scale 数据需要比 A/B 更大的搬运包，因为：
- Scale 数据量很小（K/32，每 32 个矩阵元素共享 1 个 scale），单次搬运的绝对数据量不大
- 但 Scale 搬运频率高（跟随每个 K block），如果不加大包粒度，MTE2 会被大量小包 Scale 搬运指令占据
- 用更大的 scale_factor_k 可以减少 Scale 的搬运指令数，降低 MTE2 压力

### 练习 4：init_with_zero

`init_with_zero=true` 在每个输出 tile 的 **第一个 K block**（`k_block_idx == 0`）时设置。此时 L0C 中的数据是上一个 tile 的残留结果，需要初始化为当前 tile 的首个 K block 计算结果。后续 K block 设为 `false`，在已有 L0C 数据上持续累加。

### 练习 5：L0ScaleA 与 L0A 的物理内存关系

L0ScaleA 和 L0A **共享同一块物理内存**。L0A 缓冲区存储 fp4x2_e1m2_t 类型的矩阵数据，而 L0ScaleA 存储对应的 fp8_e8m0_t 缩放因子。硬件设计中，L0A 的高位地址空间被复用为 L0ScaleA。

在代码中，通过 `make_mem_ptr<asc::te::location::l0scalea, fp8_e8m0_t>(reinterpret_cast<uint64_t>(l0_buf_a_ping) / 16)` 将 L0A 的物理地址转换为 L0ScaleA 的指针。除以 16 是因为 L0A 和 L0ScaleA 的地址映射粒度不同。

## 课后实践：随路量化参考答案

以下给出 `mmad_mx_quant_practice_kernel.h` 中 TODO 的参考填法；完整可运行参考实现对应课程工程中的 `mmad_mx_quant_answer_kernel.h`。

### TODO(1)：函数签名扩展

在 `process` 函数中添加 quant_scale 和 quant_offset 参数：

```cpp
__aicore__ inline void process(
    __gm__ fp4x2_e1m2_t* a, __gm__ fp4x2_e1m2_t* b,
    __gm__ fp8_e8m0_t* as, __gm__ fp8_e8m0_t* bs,
    __gm__ int8_t* c,
    __gm__ uint64_t* quant_scale, __gm__ uint64_t* quant_offset)
```

### TODO(2)：L1 缓冲区分配

```cpp
__cbuf__ uint64_t l1_buf_quant_scale[Trait::base_n];
__cbuf__ uint64_t l1_buf_quant_offset[Trait::base_n];

auto l1_quant_scale_tensor = make_tensor(make_mem_ptr(l1_buf_quant_scale),
    make_frame_layout<nd_layout_ptn, uint64_t>(1, Trait::base_n));
auto l1_quant_offset_tensor = make_tensor(make_mem_ptr(l1_buf_quant_offset),
    make_frame_layout<nd_layout_ptn, uint64_t>(1, Trait::base_n));
```

### TODO(3)：GM→L1 搬运量化参数

在 N/M 循环中搬运量化参数到 L1：

```cpp
auto gm_quant_scale = make_tensor(make_mem_ptr(quant_scale),
    make_frame_layout<nd_layout_ptn>(1, Trait::N));

copy(gm_to_l1_atom, l1_quant_scale_tensor,
     gm_quant_scale.slice(make_coord(0, n_block_idx * Trait::base_n), make_shape(1, cur_n)));
```

### TODO(4)：L0C→GM 量化 copy

将 3 参数 copy 改为 4 参数 copy，传入量化参数张量：

```cpp
l0c_to_gm_params fixpipe_params;
copy(l0c_to_gm_atom.with(fixpipe_params),
     c.slice(make_coord(m_block_idx * Trait::base_m, n_block_idx * Trait::base_n),
             make_shape(cur_m, cur_n)),
     l0_tensor_c, l1_quant_scale_tensor);
```

使用四参数 copy，将 L0C 中的 float 累加结果经过 Fixpipe 量化转换为 INT8 后写回 GM。

### 运行参考实现

```bash
bash run.sh --case=quant_answer
```

当 `verify_result_quant.py` 输出 `test pass!` 时，说明随路量化实现正确。

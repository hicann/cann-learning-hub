# TileLang与xLLM驱动Qwen3.5昇腾算子适配

## 背景介绍

Qwen3.5 是通义千问系列大语言模型，其架构引入了 Gated Delta Network（GDN）等创新模块，在提升模型表达能力的同时，也带来了 `causal_conv1d`、`fused_sigmoid_gating_delta_rule`、`chunk_gated_delta_rule_fwd_h` 等一批计算密集、逻辑复杂的定制算子。这些算子在推理过程中被高频调用，其执行效率直接决定了模型的整体推理性能。

xLLM 框架此前已接入这些算子，但其在昇腾 NPU 上的运行性能未达预期。与此同时，xLLM 已具备 TileLang 算子接入能力——开发者用 Python 编写算子 Kernel，再由编译器将其编译为针对昇腾 NPU 优化的 C++ 代码。为此，我们基于 TileLang-Ascend 路线对 Qwen3.5 的关键算子进行了重写，在保持开发效率的同时，显著提升算子在昇腾 NPU 上的执行性能。

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        xLLM 推理引擎                             │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    C++ Wrapper 层                         │  │
│  │   Tensor 校验 → 构建 Specialization → 查找 Kernel → 调用     │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │ 运行时调度                            │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │              TileLang 编译产物 (.so)                       │  │
│  │   causal_conv1d │ fused_sigmoid_gating │ chunk_gated_delta│  │
│  └───────────────────────▲───────────────────────────────────┘  │
│                          │ AOT 编译                              │
│  ┌───────────────────────┴───────────────────────────────────┐  │
│  │              TileLang Python Kernel                       │  │
│  │   @T.prim_func → generate_source() → Ascend C 源码         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

```

## xLLM三步接入 TileLang 算子

xLLM 接入一个新 TileLang 算子只需三步：写 Python Kernel → 编译 → 注册。

```
  Step 1                Step 2                Step 3
┌──────────┐    ┌────────────────┐    ┌───────────────┐
│ Python   │    │ AOT 编译       │    │ C++ Wrapper    │
│ Kernel   │───▶│ Ascend C 源码  │───▶│ + CMake 注册   │
│ 编写      │    │ → .so 动态库   │    │ → 集成验证      │
└──────────┘    └────────────────┘    └───────────────┘

```

### Step 1：Python Kernel 编写

以 `chunk_gated_delta_rule_fwd_h` 为例，使用 TileLang DSL 定义算子计算逻辑 （以下为简化后的概念性伪代码，主要展示 TileLang 的表达逻辑）：

```
@T.prim_func
def chunk_gated_delta_rule_fwd_h_kernel(
    h: T.Tensor, k: T.Tensor, v: T.Tensor,
    w: T.Tensor, g: T.Tensor, h0: T.Tensor, ht: T.Tensor, ...
):
    with T.Kernel(total_tasks, is_npu=True) as (cid, vid):
        # 分配片上存储：UB（向量计算）、L1/L0C（矩阵计算）
        h_state_ub = T.alloc_ub([2, K // 2, V_half], "bfloat16")
        h_state_float = T.alloc_ub([2, K // 2, V_half], "float32")
        k_chunk_l1 = T.alloc_L1([2, bt, K], "bfloat16")
        wh_frag = T.alloc_L0C([2, bt, V_half], "float32")

        for pair_idx in T.serial(num_pairs):
            # 加载初始状态（float32 精度）
            T.copy(h0[i_n, i_h, ...], h_state_float)

            for i in T.serial(NT_i):
                # Cube 核心：矩阵乘法 w @ h、k @ v_new
                with T.Scope("C"):
                    T.gemm_v0(w_chunk_l1, h_state_l1, wh_frag)
                    T.gemm_v0(k_chunk_l1, v_new_l1, hupd_frag, transpose_A=True)

                # Vector 核心：逐元素计算 delta rule 状态更新
                with T.Scope("V"):
                    # v_new = v - w @ h
                    T.tile.sub(v_chunk_float, v_chunk_float, wh_float)
                    # h *= exp(g_last)，v_new *= exp(g_last - g)
                    T.tile.mul(h_state_float, h_state_float, g_last_scalar)
                    T.tile.mul(v_chunk_float, v_chunk_float, g_exp_broc)
                    # h += k @ v_new
                    T.tile.add(h_state_float, h_state_float, hupd_float)

            # 写回最终状态（float32 精度）
            T.copy(h_state_float, ht[i_n, i_h, ...])

```

核心要点：

- Cube/Vector 双核协作：`T.Scope("C")` 调度矩阵乘法（`T.gemm_v0`），`T.Scope("V")` 调度逐元素向量计算（`T.tile.sub/mul/add/exp`）

- Double Buffer 流水线：得益于底层自动映射双缓冲，实现数据搬运与计算的高效重叠，屏蔽底层信号量同步细节。

- 精度控制：SSM 状态 `h_state_float` 全程以 `float32` 存储和更新，确保递推计算的数值稳定性

### Step 2：AOT 编译

TileLang 编译器负责将 Python 表达的 Tiling 逻辑和计算流水线，自动映射为 Ascend 的标准指令，并自动计算和填入对应硬件的 Tiling 参数，从而避免了手动计算 L1/UB 偏移量的繁琐工作。通过 `generate_source(...)` 将 Python Kernel 自动降级为 Ascend C 源码，再由毕昇编译器编译为动态库：

```
@register_kernel
class ChunkGatedDeltaRuleFwdHKernel(TilelangKernel):
    SPECIALIZATIONS = [
        {"variant_key": "H32_Hg8_D256_bf16",
         "num_heads": 32, "num_groups": 8, "head_dim": 256, "dtype": "bf16"},
    ]

    @staticmethod
    def generate_source(num_heads, num_groups, head_dim, dtype) -> str:
        kernel = build_chunk_gated_delta_rule_fwd_h_kernel(...)
        kernel = tilelang.engine.lower(kernel)
        return kernel.kernel_source

```

编译命令一行搞定：

```
python xllm/compiler/tilelang_launcher.py compile-kernels \
    --target ascend --device a3 \
    --kernels chunk_gated_delta_rule_fwd_h

```

### Step 3：C++ Wrapper 注册

编写 Wrapper 桥接 xLLM 运行时与编译产物，并在 CMake 中一行注册：

```
const auto* entry = find_chunk_gated_delta_rule_fwd_h_kernel_entry(specialization);
entry->fn(q_ptr, k_ptr, v_ptr, g_ptr, h0_ptr, h_out_ptr, stream);

```

```
tilelang_register_runtime_kernel(
    NAME chunk_gated_delta_rule_fwd_h
    WRAPPER_SRCS chunk_gated_delta_rule_fwd_h_wrapper.cpp
)

```

## Qwen3.5 关键算子适配

Qwen3.5 的 GDN 架构逻辑复杂且访存密集，若采用传统 Ascend C 原生开发，开发者需要耗费大量精力处理底层的 UB/L1 内存管理、Cube/Vector 硬件同步及寄存器分配。通过引入 TileLang，我们将底层硬件细节交由编译器处理，使开发者能专注于算法逻辑，大幅缩短了核心算子的开发与验证周期。

以下是三个核心算子的适配细节与收益：

### causal_conv1d：因果一维卷积

该算子负责对输入序列执行因果卷积（仅依赖当前及历史 token），并维护卷积状态（conv_state）以支持增量 Decode。在长序列场景下，访存与计算的重叠是性能优化的关键。

- Token-Block 自动流水线：为实现计算与访存的高效重叠，我们需要将序列按核数切分为连续的 token 块，并在块内串行计算 4 个 token。得益于 TileLang 的双缓冲（Double Buffer）原语，开发者只需使用简单的 `T.Pipelined`，编译器即可自动生成 Ascend 架构下的软硬件同步指令（如 DataCopy 异步搬运），免去了手动编写繁琐 Pipeline 同步信号的痛苦。

- 灵活的状态管理：通过 `cache_indices` 指定 batch 对应的 cache line，实现 prefix 缓存复用。TileLang 的 Python DSL 允许我们通过标准的控制流（`if initial_state_mode == ...`）快速定义状态加载逻辑，底层自动映射为高效的向量化读写指令。

- 开发周期：借助自动流水线生成，原先在 Ascend C 中需要反复调优内存 offset 和流水线节拍的逻辑，仅用 3 周即完成了从 Python Kernel 编写、精度对齐到性能调优的全流程。

### chunk_gated_delta_rule_fwd_h：分块门控 Delta Rule 前向

该算子实现 Chunk 级别的分块递推计算，是 Prefill 阶段的核心算子，难点在于矩阵乘（Cube）与状态递推（Vector）的异构协同，以及极高的精度要求。

- Cube/Vector 双核无缝协作：TileLang 提供了极简的 `T.Scope("C")` 和 `T.Scope("V")` 抽象。我们将 Chunk 内的矩阵乘（`w @ h`, `k @ v_new`）映射到 Cube，将 `h *= exp(g_last)` 等逐元素计算映射到 Vector。TileLang 编译器会自动完成两者之间的内存排布转换与执行调度，彻底屏蔽了 Ascend 底层矩阵格式（如 Fractal_NZ）转换的复杂性。

- 零负担的精度控制：状态矩阵 `h` 需全程以 `float32` 更新以避免精度坍塌。在 TileLang 表达中，开发者只需在分配 UB 内存时显式指定 `T.alloc_ub(..., "float32")`，编译器就会自动插入必要的 `Cast` 指令，实现 bfloat16 计算与 float32 状态存储的无缝衔接。

- 开发周期：由于 TileLang 自动计算并填入了复杂的 Tiling 参数（避免了手动计算 L1/L0C 内存上限的灾难），这一逻辑极为复杂的递推算子，核心开发仅耗时 2 周，便实现了 1.72x 的性能提升。

### fused_sigmoid_gating_delta_rule：融合 Sigmoid 门控 Delta Rule

该算子主要在 Decode 阶段逐 token 执行，将 sigmoid 激活、门控计算和 Delta Rule 状态更新融合为单一 Kernel，是 GDN 层的绝对算力瓶颈。

- AOT 解决多算子融合的寄存器分配难题：在处理此类复杂融合算子时，将 `A_log`（衰减率）、`query`、`key`、`value`、`beta` 等数十个输入输出融合在一个 Kernel 内，面临极易溢出的寄存器分配（Register Spilling）与内存对齐问题。TileLang 的编译器基于强大的底层 IR 分析能力，对 Python 编写的一系列 `T.tile.exp/add/mul` 数学表达式进行了自动的算子融合与生命周期管理，确保中间结果直接在寄存器或 UB 中流转，无需写回全局内存，极致降低了访存开销。

- 变长序列自动处理：通过传入 `cu_seqlens` 等动态 shape 信息，TileLang Kernel 在运行时自动处理 batch 内不同长度序列的独立状态指针偏移，开发者在 Python 侧仍可按照规则的稠密张量思维进行逻辑表达。

- 开发周期：原本需底层专家耗时数月进行汇编级优化的超级融合大算子，借助 DSL 的数学表达优势和编译器的自动优化，仅需 3 周即完成了开发上线，大幅加速了 Qwen3.5 整体模型在昇腾上的落地进程。

## Qwen3.5 + TileLang 收益

| 算子 | Baseline(us) | TileLang(us) | 性能提升比 |
| --- | --- | --- | --- |
| causal_conv1d(Decode) | 11173 | 4518 | 2.47x |
| chunk_gated_delta_rule_fwd_h | 4702 | 2727 | 1.72x |
| fused_sigmoid_gating_delta_rule(Decode) | 126 | 104 | 1.21x |
| (注：上表耗时数据基于特定测试条件获取。其中，causal_conv1d 呈现的为 1200 次调用的总累计耗时。) | <br> | <br> | <br> |

## 总结

TileLang + xLLM 为 Qwen3.5 在昇腾 NPU 上的算子适配提供了一条高效路线：

- 性能高: 关键算子最高实现 2.47x 加速，显著优于 Baseline 实现。

- 接入简单：Python 写 Kernel，AOT 自动编译，新增算子只需遵循三步模板。

- 可复用：标准化接入框架不仅适用于 Qwen3.5，也为后续新架构模型的快速部署奠定基础。

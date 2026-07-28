# LongCat-Flash-Lite昇腾推理优化实践

### 背景介绍

LongCat-Flash-Lite 是开源的高效 MoE 大模型。该模型在 LongCat-Flash 架构基础上引入 N-gram Embedding，将参数扩展从传统 MoE 专家层进一步拓展到 Embedding 层，在保持较低激活参数量的同时提升模型表达能力。LongCat-Flash-Lite 模型总参数量为 68.5B，每 token 动态激活约 2.9B 至 4.5B 参数，其中 N-gram Embedding 参数量达到 31.4B，占总参数约 46%。这一设计使模型在智能体、代码、数学推理等任务中具备较强竞争力，并在推理阶段带来低激活参数量与高吞吐的性能优势。

本次 LongCat-Flash-Lite 在昇腾完成推理适配，在TP4+DP8+EP32的切分模式下，围绕模型自身的 N-gram Embedding 与 EAGLE3 投机解码两项关键特性进行支持，并结合昇腾侧定制优化进一步释放推理性能：通过 ComputeNGramIds Ascend C 自定义算子加速 N-gram ID 生成，通过 SuperKernel 将模型算子融合成一个大算子，减少运行时调度耗时。适配优化后，模型 TPOT 突破 5ms，展现出优异的低时延推理性能。

### 一、模型技术特性

#### 1. N-gram Embedding：从扩专家到扩 Embedding

MoE 架构通过“总参数大、激活参数少”的方式提升模型容量。每个 token 只路由到部分专家，因此模型可以在控制单 token 计算量的同时扩大总参数规模。但当专家数量持续增加后，模型表现收益会逐渐放缓，同时还会带来更高的路由、通信、访存和调度压力。也就是说，继续单纯扩大专家规模，并不总是最高效的扩容方式。

LongCat-Flash-Lite 选择把一部分参数扩展放到 Embedding 层。Embedding 本质上是查表：输入 token ID 后，从参数表中取出对应向量。即使 Embedding 表很大，每个 token 实际访问的也只是少量表项，因此 Embedding 也具备天然的稀疏访问特征，适合作为 MoE 专家之外的另一条参数扩展路径。

N-gram Embedding 就是在这个思路上进一步扩展。这里的 gram 可以理解为一段连续 token 片段，N 表示片段长度。1-gram 是单个 token，2-gram 是由连续 2 个 token 组成的片段，3-gram 是由连续 3 个 token 组成的片段。比如输入序列为“你 很 好”，当处理“很”这个 token 时，“很”本身是 1-gram，“你 很”是 2-gram，“你 很 好”是 3-gram。

传统 Embedding 只查询当前 token 的向量，而 N-gram Embedding 会把当前 token 与前序 token 组成的连续片段也纳入表示。具体来说，N-gram 片段经过 Hash 映射后查询扩展 Embedding 表，再与基础 Embedding 结果融合，使当前 token 的输入表示不仅包含单 token 语义，也包含局部上下文组合信息。

![ngram_embedding_architecture](images/ngram_embedding_architecture.png)

这一机制的优势主要体现在以下几个方面：

- 提供区别于继续扩大专家数量的参数扩展路径。Embedding 查询天然具有稀疏访问特征，可以在扩大模型总参数量的同时，缓解 MoE 专家层持续扩张带来的通信和调度压力。

- 增强模型对局部 token 共现关系的建模能力。对于代码、工具调用、结构化文本等场景，局部片段往往携带较强语义信息，这使得 LongCat-Flash-Lite 在智能体和代码任务中可以发挥更大的优势。

- 降低 MoE 层激活压力。模型将一部分容量转移到 Embedding 层后，可以在较低激活参数量下保持较强表达能力，为推理部署提供更好的性能。

#### 2. EAGLE3 投机解码：一次验证推进多个 token

大语言模型解码通常是逐 token 自回归生成，每一步都需要调用主模型完成一次前向计算。随着输出长度增加，逐 token 解码会带来较高的端到端时延。投机解码的核心思想是：先由轻量 Draft 路径快速预测多个候选 token，再由目标模型对这些候选 token 进行验证。如果候选 token 被接受，就可以在一次主模型验证中推进多个 token，从而减少逐 token 调用主模型的次数。目标模型仍然负责最终校验，因此在提升速度的同时可以保持生成质量。

EAGLE3 的关键在于提升 Draft 路径的预测质量。它不是简单额外训练一个小模型去“猜 token”，而是复用目标模型自身的中间特征来构建 Draft 能力，使候选 token 更接近目标模型的真实分布，从而提升候选 token 的接受率。

EAGLE3 的核心特点包括：

- 直接预测 token。Draft 路径直接面向 token 预测目标学习，让生成过程更聚焦于产出可被目标模型接受的候选 token。

- 多层特征融合。目标模型不同层承载的信息并不相同：低层更偏词法和局部模式，中间层包含更丰富的语义结构，高层更接近最终输出分布。EAGLE3 融合目标模型低层、中层和高层特征，为 Draft 路径提供更完整的上下文信息，相比只依赖顶层特征更有利于生成高质量候选 token。

- training-time test。投机解码在真实推理时会连续生成多个候选 token，后续候选 token 会依赖 Draft 路径前一步的预测结果。如果训练阶段只学习单步预测，训练和推理之间容易出现分布不一致。EAGLE3 在训练阶段模拟多步推理过程，让 Draft 路径提前适应“用自己的预测继续往后生成”的场景，从而提升多步候选序列的稳定性。

### 二、昇腾定制优化

#### 1. ComputeNGramIds Ascend C 算子：加速 N-gram ID 生成

N-gram Embedding 在真正查表之前，需要先为每个 token 计算多个连续 token 片段对应的表项 ID。模型需要结合当前位置 token 与前序 token，按照不同片段长度 n 和不同子表 k 计算 Hash 结果，再加上各子表的前缀偏移，得到最终用于 Embedding 查表的行号。

如果这部分逻辑在框架层用多个小算子拼接完成，会产生较多调度开销和多余的内存搬运开销。为此，本次适配开发了 `ComputeNGramIds` Ascend C 自定义算子，将 N-gram ID 整个生成流程使用一个融合算子实现。

`ComputeNGramIds` 的输出形状为 `[token_num, (oe_n - 1) * oe_k]`，即为每个 token 生成所有 n 阶、所有子表对应的 N-gram ID。算子输入包括 `oe_weights`、`oe_mods`、`exclusive_oe_embeder_size_sums`、`oe_token_table`、`row_indices`、`column_starts` 和 `exclusive_req_len_sums` 等信息，用于描述 Hash 权重、各子表取模大小、Embedding 表前缀偏移以及不同请求在 token table 中的位置。

在 Kernel 内部，算子会先将 `oe_weights`、`oe_mods`、Embedding 表前缀偏移和请求长度等小规模元数据搬入 UB；随后按照 `(request, n, k)` 维度进行多核切分，每个核负责一部分请求、N-gram 阶数和子表组合。对于每个 token，算子在 NPU 侧从 `oe_token_table` 读取当前位置及其前序 token，完成 `token * weight`、取模累加、二次取模和前缀偏移相加，最终直接写出 N-gram ID。

通过这一 Ascend C 算子，N-gram ID 生成中的多层循环、Hash 计算、取模和偏移处理被融合到一个 NPU Kernel 内完成，减少了多个小算子调度开销，也避免了频繁的中间结果搬运，N-gram Embedding模块的整体效率得以提升。

#### 2. SuperKernel：融合算子减少调度耗时

在大模型推理过程中，耗时不仅来自矩阵乘等核心计算，也来自算子调度、下发和同步等非计算开销。针对这一问题，昇腾提供了通用的模型优化技术 SuperKernel。SuperKernel 可以将模型中指定范围内的多个算子融合成一个超级内核统一下发执行，从而减少算子之间频繁切换、逐算子调度和运行时等待带来的额外开销。

在 LongCat-Flash-Lite 的适配中，模型中所有算子都被圈定在 SuperKernel 融合范围内，形成模型级的大算子执行形态。能实现这一点依赖于模型本身的优秀设计，模型算子的shape较小，数据总量不大，提高或者降低 AIC 和 AIV 核数对算子本身执行性能影响不大，因此可以灵活调整算子 AIC 和 AIV 的配比满足 SuperKernel 的接入需求。

在使能SuperKernel前，从下图的Profiling中可以看到单次推理使用了4.69 ms，计算流水分析中有很多的气泡碎片。

![superkernel_before_profiling](images/superkernel_before_profiling.png)

使能 SuperKernel 后, 单次推理的耗时降低到了4.06 ms，计算流水是稳定且连续的，通过算子的融合消除了调度的开销时间。

![superkernel_after_profiling](images/superkernel_after_profiling.png)

这与 ComputeNGramIds 算子优化形成互补：ComputeNGramIds 负责将 N-gram ID 生成流程融合到 NPU Kernel 内，降低 N-gram Embedding 的局部计算开销；SuperKernel 则从整网执行层面压缩算子调度开销，使 LongCat-Flash-Lite 在低激活参数量和投机解码带来的结构优势之外，进一步获得端到端低时延收益。

### 总结

LongCat-Flash-Lite 是一次 MoE 模型未来发展方向的重要探索。模型通过 N-gram Embedding 将大规模参数扩展引入 Embedding 层，在保持低激活参数量的同时提升模型表达能力；通过 EAGLE3 投机解码，进一步提升推理阶段的有效 token 产出效率。模型在昇腾平台除了适配这些技术特性外，还利用昇腾的硬件和软件特性进一步优化，获得了不菲的性能收益。未来，昇腾 CANN 将持续面向 MoE、投机解码等大模型关键场景增强推理能力，助力更多开源模型在昇腾上高效部署与应用。

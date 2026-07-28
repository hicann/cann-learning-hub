# 星火大模型昇腾算力集群吞吐优化实践

## 长上下文时代的 Attention 计算挑战

随着大模型技术快速演进，长上下文能力正在成为模型处理复杂任务的重要基础。模型上下文窗口持续扩展，使代码理解、智能体任务、多文档问答、长报告分析等场景具备了更好的落地条件。但上下文长度的提升，也显著放大了训练与推理过程中的计算和访存压力，传统注意力机制在长序列场景下的性能瓶颈逐渐凸显。

在传统 Attention 架构下，每个 token 都需要与全量历史 token 建立注意力关联，核心计算复杂度随序列长度呈二次增长。随着上下文长度持续提升，如果 attention 侧仍沿用 dense attention计算方式，相关计算开销将快速放大，并可能成为影响整体吞吐与部署效率的关键瓶颈。

2025 年 9 月 29 日，DeepSeek-V3.2-Exp 引入了 DSA（DeepSeek Sparse Attention）动态稀疏注意力机制，以缓解长上下文场景下的 Attention 计算压力。CANN 开源社区围绕 DSA 训练与推理流程率先完成原生能力适配，并于 2025年 11 月陆续开源 DSA 系列核心算子，覆盖稠密预热训练、稀疏训练和推理部署等关键阶段，支持开发者在昇腾 NPU 上完成 DSA 结构适配与高效推理。科大讯飞星火大模型依托昇腾算力集群搭载 MoE 稀疏架构与 DSA 优化方案，在硬件算力规模保持不变的前提下，实现长上下文场景推理吞吐量提升 4.5 倍。

## DSA 的结构与工作原理

DSA 是面向长上下文训练与推理场景设计的稀疏注意力机制，最早由 DeepSeek-V3.2 技术报告提出（https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf）。其核心思路是通过 Lightning Indexer 与细粒度 Top-K token 选择机制，将 MLA 中的全量上下文交互转化为动态稀疏计算，从而降低计算量与访存开销。

DSA结构如下图所示，其核心架构由三个环节构成：

- Lightning Indexer：轻量级索引器，用于计算当前 query token 与历史 token 的相关性分数；

- Top-K Token Selection：根据索引器输出的相关性分数，选择得分较高的 key-value token，形成稀疏索引；

- Sparse Attention：主 attention 仅对被选中的关键 KV token 执行计算。

![dsa_sparse_attention_optimization](images/dsa_sparse_attention_optimization.png)

DSA

DSA结构图

在完整计算流程中，DSA 会先通过 Lightning Indexer 为每个 Q 计算其与历史 K/V token 的相关性得分，再通过 Top-k Token Selector 模块选出得分最高的 k 个关键 K/V token，并仅对这些 K/V token 执行 Attention 计算，从而实现稀疏 Attention。通过这一设计，主 attention 的复杂度可由传统 dense attention 的 O(L²) 降低到 O(L×k)，其中 L 为序列长度，k 为单个 query token 选中的 KV token 数，且通常远小于 L。在尽量保持模型效果的同时，DSA 能够有效缓解长序列场景下的计算和访存压力。

## CANN DSA 系列算子：覆盖训练与推理全流程

围绕 DSA 的训练与推理流程，CANN 开源社区提供了一组核心算子，可按稠密预热训练、稀疏训练和推理部署三个阶段配合使用，使开发者能够在昇腾 NPU 上将模型从 dense attention 平稳迁移至 DSA 架构下的 sparse attention，且保持性能稳定。

## 稠密预热训练阶段：初始化 Lightning Indexer

这是一个简短的预热阶段，旨在初始化 Lightning Indexer。在此阶段，模型仍采用原始的 dense attention，同时冻结除 Lightning Indexer 之外的所有参数。训练目标是使 Indexer 的打分输出与主注意力机制中的打分分布保持一致。

在稠密预热训练阶段，模型仍然使用 MLA，主要目标是初始化并训练 Lightning Indexer，使其学习 dense attention 中的注意力分布。索引器通过 KL loss 学习与主 attention 分布对齐，从而获得后续稀疏选择所需的 token 重要性判断能力。

CANN 在该阶段提供以下算子支持：

| 算子 | 功能说明 |
| --- | --- |
| DenseLightningIndexerSoftmaxLse | 计算 dense 场景下 Lightning Indexer softmax 所需的 LSE 中间量 |
| DenseLightningIndexerGradKLLoss | 计算 Lightning Indexer 反向梯度并融合 KL loss，用于对齐 dense attention 分布 |

## 稀疏训练阶段：启用 Sparse Flash Attention

完成稠密预热训练后，模型进入稀疏训练阶段。DSA 正式引入细粒度 token 选择机制，同时优化主模型和 Indexer。Lightning Indexer 根据相关性分数生成 Top-K 稀疏索引，主 attention 不再访问全部历史 token，而是仅对被选中的关键 KV token 执行 attention 计算。

CANN 在该阶段提供以下算子支持：

| 算子 | 功能说明 |
| --- | --- |
| LightningIndexer | 根据输入生成 Top-K 稀疏索引，确定当前 token 关注的关键上下文 |
| SparseFlashAttention | 基于稀疏索引执行高效 sparse attention 前向计算 |
| SparseFlashAttentionGrad | 支持 sparse attention 反向计算，处理注意力梯度 |
| SparseLightningIndexerGradKLLoss | 在稀疏训练阶段继续计算 Lightning Indexer 梯度和 KL loss，优化索引器 |

## 推理部署阶段：高效长上下文推理

在推理部署阶段，LightningIndexer 根据当前上下文动态生成稀疏索引，SparseFlashAttention 基于索引结果完成稀疏注意力计算。通过仅访问与当前 query token 更相关的关键 KV token，DSA 能够减少长文本推理中的 attention 计算量，在尽量保持模型生成效果的同时提升长上下文推理效率。

## 长上下文场景下的性能收益

基于 CANN DSA 系列算子，讯飞的星火大模型完成了Attention计算的动态稀疏化适配，并在推理和训练两大场景中验证了收益。

训练场景：在训练侧，星火大模型基于昇腾千卡集群，通过约 3 天的增量训练，即可完成从稠密注意力架构到DSA稀疏注意力架构的迁移，并达到原模型同等效果水平。该方式显著缩短了结构切换所需的适配周期，降低了模型迭代过程中的算力资源投入和工程工作量。

推理场景：在 64 路并发、固定输出长度 1K、整体算力保持不变的测试配置下，DSA算法网络在长输入场景表现出更明显优势。32K 输入长度下，首响性能约为原方案的 1.34 倍；64K 输入长度下，首响性能约为原方案的 1.55 倍。进入 Decode 阶段后，长上下文收益进一步放大：64K 输入场景下 Decode 性能约为原方案的 2.00 倍，128K 超长输入场景下 Decode 吞吐最高可达到原方案的 4.5 倍。

## 总结

CANN 通过开源社区模式，在 Transformer 相关仓中提供 DSA 系列算子，为大模型的结构优化、训练迁移和推理性能调优提供了基础能力支持。基于这些算子构建的 DSA 算法网络，可支持模型从 dense attention 平稳迁移至 DSA 架构下的 sparse attention，并在较短训练周期内恢复到原架构的同等效果水平。

在推理侧，DSA 进一步释放了长上下文场景下的性能潜力。在 MoE 星火大模型、64 路并发、固定输出 1K 的测试配置下，128K 超长输入场景的 Decode 吞吐最高可达到原方案的约 4.5 倍，有效缓解长上下文推理中的 attention 性能瓶颈。整体来看，DSA 以动态稀疏化方式减少 attention 计算量，在保持模型效果的同时提升长序列处理效率，进一步增强了超长上下文和高并发推理场景下的性能表现。

## 参考资料

- ops-transformer 算子仓：
https://gitcode.com/cann/ops-transformer

- DSA系列算子源码：
https://gitcode.com/cann/ops-transformer/tree/master/attention

- DeepSeek-V3.2 技术报告：https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

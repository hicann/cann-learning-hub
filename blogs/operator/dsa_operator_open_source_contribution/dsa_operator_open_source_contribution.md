# 从开源到共建：DSA算子的贡献之路

在大模型长序列处理场景中，传统 Transformer 注意力机制的计算量会随序列长度平方级增长，长上下文训练与推理成本也随之快速上升。DeepSeek 在 DeepSeek-V3.2-Exp 中引入 DeepSeek Sparse Attention（DSA）稀疏注意力机制，并在后续 DeepSeek-V3.2 体系中延续这一方向。DSA 通过“轻量索引筛选 + 细粒度 token 选择 + 稀疏 Attention 计算”的组合，将 Attention 计算从全量 token 交互转向 Top-k 关键 token 计算，从而提升长上下文场景下的训练与推理效率。近期，由科大讯飞主导、联合CANN共同开发的 DSA 训练算子，已贡献至CANN开源社区ops-transformer 算子仓。该实践不仅解决了讯飞自研 MoE 长序列星火大模型的实际训练需求，也形成了“基于开源算子定制化开发，再反向贡献社区”的可复用路径。本文将从 DSA 的基础机制入手，介绍讯飞如何基于CANN开源算子完成定制化开发，深度优化DSA关键算子性能，并将成果贡献回社区。

## 1. DSA是什么？

DSA（DeepSeek Sparse Attention）是一种面向长序列场景的稀疏注意力机制，核心由 Lightning Indexer（闪电索引器）与细粒度 token 选择机制构成。其设计目标是在尽量保持模型效果的同时，降低长上下文场景下全量注意力计算带来的计算和存储开销。在计算流程上，DSA 会先通过 Lightning Indexer 为每个 Q 计算其与历史 K/V token 的相关性得分，再通过 Top-k Selector 模块选出得分最高的 k 个关键 K/V token，并仅对这些 K/V token 执行 Attention 计算，从而实现稀疏 Attention。通过这种方式，DSA 将 Attention 计算从传统全量注意力的 O(L²) 降低到 O(Lk)，其中 L 为序列长度，k 为每个 Q 选中的关键 K/V token 数。上述两个环节分别对应 DSA 论文中的 Lightning Indexer 和 Top-k Selector 模块，如下图所示：

![dsa_sparse_attention_architecture](images/dsa_sparse_attention_architecture.jpeg)

DSA结构图
相较于传统注意力机制，DSA 的核心价值体现在三方面：

1. 效率提升：在长上下文场景下，通过稀疏化Attention 计算，降低训练与推理阶段的计算成本。DeepSeek-V3.2-Exp 引入 DSA 后，长上下文效率明显提升，同时基准评测表现与 DeepSeek-V3.1-Terminus 基本持平。

2. 资源优化：通过减少Attention 阶段实际参与计算的 token 数量，降低长序列场景下的计算压力，适配超长序列训练与推理需求；

3. 工程友好：DSA 机制可与 FP8、BF16、FP16 等低精度优化方向配合，也为不同硬件平台上的长上下文训练和推理算子优化提供了明确落点。

DSA 的典型应用场景包括超长文本生成、大模型预训练、多模态长序列处理、MoE 长序列训练等，尤其适用于 128K 及以上上下文长度的大规模模型开发。

## 2. 开源算子的定制化需求与 FP32 精度支持

CANN开源社区当前已提供 DSA（DeepSeek Sparse Attention）的 6 个核心算子，覆盖稀疏注意力前向计算、稀疏索引生成以及训练阶段的反向传播与损失计算等关键能力。（详细实现可参考 ops-transformer 仓库的 attention 目录：https://gitcode.com/cann/ops-transformer/tree/master/attention）。这 6 个算子分别为：

- SparseFlashAttention：用于执行基于稀疏索引的注意力前向计算。

- LightningIndexer：用于完成 Top-K 关键位置筛选与稀疏索引生成。

- SparseFlashAttentionGrad：用于支撑稀疏注意力反向传播。

- SparseLightningIndexerGradKLLoss：用于稀疏分支的反向及 KL 损失计算。

- DenseLightningIndexerGradKLLoss：用于稠密参考分支的反向及 KL 损失计算。

- DenseLightningIndexerSoftmaxLse：用于提供稠密分支所需的 Softmax 与 Log-Sum-Exp 中间统计量。

上述算子共同构成了 DSA 从索引筛选、注意力计算到训练优化的完整能力体系。
在现有实现中，部分 DSA 核心算子的权重相关参数默认支持 FP16 与 BF16。该精度配置能够较好兼顾计算效率、显存占用与基础精度需求，适用于大模型训练与推理中的主流场景。但在讯飞自研 MoE 长序列模型适配过程中，相关权重参数采用 FP32 数据类型。由于这部分CANN的DSA 算子在权重输入上尚未支持 FP32，在开展不同实现路径下的训练效果对比验证时，需要额外处理数据类型不一致带来的精度对齐、结果比对与调试问题，进而增加适配和验证成本。基于上述需求，本次优化针对 4 个 DSA 核心算子的相关参数新增 FP32数据类型支持，具体包括：

- LightningIndexer：weight 参数新增 FP32 支持。

- SparseLightningIndexerGradKLLoss：weight 和 dweight 参数均新增 FP32 支持。

- DenseLightningIndexerGradKLLoss：weight 和 dweight 参数均新增 FP32 支持。

- DenseLightningIndexerSoftmaxLse：weight 参数新增 FP32 支持。

## 3. 从需求到贡献：外部开发者实践路径

以SparseLightningIndexerGradKLLoss算子的 FP32 支持优化为例，外部开发者依托CANN开源社区的开放生态，完成了从需求分析到代码合入的实践，具体路径如下：

### 3.1 前置准备阶段：依托清晰文档，快速吃透算子核心逻辑

本次开发基于CANN官方CANN 算子开源仓ops-transformer开展，该仓库面向 Transformer 类大模型算子，提供算子源码、构建方式、样例和配套文档，是开发者进行算子学习、适配与贡献的重要入口。现阶段，仓库提供了快速入门教程和系统化的算子开发资料，帮助开发者理解算子工程结构、编译方式和调试流程。同时，仓库支持多种源码编译和样例执行方式，包括自定义算子包、ops-transformer 包和 ops-transformer 静态库等。对于需要快速定制和验证的场景，自定义算子包方式可以实现算子间解耦，支持单个或部分算子独立编译、独立调试，降低全仓编译带来的时间成本。在此基础上，CANN社区为相关算子提供了接口说明 README，覆盖功能定位、输入输出参数、数据类型要求、核心计算公式和调用示例。开发者可以通过文档快速理解 SparseLightningIndexerGradKLLoss 的设计原理：该算子是 Lightning Indexer 的反向算子，并融合了 KL Loss 计算能力，用于训练 Lightning Indexer，使其选择结果尽量接近Attention 得分所构造的目标分布。开发完成后，也可参照文档中的调用示例做快速修改以验证新功能的正确性。

详见：

https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_sparse_lightning_indexer_grad_kl_loss.md

### 3.2 代码开发阶段：精准适配 FP32 需求，完成代码修改

本次开发的核心目标是让 SparseLightningIndexerGradKLLoss 的 weight 和 dweight 参数能够接收并处理 FP32 数据。围绕这一目标，开发者重点完成了以下四个方面的核心修改：

- 算子定义层：修改算子 def 文件，新增 FP32 数据类型的支持配置，明确 FP32 精度下的参数格式要求；

- 形状推导层：更新 infershape 文件，适配 FP32 精度的张量形状推导逻辑，保证张量计算的维度一致性；

- 分片策略层：优化 tiling 文件，使算子在 FP32 数据类型下能够生成匹配的计算分片策略；

- 核心计算层：修改 kernel 文件，补齐 FP32 精度路径下的核心计算逻辑，保证正反向计算结果正确。

同一类修改也扩展到LightningIndexer、DenseLightningIndexerGradKLLoss和 DenseLightningIndexerSoftmaxLse等相关算子，最终形成 DSA 训练算子 FP32 支持的整体能力。相关代码改动可参考：
https://gitcode.com/cann/ops-transformer/pull/2033

### 3.3 定位调试阶段：通过 Issue 高效协作，快速解决问题

开发过程中，开发者依托CANN开源社区的 Issue 交流功能，快速发起问题反馈，详细描述了问题现象、复现步骤及测试环境。社区维护者与CANN技术专家基于这些信息给出调试建议，协助定位精度适配、编译构建或运行验证中的问题。这种协作方式让问题反馈、定位、修复和验证形成闭环，减少了外部开发者理解算子工程细节和排查底层问题的成本，也提升了定制化开发效率。在完成代码开发、问题定位和多轮验证后，相关FP32支持能力，最终通过社区PR形式贡献至ops-transformer仓库，为后续类似长序列训练场景提供了可复用的算子能力。

## 4. 案例价值：开源生态的双向奔赴

本次 DSA系列算子的4个算子weight权重新增FP32数据类型及社区反向贡献案例，不仅切实解决了讯飞自研 MoE 长序列模型的实际训练需求，更验证了CANN开源社区 "开放协作、共建共享" 的生态价值，为更多外部开发者提供了可直接参考的实践范本。从社区侧来看，清晰的文档指引、可复用的编译调试流程和高效的 Issue 协作机制，降低了外部开发者参与算子开发的门槛，使开发者能够更快理解算子逻辑、定位问题并推进优化；而从开发者侧来看，基于实际业务场景的需求反馈与代码贡献，也让开源算子能够覆盖更多复杂的工业级应用场景，推动算子能力持续完善，形成了正向循环。更重要的是，本案例展示了一条从实际业务需求出发、基于开源算子完成定制化开发，并将通用能力反向贡献给社区的可参考路径。这种正向循环，正是开源生态持续繁荣的关键。未来，欢迎更多开发者参与CANN算子生态的共建，基于实际场景需求进行定制化优化，将优质的开发成果反向贡献给社区，共同推动形成更繁荣、更具活力的开源生态。

## 参考资料

- SparseLightningIndexerGradKLLoss 接口文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/aolapi/context/ops-transformer/aclnnSparseLightningIndexerGradKLLoss.md

- ops-transformer PR：
https://gitcode.com/cann/ops-transformer/pull/2033

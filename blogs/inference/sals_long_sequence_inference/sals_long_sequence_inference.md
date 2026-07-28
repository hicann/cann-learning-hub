# SALS长序列推理优化

## 背景

现在的大模型能力越来越强，大家在工作与生活中也越来越离不开它，可一到长序列推理，模型不仅算得慢而且占内存大，直接拖慢了它在实际场景里的效率。论文SALS:Sparse Attention in Latent Space for KV cache Compression(arxiv.org/abs/2510.24273)提出的SALS算法是面向长序列场景的通用型加速算法。算法采用在线稀疏优化方案，在实现推理时延大幅降低的同时，保障模型推理精度近乎无损，显著提升了模型长序列推理能力。区别于DeepSeek、Qwen等模型自带的稀疏功能，SALS算法作为通用方案可直接应用于各类开源模型，显著提升其长序列推理能力。

本文基于SALS算法原理，重点介绍了方案内核心算子Quant Sals Indexer(QSI)和Sparse Flash Attention Antiquant(SFAA)在昇腾NPU上的实现方案及性能优化策略。实测表明，GQA结构的大模型使用QSI、SFAA算子实现SALS算法，在10K序列长度场景下相较于稠密FA算子，四倍稀疏每层可以获得10%-15%的性能收益，六倍稀疏每层可以获得15%-25%的性能收益，并且这一性能收益随着序列的增长可以得到进一步提升。

## 算法简介

SALS算法主要分为稀疏token选择与稀疏attention计算两部分，如下图所示：

![sals_algorithm_overview](images/sals_algorithm_overview.png)

除去头部low rank预处理小算子，主要由QSI与SFAA两个融合算子组成：

- QSI

  - 稀疏token选择算子，通过query与low rank key cache计算出topk token索引位置；由于需要计算全序列空间，本身开销不可忽略，算法层面通过headdim低秩与低比特量化降低开销；

- SFAA

  - 根据QSI输出的topk indices索引进行稀疏attention计算，总体计算流程与普通attention无异，主要难点如何优化稀疏访存；

## Quant Sals Indexer（QSI）

`Quant Sals Indexer`是基于一系列计算操作得到每一个query token对应的Top- 个位置的算子。Decode场景，对于Index Query （query这里的维度为1，是由前置算子对query的group维度取平均），给定上下文 Index Key ，其中 为每一个头的维度， 是上下文的长度。同时，考虑到上下文长度较长，我们采取了int4的量化方案，计算公式如下：

其中， 为固定尾部块索引集合；块级 Log-Sum-Exp 算子  定义为：

可拆分为如下计算步骤：

1. Query  与 Key 进行矩阵乘法，得到

2. 反量化：

1. 块级 Log-Sum-Exp (LSE) 计算：将 Key 序列按块大小  划分为  个块，对每个块（排除固定尾部块）计算 LSE 值（用于近似该块的 Attention 权重总和）

2. 排序与 Top- 选择

3. 拼接固定尾部索引

## QSI算子 Tiling 设计

我们将QSI算子基本块大小设置为，这个值的选取可以保证流水线头尾开销可控以及 Cube 核与 Vector 核之间同步开销可被计算掩盖，并尽量减少算子内部的核间同步的scalar开销。

核内 Tiling 设计采用了如下方案：

- L1 空间划分为如下部分：

  -  2-Buffer循环复用：分配；

  -  矩阵 3-Buffer 循环复用：分配，L1层级基本块为 。

- L0A，L0B，L0C使能 Double Buffer，分别划分，，，这是一种对昇腾较为亲和的设置，可以提高算力利用率。

## QSI Top-k 计算实现

QSI融合算子的核心是在长达数十万的序列中，为每个 token 高效地筛选出分数最高的 （例如2048）个索引。同时，对于算子而言，Top- 的计算必须是准确无误的，不能采用近似算法求解。

当前的实现方案基于昇腾支持的排序指令进行全量排序，Top- 计算方案过程分为三步：

- 分组排序：将每32个 Sparse Block按照其  进行稳定降序排列，输出其排序向量以及对应索引向量，直到将整个序列的 Sparse Block分组排序完毕。

![group_sort_stage](images/group_sort_stage.png)

- 归并：将至多4组（可为2/3组）长度为32、128、512的已排序向量进行归并，直到合并后的向量长度达到2048。合并排列仍然按照其对应  进行稳定降序排列，最终输出的向量长度为4组输入向量长度之和，同时输出其对应索引向量。

![merge_sort_stage](images/merge_sort_stage.png)

- 规约：将至多4组长度为2048已排序的向量进行归并。取出前  个参数组成的向量与另外未参与合并排序的有序向量重复进行 Top- 计算，直到将所有长度为2048的有序向量比较完毕，得到最终所有向量中得分最高的2048个对应的 Index。

![topk_reduction_stage](images/topk_reduction_stage.png)

## Sparse Flash Attention Antiquant（SFAA）

在SALS中，Key和Value设计为int8的量化输入以应对长上下文场景下，显存占用过大的挑战，SFAA计算可表示为

其中  为基于某种选择算法 (如`Quant Sals Indexer`) 得到的重要性较高的 Key 和 Value，一般具有稀疏或分块稀疏的特征， 为  每一个头的维度。
SFAA针对离散访存进行了指令缩减及搬运聚合的细致优化，计算流程沿用 FlashAttention的计算流程，分为四个阶段：

- ：；

- ：online softmax；

- ：$P@V；

- ：rescaling 。

## SFAA 算子 Tiling 设计

`SFAA`一次迭代计算的基本块大小为。以如下Decode场景为例，为10，时，表示所有共用一组稀疏索引，那么一次迭代计算的基本块大小就为

算子核内 Tiling 设计如下：

- L1 空间划分为如下部分：

-  矩阵常驻，分配；

  -  矩阵 3-Buffer 循环复用，分配

  -  矩阵 4-Buffer 循环复用，分配

  - 其中矩阵每次的搬运大小为

- L0A，L0B，L0C使能 Double Buffer，分别划分为，，，在  阶段一次搬入 L0A 和 L0B的矩阵块大小分别为 ；在  阶段一次搬入 L0A 和 L0B 的矩阵块大小都为 。

## SFAA Pipeline 设计

对于复杂的融合算子而言，Cube 核和 Vector 核之间的高效协同是发挥硬件算力的核心挑战，流水掩盖作为连接二者的核心调度机制，其设计优劣直接决定了算子性能。SFAA有4个计算阶段,依赖关系多，朴素的流水排布会导致计算过程中出现大量的空泡。如下图的上半部分所示，由于计算流程中的不同阶段之间存在数据依赖，不论是 Cube 流或 Vector 流上都出现了很多的空闲部分，导致算力利用率较低。本算子采用 Preload 流水排布以消除依赖，实现了除头尾以外的平台期部分 Cube 流完美掩盖 Vector 流，效果如下图下半部分所示，可以看到这种方式相较于朴素的流水排布有较大的性能收益。

![preload_pipeline_optimization](images/preload_pipeline_optimization.png)

## SFAA访存优化

实际的访存行为可以分为两个步骤：指令下发及数据搬运。对于非 Sparse 场景的 Attention 算子，数据访存一般是连续的，因此单次数据搬运耗时相较于指令下发耗时更长，可实现连续的多次访存，达到较高的访存带宽。然而对于SFAA 的离散访存而言，单次访存的耗时大幅下降，甚至可能小于指令下发的耗时，这就会导致严重的指令下发阻塞，如下图所示。

![discrete_memory_access_bottleneck](images/discrete_memory_access_bottleneck.png)

为了优化离散访存，我们利用 Vector 核发射访存指令，将指令的下发效率提高一倍。同时，我们将离散的数据点两两聚合，利用 srcGap 参数实现成对的数据访存，提升访存带宽，如图所示。

![vector_pairwise_memory_access](images/vector_pairwise_memory_access.png)

此外，为了缓解Vector核的访存负载，我们还卸载了一部分访存到Cube核，从而达到Cube和Vector核访存的均衡，相比于纯Vector核的访存，卸载一部分访存负载到Cube核实测性能可提升10%~15%

![cube_vector_memory_balance](images/cube_vector_memory_balance.png)

## AICPU Tiling分核优化

在实际业务场景中，不同核之间的负载会由于batch序列长度差异、因果掩码等因素存在差距，类似于木桶效应，最慢的核决定了算子端到端计算耗时，为此，我们提供了专门的tiling算子为对应算子计算最优的分核策略。

## AICPU Tiling

1. Tiling拆分

传统意义上的算子Tiling包含了TilingKey的计算和分核计算，当采取复杂分核策略时，会导致这部分计算耗时过长，出现host计算bound。

![aicpu_tiling_split](images/aicpu_tiling_split.png)

针对这个问题的解决思路是将传统意义上的算子Tiling拆分开，将分核计算封装为一个独立的aicpu tiling算子，用aicpu的计算资源缓解host bound，以aicore算子的输入shape为入参，以负载均衡分核结果为出参，与aicore算子形成上下游调用关系。

1. 多流拆分

将aicpu tiling算子和对应下游算子放在两个流中执行，可以实现多流掩盖，aicpu tiling算子可以提前启动kernel的执行，两流之间通过record与wait同步。

![aicpu_tiling_multistream_overlap](images/aicpu_tiling_multistream_overlap.png)

## Tiling算法

针对不定长输入，若按照计算行进行负载分核，容易产生极度不均衡的分配结果，导致慢核严重影响计算性能，当前Tiling算法的核心就是通过对计算行进行切块->分组->计算->聚合的形式，达到核间计算负载趋近均衡。

![tiling_load_balancing_workflow](images/tiling_load_balancing_workflow.png)

以SparseFlashAttentionAntiquant为例，该算子的分核存在两个挑战：

- 挑战1：不同batch的序列长度差异、掩码导致每个计算行的计算量严重不均

针对这个问题，我们以FlashAttention的理论基础作为指导，对计算行进行分块，建立基本块开销模型，使用理论模型预估每个基本块的计算耗时，进而估计总开销与每个核的平均开销，以平均开销为理论上限依据，完成多核负载均衡分配。

![basic_block_cost_model](images/basic_block_cost_model.png)

采用三级分配逻辑，降低分配计算次数：

![hierarchical_task_allocation](images/hierarchical_task_allocation.png)

- 挑战2：单个计算行可能被切分到多个核，总体产生不定量的归约计算任务

计算行如果被分配到不同核上，则需要在FlashAttention计算流程结束之后，对每个计算行的各个部分进行归约汇总，不同输入产生的归约计算任务数量及所属计算行是不定的，为了避免该过程堆积在某些vector核上造成计算bound，我们将所有任务沿QSxGroup的轴划分，再分配到所有的vevtor核上，进行KV轴的reduce计算，从而达到归约计算的负载均衡。

QSI、SFAA算子现已开源至CANN社区，欢迎大家与我们issue互动：

https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/quant_sals_indexer
https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/quant_sals_indexer_metadata
https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/sparse_flash_attention_antiquant
https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/sparse_flash_attention_antiquant_metadata

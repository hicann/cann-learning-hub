# DeepSeek V4昇腾超节点支持

2026年4月24日，DeepSeek V4-Pro和DeepSeek V4-Flash正式发布并开源，模型上下文处理长度由原有的128K显著扩展至1M，首次增加了KV Cache滑窗和压缩算法，大幅减少Attention计算和访存开销，并通过模型架构创新更好地支持了Agent和Coding场景。昇腾一直同步支持DeepSeek系列模型，本次通过双方芯模技术紧密协同，实现昇腾超节点全系列产品支持DeepSeek V4系列模型。昇腾950通过融合kernel和多流并行技术降低Attention计算和访存开销，大幅提升推理性能，结合多种量化算法，实现了高吞吐、低时延的DeepSeek V4模型推理部署。昇腾A3超节点系列产品也全面适配，同时为便于用户快速微调，提供了基于昇腾A3集群的训练参考实现。

## 1. 昇腾950超节点长文本推理

昇腾950超节点重新定义长文本推理的性能天花板，实现DeepSeek V4-Pro 20ms 和DeepSeek V4-Flash 10ms低时延推理基于DeepSeek V4-Pro模型，在8K输入场景，昇腾950超节点可实现TPOT约20ms时单卡Decode 吞吐4700TPS。DeepSeek V4-Flash模型，8K长序列输入场景下可实现TPOT约10ms时单卡Decode吞吐1600TPS（注：上述Benchmark数据均基于Offine推理模式采集，不包含Serving调度和框架负载均衡影响）。极低时延的实现源于昇腾950代际底层架构的三大升级：

- 原生精度加速： 全面支持 FP8 、MXFP8、MXFP4等数据格式，在保证模型精度的同时，可实现内存占用降低50%+，计算能力翻倍。

- 稀疏访存优化： 针对MoE模型的离散访存特征，通过大幅提升硬件级稀疏访存能力，有效解决了专家路由过程中的带宽瓶颈。

- 增强Vector与Cube间的数据通路： 创新的存储架构设计，实现了向量单元（Vector）与矩阵单元（Cube）的 Memory通路，极大地降低了端到端推理时延。

除了底层架构的升级，昇腾950超节点从基础器件、协议算法到光电互联，实现了系统级的创新突破，支持用户以64卡为步长按需扩展，可实现8192卡无收敛全互联，提供业界最大Scale Up能力。

同时我们联合定义昇腾超节点，进一步大幅提升延迟和吞吐，同时实现低成本，且兼顾万卡级别的Scale out 集群规模。解决了长序列4K到1M 序列长度范围内都有低延迟和高吞吐。此架构支持基于NAND SSU的超低成本、超大容量、高性能KV cache有效支撑支持长序列应用。

## 2. A3超节点系列产品，实现DeepSeek V4-Flash模型单卡Decode吞吐2000+ TPS

Atlas 900 A3 SuperPoD液冷超节点及Atlas 800 A3风冷超节点采用平等架构、全局内存统一编址、点对点互联带宽达784GB/s。提供32到384多种规格满足不同业务需求，昇腾超节点是国内唯一成熟规模商用的超节点产品，满足互联网、运营商、金融等行业对大模型推理超高吞吐、超大并发的极致性能需求。

基于昇腾A3 64卡超节点结合大EP模式部署，DeepSeek v4-Flash 模型，8K/1K输入输出场景，基于vLLM推理引擎可实现2000+ TPS的单卡Decode吞吐，单卡吞吐持续提升。针对DeepSeek V4-Pro模型，昇腾A3同步支持推理部署，性能持续优化中。

## 3. 昇腾同步支持并开源DeepSeek V4 复杂Sparse Attention + mHC架构续训练参考实现，TorchTitan-NPU 携手 Autofuse，助力训练轻松入图、开箱即优

昇腾 CANN 基于 A3 64卡超节点正式完成 DeepSeek V4-Flash模型续训练（CPT）的0-day 适配支持。通过 TorchTitan-NPU 插件与 Autofuse 自动融合技术的深度协同，实测模型吞吐量最高达到 1100 tokens/p/s，实现模型训练性能开箱即优。而这一亮眼的开箱表现，主要源自以下三大维度的硬核系统级优化：

极简分布式并行架构： 突破传统复杂的混合并行设计，采用超节点亲和的大 EP + 纯 FSDP 的极简并行切分策略，以极低适配成本和通信开销达成内存占用最优，实现易用性与性能的较好均衡

原生“入图”与自动融合：TorchTitan-NPU 深度适配 torch.compile 机制，使能训练入图技术，依托 Inductor + AutoFuse（基于Ascend C的Codegen后端）实现端到端的 Vector 算子自动融合，为整网带来高达 31.8% 的开箱即用性能收益

稀疏Attention高效融合算子： 针对稀疏注意力等复杂结构，开发 SparseAttnSharedkv、LightningIndexer 等多个高效的 NPU 融合算子，从负载均衡分核计算、内存与计算均衡等维度协同优化，充分释放芯片稀疏算力

## 4. PyPTO编程新范式与TileLang方案同步开源

为了解决自定义算子开发门槛高、周期长的痛点，昇腾CANN 推出了 PyPTO编程范式。PyPTO 提供完善的 Python API，使开发者能够以符合Python习惯的语法进行算子开发。

高效的算子开发：PyPTO 依托内置高级编译优化，可自动完成流水编排与内存管理，使开发者无需关注硬件细节而专注于计算流表达，实现DeepSeek V4新一代模型算子开发周期可缩短至天级。

高性能Kernel自动生成：针对 Attention、Compressor、mHC 等复杂逻辑算子，PyPTO 可自动生成高度优化的 Kernel，避免开发者手动处理繁琐的同步与数据搬运，显著缩短从算法验证到部署落地的开发周期。

PTO ISA虚拟指令集跨代兼容：PyPTO 基于PTO虚拟指令集（PTO ISA），实现了对硬件新特性的“零感适配”，针对不同代际芯片统一指令接口，实现了同一套算子代码，在不同代际芯片上的兼容实现。借助毕昇编译器的VF（Vector Fusion） 自动融合能力，可在 micro kernel 级别实现更优融合。

TileLang社区生态：TileLang-Ascend 是 TileLang 针对华为昇腾平台深度优化的实现，分别对应Tilelang-Ascend的Expert和Developer开发模式，提供AscendC基础指令和PTO AS两种对接层次，为各种编程前端语言和编译器提供多层开放接口。DeepSeek V4模型相关实现已在 TileAI 开源社区正式发布，后续将持续推进性能优化与功能迭代。

昇腾A2、A3及950全系列产品适配DeepSeek v4-Flash、DeepSeek v4-Pro。昇腾始终致力于为世界提供新选择，以极致的算力与开放的生态，加速 AI 产业的繁荣。我们期待与广大客户及开发者携手共进，在 DeepSeek V4 的新纪元中探索无限可能。

## 5. 资源链接

- DeepSeek V4 模型推理优化实践：https://gitcode.com/cann/cann-recipes-infer/tree/master/docs/models/deepseek-v4/deepseek_v4_inference_guide.md

- DeepSeek-V4 Ascend C 融合算子优化:https://gitcode.com/cann/cann-recipes-infer/tree/master/docs/models/deepseek-v4/deepseek_v4_ascendc_operator_guide.md

- 基于CANN平台的TorchTitan-NPU + AutoFuse 极简训练优化实践：
https://gitcode.com/cann/cann-recipes-train/blob/master/docs/llm_pretrain/deepseek-v4_torchtitan_npu_autofuse.md

- 大模型推理引擎 vLLM 及昇腾实现：https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V4.html

- 大模型推理引擎 SGLang 及昇腾实现：https://github.com/sgl-project/sglang/issues/23598

- TileLang-Ascend 开源社区：
https://github.com/tile-ai/tilelang-ascend

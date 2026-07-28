# xLLM大模型推理性能优化

## 1. 背景介绍

随着大模型在智能客服、搜索推荐、智能购物等互联网场景的持续应用和扩展，业界对于推理性能和部署效率提出了更高的要求。此次优化方案通过开源AI框架xLLM与Atlas 800I A3系列产品设备深度结合优化，在20天内显著提升了Qwen3-30B-MoE，DeepSeek-V3.2，ChatGLM4.6/4.7的性能表现，提升比例高达58%~103%，达到业界的一流水准。

## 2. xLLM介绍

xLLM是一个高效的开源大模型推理框架，提供企业级的服务部署，使得推理服务性能更高、成本更低。该框架采用服务-引擎分离的推理架构，通过服务层的在离线请求弹性调度、动态PD分离、EPD混合机制及高可用容错设计，结合引擎层的多流并行计算、图融合优化、投机推理、动态负载均衡及全局KV缓存管理，实现推理效率突破性提升。目前xLLM在昇腾上已支持主流大模型（如 DeepSeek-V3.1，Qwen2/3等）的高效部署，助力企业实现高性能、低成本的 AI 大模型应用落地。xLLM已在客户的多个业务场景落地，涵盖智能客服、风控、供应链优化、广告推荐等多种场景。

![xllm_framework_architecture](images/xllm_framework_architecture.png)

## 3. 优化措施

（1）xLLM框架通用优化能力：

![deepseek_model_optimization_summary](images/deepseek_model_optimization_summary.png)

（2）基于模型的定制优化：a）Qwen3-30B-MoE：
算子侧：swiglu算子高性能版本替换、GroupMatmul算子二次开发优化、Cast算子消除、替换MoEInitRoutingV3融合算子
框架侧：使能MTP Eagle3特性

b）DeepSeek-v3.2：
算子侧：GroupMatmul算子使能NZ格式、o_proj替换爱因斯坦乘融合算子、替换gmmSwigluQuant和MoeInitRoutingV3融合算子
框架侧：使能aclgraph特性、使能MTP（K=3）

c）ChatGLM 4.6/4.7:
算子侧：PageAttention支持FlashDecoding
框架侧：使能aclgraph特性和ChunkPrefill特性

![chatglm_model_optimization_summary](images/chatglm_model_optimization_summary.png)

## 4. 模型优化效果

| 模型 | 目标(E2E/TPS) | 优化前 | 优化后 | 提升百分比 |
| --- | --- | --- | --- | --- |
| Qwen3-30B-MOE | 4s | 5.1s | 2.75s | 85.45% |
| DeepSeek-3.2 | 55 | 27 | 55.2 | 103.70% |
| GLM-4.5/4.6 | 60 | 51 | 81 | 58.82% |

## 5. 总结

xLLM开源框架支撑了大模型业务在昇腾设备的快速部署优化，显著提升吞吐表现、降低响应时延和资源开销，为xLLM框架优秀的大模型加速能力提供了有力证明；同时xLLM框架也在快速的迭代演进，开发者们可以通过开源仓获取xLLM最新的技术特性。

xLLM开源仓：https://github.com/jd-opensource/xllm

xLLM更多介绍：https://oxygen.jd.com/sdetail/xLLM

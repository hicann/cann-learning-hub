# PyPTO 中高级算子实践（合并版）

本目录把 `02_intermediate` 和 `03_advanced` 合并为一条连续学习线，延续 `03_beginner_operator_practice` 的 notebook 风格：先讲概念，再给最小可运行示例，最后用 PyTorch 做验证。

## 章节安排

| Notebook | 说明 |
| --- | --- |
| `04.01_chapter_intro.ipynb` | 合并版总览、学习路线、examples 对照 |
| `04.02_operator_composition_and_softmax.ipynb` | 自定义激活函数、Softmax、相关 reduction |
| `04.03_normalization_and_ffn.ipynb` | LayerNorm、RMSNorm、FFN |
| `04.04_dynamic_shape_and_controlflow.ipynb` | dynamic shape、loop、condition |
| `04.05_attention_and_transformer.ipynb` | attention、QKV、Transformer block |
| `04.06_system_and_acceleration.ipynb` | cost model、ACLGraph、运行与优化视角 |

## 对应 examples

| examples 目录 | 将进入的合并章节 |
| --- | --- |
| `pypto_clean/examples/02_intermediate/operators/activation` | `04.02_operator_composition_and_softmax.ipynb` |
| `pypto_clean/examples/02_intermediate/operators/softmax` | `04.02_operator_composition_and_softmax.ipynb` |
| `pypto_clean/examples/02_intermediate/basic_nn/layer_normalization` | `04.03_normalization_and_ffn.ipynb` |
| `pypto_clean/examples/02_intermediate/basic_nn/ffn` | `04.03_normalization_and_ffn.ipynb` |
| `pypto_clean/examples/02_intermediate/controlflow` | `04.04_dynamic_shape_and_controlflow.ipynb` |
| `pypto_clean/examples/03_advanced/advanced_nn/attention` | `04.05_attention_and_transformer.ipynb` |
| `pypto_clean/examples/03_advanced/patterns/function` | `04.05_attention_and_transformer.ipynb` |
| `pypto_clean/examples/03_advanced/cost_model` | `04.06_system_and_acceleration.ipynb` |
| `pypto_clean/examples/03_advanced/aclgraph` | `04.06_system_and_acceleration.ipynb` |

## 写作约定

- 保持和初级教程一致的节奏：先概念，再代码，再验证。
- 每个小节围绕一个明确主题展开，不把工程脚手架放在正文中心。
- 代码示例优先复用 examples 中已经验证过的 PyPTO 写法，再逐步抽象出可讲解的规律。
- 验证部分继续采用 PyTorch reference 对照。


# 第 11 章示例与优化记录

补同分选块与原始位置 causal 手算；提供稳定选择→第 10 章注意力的完整阶段参考、空可见行 Golden、分层验收与 ratio 计时入口。保留 0/1 跳过和同 Kernel 发布为增量练习。

源码依据（相对 PyPTO 仓，基线 eb60d7b）：

- `python/pypto_pro/language/_simt_api.py`
- `python/pypto_pro/language/_api.py`
- `python/tests/st/pypto_pro/frontend/fa/test_flex_attention_prefill.py`

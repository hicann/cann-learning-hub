# 第 10 章示例与优化记录

补两块 online 状态手算、128 块完整物理预算和四事件闭环；提供支持 H=64/128、动态 B/S、causal/尾块的完整串行 GM 中转 SDPA 及验收入口。

源码依据（相对 PyPTO 仓，基线 eb60d7b）：

- `python/pypto_pro/language/_api.py`
- `python/pypto_pro/language/parser/_type_resolver.py`
- `python/tests/st/pypto_pro/frontend/tile_vector/test_layernorm.py`
- `python/tests/st/pypto_pro/frontend/fa/test_fa_perf_tkv_preload_dn_vf_tail_tilegroup.py`
- `docs/zh/guide/programming_guide/pro/development/cube_computation.md`

扩展尾块通过 Host 补零到 128 整块，再按原始 S 做 mask 与输出裁剪；这部分准备成本不计入预热后的 Kernel 闭包时间，应另记端到端时间。

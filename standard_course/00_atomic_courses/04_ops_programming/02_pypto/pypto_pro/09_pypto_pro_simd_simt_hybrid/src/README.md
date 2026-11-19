# 第 9 章示例与优化记录

补完整 UB SIMD/SIMT 协同、变长筛选与 SIMD 后处理阶段基线、稳定 Top8；验收包含空/满输出、无序来源索引和同分。明确单 Kernel GM 发布优化仍是后续任务。

源码依据（相对 PyPTO 仓，基线 eb60d7b）：

- `python/pypto_pro/language/_simt_api.py`
- `python/tests/st/pypto_pro/frontend/tile_vector/test_layernorm.py`
- `docs/zh/guide/programming_guide/pro/development/simt_computation.md`

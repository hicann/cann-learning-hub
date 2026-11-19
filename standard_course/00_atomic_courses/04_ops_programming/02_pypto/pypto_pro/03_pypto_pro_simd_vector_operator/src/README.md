# 第 3 章示例与优化记录

新增完整 Tile/VF RMSNorm，明确统计量双视图、四片段归约和 lane 0 广播；补三类完整实践实现及四组数值边界用例。

源码依据（相对 PyPTO 仓，基线 eb60d7b）：

- `python/tests/st/pypto_pro/frontend/tile_vector/test_layernorm.py`
- `python/tests/st/pypto_pro/frontend/vf_api/test_layernorm_tile_group_vf.py`
- `python/pypto_pro/language/_vf_api.py`

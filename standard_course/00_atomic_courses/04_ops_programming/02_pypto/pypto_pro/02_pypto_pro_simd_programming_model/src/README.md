# 第 2 章示例与优化记录

补充动态维覆盖、pitched 视图地址、缓冲区间和四类跨轮依赖；正文增加完整预取缩放 Kernel 与校验，实践提供三个独立实现。

源码依据（相对 PyPTO 仓，基线 eb60d7b）：

- `python/pypto_pro/language/_api.py`
- `docs/zh/guide/programming_guide/pro/development/tile_creation_and_operations.md`
- `python/pypto_pro/language/parser/_buffer_parser.py`

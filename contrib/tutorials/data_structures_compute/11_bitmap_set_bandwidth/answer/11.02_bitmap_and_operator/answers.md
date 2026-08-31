# 11.02 练习参考

- `DenseAnd` 的物理流量为 `3 * N * AlignUp(U, 32)` Byte。
- `BitmapAnd` 的物理流量为 `3 * N * AlignUp(CeilDiv(U, 32), 8) * 4` Byte。
- 短行会被 32 Byte 行填充和 Kernel 启动开销主导，所以理论 8 倍压缩不能直接写成 8 倍加速。
- 本实验使用单缓冲，三条队列共占 `3 * 8192 = 24576` Byte；这使归因重点保持在物理表示和 GM 流量。

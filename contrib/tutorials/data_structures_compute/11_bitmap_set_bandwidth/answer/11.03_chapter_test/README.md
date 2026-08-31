# 11.03 参考答案

参考实现位于 `student_compute.h`，完成三个彼此独立的关键点：

1. 使用 `blockFormer * blockIdx` 计算连续且互不重叠的核间偏移；
2. 只在最后一个 Block 选择 `blockTail`，普通 Block 使用 `blockFormer`；
3. 使用 `ComputeSetAnd(z, x, y, len)` 完成逐物理单位集合交。该辅助函数把 Dense 的 `uint8` 和 Bitmap 的 `uint32` LocalTensor 都重解释为 `uint16` 后执行逐 bit AND，以使用课程 Atlas A2/CANN 9.0 正式支持的基础 API 路径；重解释不会改变 GM 布局或物理流量。

答案不改变 Host 的位图打包规则。`U=31/32/33` 的末字高位和每行填充由规范化输入保持为 0。

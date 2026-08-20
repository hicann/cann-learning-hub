# 03.02 课后练习参考答案

## 1. depth 与 num

<code>TQue&lt;position, depth&gt;</code> 的第二个模板参数描述队列抽象允许连续入队的深度；<code>InitBuffer(queue, num, size)</code> 中的 <code>num</code> 描述为该队列准备的物理 Buffer 数量。本实验的调度会先取出当前 Tile，再预取下一 Tile，任一时刻队列中至多保留一个已入队 Tensor，因此 <code>depth=1</code> 足够；将 <code>num</code> 从 1 改为 2 才会分配双缓冲槽位。

## 2. Tensor 生命周期

输入 Tensor 的顺序是：<code>AllocTensor → DataCopy → EnQue → DeQue → Compute → FreeTensor</code>。输出 Tensor 的顺序是：<code>AllocTensor → Compute → EnQue → DeQue → DataCopy → FreeTensor</code>。

<code>EnQue/DeQue</code> 同时承担队列所有权转移与流水同步。如果搬入后不入队，Vector 阶段可能在 MTE2 尚未完成时读取数据；如果消费后不释放，后续 Tile 将无法取得可复用槽位。

## 3. Local Memory 预算

本实验有两个输入队列和一个输出队列，共三个 TQue。每核队列有效载荷为：

    queueBytes = 3 × bufferNum × tileLength × sizeof(float)

默认 <code>tileLength=256</code>、<code>sizeof(float)=4</code>：

- 单缓冲：<code>3 × 1 × 256 × 4 = 3072 Byte</code>；
- 双缓冲：<code>3 × 2 × 256 × 4 = 6144 Byte</code>。

该数值是每个逻辑 Block 的三个队列槽位有效载荷，不是整个输入输出张量的 GM 字节数，也不需要再乘 <code>blockDim</code> 才能判断单核 UB 是否超限。

## 4. tileCount 与 bufferNum

<code>tileCount</code> 是每个 Block 实际覆盖的 GM Tile 数，决定数据区间和循环次数；<code>bufferNum</code> 只决定每个队列有多少物理槽位。把单缓冲改成双缓冲后，默认配置仍处理 8 个 Tile，不能把循环次数改成 <code>tileCount / bufferNum</code>。

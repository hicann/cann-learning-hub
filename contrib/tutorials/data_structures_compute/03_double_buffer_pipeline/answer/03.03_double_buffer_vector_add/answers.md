# 03.03 课后实践参考答案

## 1. 资源计算函数

    def queue_payload_bytes(tile_length, buffer_num, tensor_count=3, dtype_bytes=4):
        if tile_length <= 0 or buffer_num not in (1, 2):
            raise ValueError('invalid pipeline configuration')
        return tensor_count * buffer_num * tile_length * dtype_bytes

默认配置应得到单缓冲 3072 Byte、双缓冲 6144 Byte，二者比值为 2。

## 2. 预取与排空顺序

双缓冲先执行 <code>CopyIn(0)</code>。每轮先从输入队列取出当前 Tile，再预取 <code>tile+1</code>，从第二轮开始写回 <code>tile-1</code>，最后计算当前 Tile。循环结束后还要写回最后一个 Tile。

先取出当前输入很重要：本实验的 queue depth 固定为 1；若在 DeQue 前连续入队 Tile 0 和 Tile 1，会超过队列深度。当前 LocalTensor 仍占用第一个物理槽位，预取下一 Tile 会使用第二个槽位。

## 3. 边界分析

- <code>tileCount=1</code>：只执行一次预装、一次计算和一次最终写回，没有可重叠的下一 Tile；
- <code>tileCount=2</code>：第 0 轮预取 Tile 1，第 1 轮写回 Tile 0，循环后写回 Tile 1；
- 任意正整数 Tile 数：<code>tile+1&lt;tileCount</code> 防止末尾越界，<code>tile&gt;0</code> 防止无符号 <code>tile-1</code> 下溢。

## 4. 性能结论

正确性、队列配置和资源公式可以作为硬性检查；单次 Host 计时得到的加速比不应作为验收门槛。<code>avg_kernel_us</code> 的口径是多次 Kernel launch 加一次 stream synchronize 后取平均，包含摊薄后的 Host 启动与同步开销，不等同于 msprof 的设备侧 Task Duration。应在相同 Shape、相同 Tile 切分和相同计时口径下多次测量，再把加速比作为观察结果。

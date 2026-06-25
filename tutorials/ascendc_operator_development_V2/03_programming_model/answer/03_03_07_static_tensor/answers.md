# 03.03.07 基于静态 Tensor 框架的编程课后练习答案

## 题目 1

答案：B

说明：静态 Tensor 写法通过 `LocalMemAllocator<AscendC::Hardware::UB>` 等 allocator 在指定硬件存储上分配 `LocalTensor`。

## 题目 2

答案：`GlobalTensor`

说明：`GlobalTensor` 用于绑定 GM 输入输出地址，提供 Device 侧访问全局内存的视角。

## 题目 3

答案：A

说明：`MTE2_V` 表示 MTE2 搬入完成后，Vector 才能读取对应 UB buffer。

## 题目 4

答案：`PipeBarrier<PIPE_V>()`

说明：同一 Vector 流水内部，后一条 Vector 指令依赖前一条 Vector 指令结果时，需要使用 `PipeBarrier<PIPE_V>()` 保证顺序。

## 题目 5

答案：C

说明：RegBase 路径需要先由 kernel 侧通过 `asc_vf_call` 进入 VF 函数，再在 VF 函数中执行 `LoadAlign → RegBase 计算 → StoreAlign`。

## 题目 6

答案：B

说明：当尾块或地址不满足普通连续搬运对齐要求时，应使用 `DataCopyPad` 或非对齐搬运接口处理，避免直接按完整 tile 读取无效区域。

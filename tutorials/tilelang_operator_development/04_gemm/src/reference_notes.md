# 本章API与预算依据

锁定TileLang fork commit：`130962668c3cc79b3a41c92b57c81923cfcc669f`。

- `src/ascend/transform/auto_schedule.cc:109–121`：GetL1MemoryLimit返回512*1024，GetL0CMemoryLimit返回256*1024。它们是本后端的预算，不是跨型号通用容量。
- `tilelang/ascend/language/allocate.py:12,41`：alloc_l1分配shared.l1；alloc_l0c分配shared.l0c，layout默认True。
- `tilelang/ascend/op/gemm/gemm_mad.py:164`：L1 GEMM断言trans_A=False且trans_B=True。
- `tilelang/language/loop.py`的Persistent(domain,wave_size,index,...)构造遍历；本章采用一维 domain：4.4 按 idx=bx+8*w 遍历；4.5 列优先按 ni=bx//2+18*w 遍历。
- `tilelang/language/allocate.py`的alloc_var(dtype,...)产生标量Buffer；`T.macro`在生成IR时展开。

GEMM原版对照见[example_gemm.py](original/example_gemm.py)。原版含本章不展开的参数与路径，教学与验证使用本章简化代码。

另一个已锁定素材为TileKernels-Nightly commit `86bc72e0b519d58ab407a157182fd77df8b2395f`：`tile_kernels/transpose/batched_transpose_asc.py:58`确实在Vector转置算子中调用`T.Persistent`。这说明原语不专属于GEMM；本章没有把转置算子作为额外教学实验。

本次设备查询给出36个Cube核、72个Vector核。正文利用率采用配置峰值估算：8核为117.9648 TFLOPS，36核为530.8416 TFLOPS；有效带宽按输入输出的最小逻辑数据量计算，不等同于实际HBM流量。历史性能表仅作教学示例，当前性能需用msprof重测。

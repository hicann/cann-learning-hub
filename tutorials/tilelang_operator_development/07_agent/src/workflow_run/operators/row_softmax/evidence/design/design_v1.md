# row_softmax：A5 TileLang 算子设计

- 目标平台：Ascend 950 / A5；实机 Ascend950DT_9592，DAV_3510，SocVersion 枚举 4（ASCEND950）。物理设备 0，唯一可用 NPU；torch 属性及 ACL 核数查询均为 AIC 36、AIV 72。PlatformAscendC 以 aclrtGetSocName() 返回的型号初始化，UB 为每 AIV 253952 字节。来源：evidence/environment/runtime.json、platform.log 及对应探测源码。
- 运行环境：使用 TileLang AscendC + SimdVF 后端，并在隔离环境中完成编译、测试和性能采集。
- 编译配置：target=ascend，codegen=AscendC，execution backend=tvm_ffi（显式指定）；CANN 9.2.0-beta.1 / Bisheng，GCC 11 头文件，向量 SimdVF。每次运行 source setup_env.sh；本轮缓存目录覆盖到 operators/row_softmax/evidence/runtime/，不修改安装。

## 1. 算子定义与支持范围

### 1.1 计算语义与接口规格

依据 course_inputs/task.md，逐行计算 m[i]=max_j X[i,j]，e[i,j]=exp(X[i,j]-m[i])，s[i]=sum_j e[i,j]，Y[i,j]=e[i,j]/s[i]。axis 固定为 1，无 epsilon、缩放、mask 或隐式裁剪。输出是一个新分配的张量，输入逐位不变。仅前向，无梯度交付要求、随机算法或原地输出。

| 参数 | shape / dtype / stride 或 layout | 含义与约束 |
|---|---|---|
| X / run(x) | (M,N)，float32，Ascend NPU，strided 且连续，标准行主序 | 4<=M<=256 且 M%4=0；128<=N<=2048 且 N%64=0；所有元素有限且 abs(x)<=80 |
| Y / 返回值 | shape、dtype、device 与 X 相同，新分配且连续 | 不与 X 别名，不修改 X，完整覆盖每行 |
| build(rows,cols) | 编译期 Python 整数 M,N | 检查同一 shape 域；输出已编译 callable，底层参数为 X,Y |

公开 run 对 dtype 不符抛 TypeError；非张量、rank、shape、layout、device、非有限值及范围错误抛 ValueError，消息分别含 tensor、rank、rows/cols、contiguous/strided、NPU、finite、80 等原因。多重违规只保证按入口检查顺序报告首个原因。合法连续视图允许非零 storage_offset。数值检查可调用 torch 比较/归约，Softmax 本身必须在本地 TileLang kernel 完成。

### 1.2 指定 case

本阶段生成和指定精度验证仅使用下表九项。用户明确约定的完整合法域保持为第 1.1 节；扩展测试由后续 ST 承担，不将表外合法输入改为域外。此表述遵循任务书的显式支持域。

| case ID | 各输入 shape / dtype | 属性与布局 | 预期输出 shape / dtype |
|---|---|---|---|
| m4_n128 | X=(4,128), float32 | axis=1，连续，device=0 | (4,128), float32 |
| m4_n2048 | X=(4,2048), float32 | 同上 | (4,2048), float32 |
| m64_n128 | X=(64,128), float32 | 同上 | (64,128), float32 |
| m64_n512 | X=(64,512), float32 | 同上 | (64,512), float32 |
| m64_n2048 | X=(64,2048), float32 | 同上 | (64,2048), float32 |
| m256_n128 | X=(256,128), float32 | 同上 | (256,128), float32 |
| m256_n512 | X=(256,512), float32 | 同上 | (256,512), float32 |
| m256_n2048 | X=(256,2048), float32 | 同上 | (256,2048), float32 |
| m68_n192 | X=(68,192), float32 | 同上 | (68,192), float32 |

M,N 编译期固定，输入元素为运行时数据；无 workspace 初始化要求。N*4 为 256 字节倍数，DMA 行长度和每行偏移满足 32 字节对齐；不对张量增加复制或 padding 要求。

## 2. 算法与实现架构

### 2.1 算法、执行模式与 kernel 划分

推荐一个纯 AIV kernel，每个逻辑 block 处理一整行，沿连续列轴归约。行完整驻留 UB；四个 SimdVF 阶段依次完成最大值、指数、指数和、除法。在本仓独立实现编译封装和公开约束。AIC 不参与，无跨行状态和跨 kernel 合并。

| 计算阶段 | 子公式与依赖 | kernel / 执行单元 | 中间结果 |
|---|---|---|---|
| 行最大值 | m=max(X)，依赖搬入 | row_softmax / AIV SimdVF | m UB |
| 指数 | e=exp(x-m)，依赖 x,m | 同 kernel / AIV SimdVF | e UB |
| 分母 | s=sum(e)，依赖 e | 同 kernel / AIV SimdVF | s UB |
| 归一化 | y=e/s，依赖 e,s | 同 kernel / AIV SimdVF | y UB，随后写 Y |

### 2.2 数值精度与初始化

输入、UB、中间值、向量运算、归约和输出均为 float32。reduce_max(...,dim=1,clear=True) 的实际 SimdVF lowering 初值为 -3.402823e38，覆盖 [-80,80]；sum 初值为 0，每行重新初始化。当前 lowering 每 64 个 fp32 元素做寄存器归约，再顺序合并 N/64 个部分结果；不存在跨行累加。指数输入在 [-160,0]，分母至少 1，不添加 epsilon。极小指数下溢允许为 0，但仍必须满足用户逐元素与行和标准。不使用低精度 cast、fast math 额外开关或 CPU 计算输出。

### 2.3 调用接口与工程集成

文件为 operators/row_softmax/row_softmax.py 和同目录 test_row_softmax.py。build(rows,cols) 验证 shape，生成 T.prim_func 并调用 tilelang.compile(...,target='ascend',execution_backend='tvm_ffi')，用 lru_cache 缓存固定 shape。run(x) 先检查 metadata、有限性和范围，然后在 x.device 上新分配连续 Y，执行 build(M,N)(X,Y)，返回 Y。核心计算不使用 torch.softmax。所有验证和分配均在 kernel 外，底层 callable 便于后续单独计量 kernel；此分离不省略公开入口拒绝语义。无额外 GM workspace、host 重排或 dtype 转换。

## 3. Tiling、数据流与资源规划

### 3.1 分块与任务映射

| 适用 case ID | 初始 tile / 分派条件 | task/core 映射 | 流水级数与 buffer 版本数 |
|---|---|---|---|
| 全部九项 | block_rows=1，block_cols=N | T.Kernel(M)，task bx 对应行 bx，输出 [bx:bx+1,0:N] | 无任务循环和显式流水，全部单版本 |

总 task 数=M。运行时将 M 个逻辑 block 调度到 72 个 AIV；最多 min(M,72) 个 block 同时执行。M=4/64/68 各只有一组任务；M=256 需要多个调度波次，最后一波 40 个 block。block id 是逻辑 task，不视为物理核 id。无需硬编码物理核数或手动循环。每行唯一写者，未跨任务读写。初始方案不因小任务数增加流水或空任务。

### 3.2 Buffer 布局与容量计算

| Buffer | 物理 shape / dtype | 存储层级及所属核 | 版本数 / 生命周期 | 字节占用 |
|---|---|---|---|---|
| x | (1,N), float32 | 每 AIV UB | 1，搬入至指数阶段结束 | 4N |
| e | (1,N), float32 | 每 AIV UB | 1，指数至除法结束 | 4N |
| y | (1,N), float32 | 每 AIV UB | 1，除法至搬出结束 | 4N |
| m | (1,), float32，分配按32B对齐 | 每 AIV UB | 1，max 至指数结束 | 32（逻辑4B） |
| s | (1,), float32，分配按32B对齐 | 每 AIV UB | 1，sum 至除法结束 | 32（逻辑4B） |
| 隐式寄存器临时值 | 64-lane fp32 向量及 mask | AIV VF 寄存器 | 每 VF 内，lowering 生成 | 不计入显式 UB；为潜在编译临时 UB 另留余量 |

预算不依赖编译器复用 x/e/y。单版本全部同时计入：12N+64，N=128/192/512/2048 分别为 1600/2368/6208/24640 字节。共享分配 pass 按32B对齐；标量 store 使用 ONEPT_B32，读取广播 BRC_B32。编译器可能复用生命周期不重叠的空间，但可行性不依赖该优化。

| 存储层级及所属核 | 峰值占用计算式与结果 | 可用容量及来源 | 可行性结论与成立条件 |
|---|---|---|---|
| 每 AIV UB | 最坏 12*2048+64=24640B，加 16384B 临时/对齐预留=41024B | 253952B，实机型号对应 PlatformAscendC.GetCoreMemSize(UB)，evidence/environment/platform.log | 预留后约16.2%，充足；生成源码须核对实际分配没有超出此预算 |

纯 Vector 不分配 L1/L0A/B/C；不以整组物理 UB 容量替代每 AIV 可用值。寄存器归约 helper 不额外申请行级 GM/UB workspace。

### 3.3 数据搬运与同步

| 数据流阶段 | 源 buffer → 目标 buffer | 执行单元 / 关键 API | 数据依赖与复用条件 |
|---|---|---|---|
| 搬入 | X[bx:bx+1,:] → x | MTE2 / T.copy | x 写完成后 max/exp 才能读取 |
| 最大值 | x → m | AIV / T.SimdVF + T.reduce_max | m 写完成后广播；x 在 exp 后不再使用 |
| 指数 | x,m → e | AIV / T.Parallel + T.exp | e 完整写入后 sum 与除法才能读取 |
| 分母 | e → s | AIV / T.reduce_sum | s 写完成后归一化广播 |
| 归一化 | e,s → y | AIV / T.Parallel | y 完整写入后 MTE3 才能读取 |
| 搬出 | y → Y[bx:bx+1,:] | MTE3 / T.copy | 写完成才结束任务或复用该输出空间 |

保留默认 AutoSchedule 和 InsertSync。pipeline.py 在 LowerSimdVF 之前依 buffer 区域读写建依赖并插入 MTE2→VF、各 VF 生产消费关系和 VF→MTE3 同步；MergeUBAllocations 在生命周期与32B对齐约束下合并分配。核内事件负责 DMA 与计算可见性，无跨核依赖、原子操作或手动 set/wait。验证时同步设备后再读取 CPU 结果。

### 3.4 边界处理与计算骨架

所有指定 M 都整除 block_rows=1；所有 N 都是 fp32 VReg 宽度64的倍数。N=192 为3个向量块，不按二次幂补齐。任务调度波次不足72时由运行时只执行合法 block id；逻辑行和列均无 padding。不存在输出裁剪、读取无效列或多任务归并。

伪代码：构建 M 个 block → 分配 x/e/y/m/s → 拷入一行 → reduce_max(dim=1,clear=True) → exp(x-m) → reduce_sum(dim=1,clear=True) → e/s → 拷出一行。四个计算阶段各自位于 T.SimdVF，逐元素阶段用 T.Parallel(1,N)。

## 4. API 依据与性能分析

### 4.1 关键 API 与参考实现

| 设计决策 / API | 用法与限制 | 来源文件 / 函数或类 |
|---|---|---|
| 一行一任务计算结构 | 完整行归约，四段 VF，添加公开校验和编译缓存 | 本任务的逐行独立计算约定与容量分析 |
| T.reduce_max / T.reduce_sum | 2D UB → 1D UB，dim=1，全 buffer，fp32 归约长度必须整除64 | TL tilelang/language/reduce_op.py；src/ascend/transform/ascend_simdvf_lower_parallel.cc:LowerReduce |
| 归约结果存储与广播 | ONEPT_B32 写单点，BRC_B32 广播；lowering 和 helper 使用课程补丁 | TL src/ascend/transform/ascend_simdvf_lower_parallel.cc；src/tl_templates/ascend/simd_inst.h:vcadd/vcmax |
| T.SimdVF / T.Parallel | 向量执行域；连续列维是寄存器向量维，extent%64=0 | TL tilelang/ascend/language/frame.py:SimdVF；ascend_simdvf_lower_parallel.cc:LowerParallel |
| T.alloc_shared / T.copy | GM↔UB，全行连续 fp32，32B 对齐 | TL tilelang/language/allocate.py:alloc_shared；src/ascend/op/copy.cc |
| 自动同步与 UB 规划 | 保留默认 AutoSchedule、InsertSync；按32B合并分配 | TL tilelang/ascend/pipeline.py |
| 编译与执行后端 | target=ascend 不含 pto key；显式 tvm_ffi | TL tilelang/ascend/target.py:target_is_plain_ascend；execution_backend.py；tilelang/jit/__init__.py:compile |

参考结构与九项 shape 相容；本次不复用旧候选、旧测试或旧结果。课程补丁特别影响 SIMD 标量广播与归约，是本环境依赖，不对共享安装作修改。

### 4.2 性能考虑

初始设计预期瓶颈为小 M 并行度不足、四段 VF 间 UB 读写和较大 N 的多块归约；这是源码推断，不是实测结论。一次 GM 读写共8MN字节，无中间 GM 往返。公开数值拒绝检查包含设备比较和 host 同步，其成本与真实 kernel 时长分别说明。

性能任务以本轮 ST 通过初始实现冻结为不可变 B0，九项全部测量；每项候选/B0<=1.05，九项比值几何均值<=0.95，最终完整 ST 通过。该值是任务接纳政策。首轮性能采集之前按 ops-profiling 锁定同设备、同输入、同计时工具、预热、重复次数、缓存和统计量；本设计不开展性能搜索或承诺加速。进入调优必须按入口取得 ST 后的明确确认。

## 5. 指定 case 的精度验证

### 5.1 验证用例与流程

覆盖第 1.2 节的全部指定 case，不另行扩展用例。

| 指定 case ID | 输入生成方式 | Golden / 比较规则 |
|---|---|---|
| m4_n128、m4_n2048 | CPU torch.Generator，逐 case manual_seed(707)，randn float32，clamp[-80,80]，上传 npu:0 | 下述 CPU double Softmax；全部断言 |
| m64_n128、m64_n512、m64_n2048 | 同上，shape 按 cases.json | 同上 |
| m256_n128、m256_n512、m256_n2048 | 同上，shape 按 cases.json | 同上 |
| m68_n192 | 同上，shape=(68,192) | 同上 |

- Golden：test_row_softmax.py 中独立 CPU reference：torch.softmax(X_cpu.double(),dim=1).float()。reference 不导入或调用 DUT。
- 验证流程：读取完整 cases.json → 固定种子 CPU 生成 → 上传并保存输入原始位模式 → 调用本地 row_softmax.run → NPU synchronize → 复制输出到 CPU → 检查数值、行和、输入不变、shape/dtype/device、连续和存储不别名。打印每项最大绝对误差、按容差归一化最大误差及行和误差并保存日志。

### 5.2 精度标准

| 输出 / 适用 case | 比较指标与定义 | 判定条件及具体阈值 | 标准来源 |
|---|---|---|---|
| Y / 全部九项 | 每个元素 abs(Y-ref) | <=1e-6+1e-4*abs(ref)，逐元素检查，无比例豁免 | course_inputs/task.md |
| Y 行和 / 全部九项 | abs(sum_j Y[i,j]-1)，CPU double 累加输出 float32 | 每行 <=1e-5 | 同上 |
| X / 全部九项 | 调用前后 float32 位模式，CPU view(int32) | 精确相等 | 同上，输入精确不变 |
| Y metadata / 全部九项 | shape、dtype、device、is_contiguous、data_ptr | 与输入规格一致且新分配 | 同上 |

同时要求 Y 全部有限且非负。未改动容差、reference 精度或用户案例。

## 6. 设计约束与实现建议

### 6.1 技术假设与依赖（按需）

| 假设或依赖 | 对方案的影响 | 核实方法或替代方案 |
|---|---|---|
| course_inputs/runtime_lock.json 对应已安装 SIMD 兼容补丁 | 所有 shape 的标量广播和向量归约语义 | runtime.json 中提交、patch 及两处源码哈希均匹配；交付依赖此安装，不修改系统 |
| Ascend950DT_9592 / DAV_3510 运行时与工具链 | 全部 SimdVF 编译和执行 | 同设备实机参数见环境证据；切换平台须重新查证 |

### 6.2 实现建议与可调参数

必须保持完整支持域、九个指定案例、独立 CPU double reference、所有精度/副作用/异常约束、AscendC 和 SimdVF。初始 block_rows=1；未来若调节 block_rows、核数、流水或 buffer 版本，须重新证明任务覆盖、N向量整除和 UB预算，并同步受影响设计及完整测试。build 内只含 kernel 与编译封装；测试、Golden、性能和设备探测不放入生产 kernel 文件。run 内只做约束检查、分配和调用。

### 6.3 修订说明（仅修订时填写）

初始设计。

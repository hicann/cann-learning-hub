# 昇腾950 AIV直驱URMA的SHMEM跨PE通信实践

## 阅读说明

本文面向首次接触 AIV 直驱URMA 的开发者。文中的 SHMEM 特指昇腾上基于 OpenSHMEM 实现的分布式通信库——它沿用 OpenSHMEM 的对称内存与 PE 语义，并以 UDMA 作为 Device 侧 RMA 的后端引擎。

阅读前需要掌握如下基础概念：

- NPU kernel 的基本概念：Host 端 CPU 负责启动，device 端核（如 AI Core）执行。

- 多卡训练中的"通信域"概念：一组协同工作的 NPU 集合，每张卡是一个通信端点。

## 1. 要解决的问题

在 MoE 训练或推理任务的 profile 中，常见现象是：计算 kernel 持续时间较短，通信操作数量较多，阶段之间还存在 Host launch 与同步等待。此类阶段化执行会降低流水连续性，使端到端时延难以进一步优化。

这种现象在 MoE dispatch / combine、KV cache shuffle、tensor parallel alltoall 等场景尤其明显。它们的共同特征不是单次通信规模大，而是通信粒度细、轮次多，并且与 device 侧的数据布局转换、按目标卡重排、归一化等计算强相关。继续在 Host 侧调度通信，通常会遇到两个限制：

1. 启动与同步开销较高：每轮通信都要从 Device 返回 Host，由 Host 发起下一个 kernel。

2. 计算与通信重叠不足：阶段被明显切分，难以让计算覆盖通信延迟。

AIV直驱URMA 的核心思路是：让 Device 侧计算核在 kernel 内直接发起跨卡内存访问，把通信纳入算子逻辑，而不是返回 Host 调度。Host 只负责一次性初始化通信域、对称内存和队列资源；进入 kernel 后，Device 自己按目标卡组织数据、提交远端访问、按需等待或通知。

核心概括：

> AIV直驱URMA = SHMEM 对称内存 + Device 侧 UDMA + 计算和通信同 kernel 编排。

## 2. 术语定义

后文会反复出现下列术语，先统一语义。

| 名词 | 含义 | 本文中的作用 |
| --- | --- | --- |
| Host | CPU 侧的控制程序 | 负责初始化、kernel 启动、资源管理 |
| Device | NPU 侧的执行单元 | 实际执行计算和通信 |
| NPU | AI 加速器设备，例如昇腾系列芯片 | 承载 AI Core、GM 等硬件资源 |
| AI Core | NPU 上的计算核，含向量、标量、矩阵等子单元 | 执行算子代码 |
| AIV | AI Vector，AI Core 内的向量计算子单元 | 执行数据布局转换、按 PE 重排、逐元素计算，也负责发起 UDMA |
| GM | Global Memory | NPU 全局显存，AIV 可访问 |
| UB | Unified Buffer，AI Core 内部的快速缓存 | 本文中 `__ubuf__` 修饰的指针指向此处 |
| PE | Processing Element | SHMEM / RMA 视角下的通信端点，通常对应一张卡或一个通信域内的 rank |
| RMA | Remote Memory Access | 单边远端内存访问，发起方直接读写远端内存，无需对端配合 |
| AMO | Atomic Memory Operation | 远端原子操作，常用于 counter / flag |
| UnifiedBus（灵衢） | 面向超节点的高速互联底座 | 提供"内存可互访"的物理基础 |
| URMA | Unified Remote Memory Access | 软件抽象层，将远端读写表达为标准操作 |
| UDMA | Unified Direct Memory Access | 数据传输引擎，作为 SHMEM RMA / AMO 的底层后端引擎；本文重点讨论其 Device 侧提交WQE能力 |
| jetty | URMA 的通信队列 / 连接抽象，可类比 RDMA QP | 承载 WQE 投递、目标端关联和完成处理 |
| EID | Endpoint Identifier，URMA 端点标识 | 在 URMA 层标识远端通信端点，用于寻址和路由 |
| WQE | Work Queue Entry | 一次 RMA / AMO 的硬件工作请求 |
| SHMEM | 昇腾上基于 OpenSHMEM 实现的分布式通信库 | 提供 PE 编号、对称内存、同步原语；UDMA 是其 device 侧 RMA 后端 |

AIV 直驱 URMA 的通信路径如图 1 所示：

![aiv_direct_urma_communication_path](images/aiv_direct_urma_communication_path.png)

本文重点讨论图 1 所示的通信发起链路：AIV 如何在 Device 侧构造 WQE，并通过 RMA 将数据写入其他 PE 的对称内存。

## 3. 软件栈：从 UnifiedBus 到 AIV

![unifiedbus_udma_shmem_software_stack](images/unifiedbus_udma_shmem_software_stack.png)

AIV直驱URMA 依赖一组分层的软件和硬件通信栈，由下到上为 UnifiedBus → UDMA → SHMEM → AIV 算子代码。

### 3.1 UnifiedBus：提供"内存可互访"的底座

UnifiedBus（UB，灵衢）面向超节点互联，目标是让多个 PE 之间能以更接近内存访问的方式通信。对上层来说，关键点不是它的链路细节，而是它提供了一个可以承载 RMA / AMO 的互联底座。开发者不直接处理物理链路，而是通过 URMA、SHMEM、UDMA 等抽象完成远端读写。

### 3.2 URMA：将远端访问表示为标准操作

URMA（Unified Remote Memory Access）负责将"写远端一段内存"描述为硬件可执行的操作。它在 SHMEM 的 PE 语义和底层互联的端点寻址之间提供转换与投递能力。两个关键概念：

- EID 是 URMA 层的 endpoint identifier，用于标识可达的远端通信端点。SHMEM 中的 PE 编号需要由运行时映射到对应的 URMA endpoint。

- jetty 是 URMA 的通信队列或连接抽象，可类比 RDMA 中的 QP。发起端通过 jetty 向目标端投递 WQE，硬件再根据 WQE 中的地址、长度、操作类型和目标端信息执行 RMA / AMO。

一次典型 RMA 操作需要描述：本端地址、远端地址、数据长度、目标 EID / 目标 PE、承载投递的 jetty、操作类型（put / get / atomic）。这些信息最终形成 WQE，再投递到硬件队列执行。

### 3.3 SHMEM：让远端地址可由"同名指针 + PE"指代

SHMEM 使用 PGAS（Partitioned Global Address Space）模型。所有 PE 同步调用`aclshmem_malloc(size)` 分配相同大小的对象，返回的指针就是对称内存指针。仓内 Device 代码把这种指针变量普遍命名为 `gva`（global virtual address）；本文后续统一用"对称指针"或"对称内存"指代它，以避免读者把它当作某个具体对象的名字。

对称语义来自 SHMEM 的 heap 布局约定（见 `docs/principles.md`）：

- 每个 rank 持有自己的 `heap_base[i]`，`aclshmem_malloc` 从该 heap 内分配。

- 所有 rank 同步分配相同 size 时，相邻 rank 的 heap 在虚拟地址空间等距排布：`heap_base[i+1] − heap_base[i] = heap_size`。

- 因此同一对象在 PE[i]上的地址 = `heap_base[i] + offset`，不同 PE 上的虚拟地址不相同，但相互之间相差一个固定步长 (`heap_size`) 的整数倍。

![symmetric_heap_address_mapping](images/symmetric_heap_address_mapping.png)

`aclshmem_ptr(p, pe)` 的逻辑就是这条公式：

```
remote_p = heap_base[pe] + (p − heap_base[my_pe])
        = p + (pe − my_pe) × heap_size

```

调用方只需提供"本端对称指针 + 目标 `PE`"，UDMA 等 RMA 接口会按这个固定步长定位远端 PE 上的同一对象，不需要业务显式调用 `aclshmem_ptr`。`aclshmem_ptr` 一般用在 Device 代码需要直接拿到一个远端 PE 上的对称指针时（例如 `aclshmem_signal_wait_until` 要等远端 signal）。

### 3.4 UDMA：Device 侧 RMA 提交

UDMA（Unified Direct Memory Access）在 SHMEM 体系中是数据传输引擎和 Device 侧数据面接口。它面向跨 PE 对称内存访问，将 put、get、AMO、put-with-signal 等操作提交到底层传输路径执行。

它和传统路径的区别只有一个：提交点的位置。传统路径中，Host 调用通信 API，Device 只负责计算；UDMA 让 AIV 在 kernel 内直接提交 RMA / AMO，使通信成为算子数据面的一部分。

> 这里的"直驱"强调的是 AIV 发起远程访问请求，而不是 AIV 直接管理物理链路、连接建立或通信域生命周期。

SHMEM 对外提供 `aclshmemx_udma_*` 系列 Device API（前缀中的 `x` 表示扩展接口），常用的有：

- `aclshmemx_udma_put_nbi(dst, src, buf, elem_size, pe)`：非阻塞 put

- `aclshmemx_udma_get_nbi(dst, src, buf, elem_size, pe)`：非阻塞 get

- `aclshmemx_udma_put_signal_nbi(dst, src, elem_size, sig_addr, signal, pe)`：put 完成后写远端 signal

- `aclshmemx_udma_quiet(pe)`：等待本端发往该 PE 的已提交 UDMA 操作完成（带 PE 参数，区别于通用 `aclshmem_quiet()`）

- `aclshmemx_udma_atomic_add(dst, value, pe)`：远端原子加

`nbi` = non-blocking immediate`：接口调用后立即返回，不等待远端完成；后续完成语义需要通过 `quiet`、`signal`或`AMO counter` 显式表达。换言之，UDMA 提供的是 Device 侧提交和完成等待能力，不替代 SHMEM 的对称内存、Team、Barrier 等编程模型。

## 4. UDMA 基本原理

### 4.1 UDMA 与 DMA / RDMA 的区别

| 机制 | 发起位置 | 访问范围 | 典型语义 |
| --- | --- | --- | --- |
| DMA | 设备或 DMA engine | 本设备内存或主机内存 | 本地搬运，常用于片内或主机与设备之间的数据移动 |
| RDMA | Host 侧通信库 / NIC 队列 | 远端节点内存 | Host 准备 QP / MR / WR，网卡执行远端读写 |
| UDMA | Device kernel 内部 | 远端 PE 的对称内存 | AIV 在 kernel 内构造远端访问请求，由 URMA / UB 执行 |

UDMA 的关键点不是 DMA 搬运能力本身，而是远端访问提交点在 Device 侧。

### 4.2 UDMA 的资源模型

一次 UDMA 操作通常依赖以下资源：

| 资源 | 来源 | 作用 |
| --- | --- | --- |
| 对称内存 | SHMEM Host 初始化阶段分配 | 提供可推导的本端 / 远端地址关系 |
| PE / team 信息 | SHMEM runtime | 将业务目标 PE 映射到通信端点 |
| EID | URMA | 标识远端 endpoint，用于硬件寻址 |
| jetty | URMA | 承载 WQE 投递和完成处理 |
| WQE | AIV / UDMA 接口构造 | 描述本端地址、远端地址、长度、操作类型和目标端 |
| UDMA work buffer | kernel 入参或 Device 侧临时空间 | 接口在 UB 中构造 / 提交请求时使用的工作区 |

Host 在 kernel 启动前完成对称内存、通信域、jetty / EID 映射等准备工作，并将 device 侧所需状态下发给 kernel。进入 kernel 后，AIV 不再重新建立连接，而是复用这些已初始化资源提交 UDMA 操作。

参考 UMDK 的资源模型，Host 初始化可以概括为四类工作：

1. 端点发现：确定本设备可用的 EID，并建立 SHMEM PE 到 URMA endpoint 的映射关系。

2. 队列准备：创建本端用于提交和完成处理的队列资源，并建立与远端 jetty 的关联。

3. 内存注册：将本端缓冲区注册为可被远端访问的内存段，记录虚拟地址、长度、访问权限和 token 等属性。

4. 远端资源交换：通过控制通道交换远端 endpoint、内存段和 jetty 描述，使本端能够构造面向指定目标 PE 的远端访问请求。

完成这些准备后，UDMA 不再需要在 kernel 内建立连接或重新注册内存。

### 4.3 一次 UDMA put 的执行路径

以 `put_nbi` 为例，Device 侧执行路径可以拆成五步：

1. AIV 在本地 GM 上准备好 `src`（任意本地指针即可，不要求是对称对象）。

2. AIV 决定 `dst`：传入本地的对称指针 + 目标 `pe`；RMA 接口按 `heap_base[pe] + offset` 公式定位远端 PE 上的同一对象，写入到对应位置。

3. UDMA 接口把 `dst`、`src`、`buf`、`elem_size`、`pe` 等参数组织为 WQE。

4. WQE 通过对应 jetty 投递到 URMA / UB 硬件路径。

5. 硬件完成远端写入；AIV 后续通过 `aclshmemx_udma_quiet(pe)`、signal 或 AMO counter 管理完成语义。

> `put_nbi` 返回表示请求已经发起，不表示远端写入已经完成；远端数据能否被消费，由完成语义单独表达。

类比传统 RDMA：软件准备 WQE → 通知硬件有新请求 → 硬件解析 WQE 中的地址、长度、权限和目标端 → 执行 DMA / 网络发送 → 完成后产生完成事件。UDMA 的抽象类似，但发起者从 Host 用户态程序变成了 AIV kernel；完成事件也不一定表现为 Host 侧 CQE，而是通过 `quiet(dst_pe)`、signal 或 AMO counter 在 device 侧表达。

### 4.4 控制面与数据面分离

UDMA 路径可以分为：

- 控制面（Host 负责）：SHMEM 初始化、对称内存分配、通信域建立、jetty / EID 映射、device state 下发。

- 数据面（AIV 负责）：本地 src 组织、对称指针 + `pe` 的传递、WQE 提交、per-PE `udma_quiet`、signal / AMO 通知、后续消费。

这种分离是 UDMA 能够降低 Host round trip 的原因。Host 不再参与每轮 tile 或每个目标 PE 的通信提交，AIV 在单个 kernel 内完成"计算、提交通信、等待局部完成、继续计算"的闭环。

### 4.5 UDMA 适合的访问形态

UDMA 更适合连续、可聚合、目标端可推导的访问形态：

- 数据可以按目标 PE 聚合为连续通信段，减少 WQE 数量。

- 远端目标地址可以由对称对象和固定偏移推导。

- 同步条件可以用 per-PE quiet、signal 或 AMO counter 表达。

- shape、tile size、PE 数相对稳定，便于模板化和流水化。

如果数据粒度过小、远端通信段不规则、或完成条件只能依赖全局 barrier，需要先通过 AIV 数据重排、layout 约束或阶段拆分降低 UDMA 路径的复杂度。

## 5. 传统路径 vs AIV + UDMA 路径

![traditional_vs_aiv_udma_path](images/traditional_vs_aiv_udma_path.png)

传统路径的主要问题不是单个 launch 的绝对耗时，而是连续业务被拆分为多个 Host 调度阶段。对于小 tile、多 PE、多轮次的通信，Host round trip 与阶段同步开销会被放大。

AIV直驱URMA 在三个维度上改变了通信组织方式：

| 传统问题 | AIV直驱URMA 的处理方式 |
| --- | --- |
| Host launch 和同步降低流水连续性 | 通信在 kernel 内发起，Host 只负责初始化和启动 |
| 远端地址要 Host 拼装 | 对称内存对象同名同地址，本地传指针 + pe 即可 |
| barrier 粒度太粗 | 使用 aclshmemx_udma_quiet(pe)、signal 或 AMO counter 表达局部完成 |

收益来源可归纳为三点：减少 Host round trip、用对称内存的同名约定省掉远端地址拼装、把同步粒度从全局 barrier 降到 PE 或 tile 级别。

相应地，开发者需要在 kernel 内显式管理本地 buffer 生命周期、PE 映射和完成语义——这是后文几节要展开的内容。

## 6. 地址模型：本地 src 与对称内存的非对称用法

device-side allgather 中最容易出现错误的环节是地址处理。关键事实是：`put_nbi` 的两个地址参数语义不对称——`src` 是本端任意本地指针，`dst` 必须是对称内存上的指针。

对应的数据流如图 5 所示：

![remote_write_and_signal_data_flow](images/remote_write_and_signal_data_flow.png)

### 6.1 本端 `src` vs 远端 `dst`

| 参数 | 表达 | 说明 |
| --- | --- | --- |
| 本端 src | 任意本地 GM 指针 | 由 AIV 在本地组织，可以是 input_gm 的某个切片，也可以是 AIV 自己开的中转 buffer。不要求落在对称对象上——UDMA 本端 SGE 只填 va/len |
| 远端 dst | 本地的对称指针 + pe | 必须是对称内存。调用方传入的是本地那一份对称指针，RMA 接口按 heap_base[pe] + offset 公式定位到 pe 上的同一对象 |

换言之：

- 本端是直接拿——AIV 已经把数据组织在某处，把指针交给 `put_nbi` 即可。

- 远端不需要"推导"——只要 `dst` 是对称对象，传指针就够；不必手动算偏移。

如果业务确实需要在同一个对称对象内按 PE 分段（例如目标 PE 上要把"来自各 src PE 的数据"放在不同位置），那是业务自己定义的 layout，不是 SHMEM 强制的。下面是一个常见写法——本端把要发给 PE 0..N-1 的数据按 dst_pe 顺序摆好在 `input_gm` 上，每个 PE 的那段长度都是 `elem_size`：

```
T *src = input_gm  + dst_pe * elem_size;       // 本端：从本地 buffer 中取「要发给 dst_pe」的那一段
T *dst = symm_dst  + my_pe  * elem_size;       // 远端：对称指针 + 业务约定的写入位（按 src_pe 切片）

aclshmemx_udma_put_nbi(dst, src, work_buf, elem_size, dst_pe);

```

注意：

- 本端 `+ dst_pe * elem_size` 是业务自己对 `input_gm` 的本地排布约定，SHMEM 不强制；只要 `src` 指向"这一轮要发给 `dst_pe` 的数据"即可，按 `dst_pe` 摆只是方便循环。

- 远端 `+ my_pe * elem_size` 是业务自己对接收对称对象的切片约定（让目标 PE 知道哪段来自哪个 src），SHMEM 也不强制。

- SHMEM 唯一保证的事情是：当传入"本端对称指针 + `dst_pe`"时，RMA 接口能在 `dst_pe` 上找到同一对象的对应位置。

### 6.2 关键约束

- `src` 是本地视角的地址，AIV 怎么拿到都行。

- `dst` 是本地的对称指针 + 目标 `pe`；因为所有 PE 的 heap 在虚拟地址空间等距，RMA 接口按 `heap_base[pe] + offset` 找到远端的同一对象。

- 所有 PE 同步调用`aclshmem_malloc(size)` 分配相同大小，否则相邻 heap 的等距步长被破坏，`aclshmem_ptr` 公式失效（见 `docs/principles.md` 与 `docs/debug/Troubleshooting_FAQs.md`）。

## 7. 编程模型：AIV 直驱通信的分工

"AIV 直驱"指通信提交点由 Host 侧通信库调用下沉到 AIV kernel 内部。Host 只负责一次性资源准备，AIV 在执行算子逻辑时直接通过 UDMA 发起 RMA / AMO。

| 阶段 | 责任主体 | 主要工作 |
| --- | --- | --- |
| 初始化 | Host | aclshmemx_init_attr、aclshmem_malloc 分配对称对象、建立通信域、准备 jetty / EID 映射、下发 device state |
| 数据组织 | AIV | 在本地组织好要发给各 PE 的 src（任意本地指针，不要求对称） |
| 通信提交 | AIV | 调用 aclshmemx_udma_put_nbi / get_nbi / put_signal_nbi / AMO 类接口提交 WQE，传入本地的对称指针 + pe |
| 完成处理 | AIV | 通过 aclshmemx_udma_quiet(pe)、signal 或 AMO counter 表达局部完成条件 |
| 后续消费 | AIV / 下游 kernel | 在完成条件满足后复用本地 buffer 或在目标对称对象上消费数据 |

## 8. 完成语义与同步模型

AIV 直驱通信中，`put_nbi` / `get_nbi` 的返回并不表示远端数据已经可消费。`nbi` 只表示操作已经完成发起，后续还需要通过明确的完成语义约束本地 buffer 复用、目标端消费和阶段切换。

完成语义可以拆成三个层次：

| 层次 | 表达方式 | 语义边界 |
| --- | --- | --- |
| 本端提交 | put_nbi / get_nbi 返回 | WQE 已发起，不代表远端数据可见 |
| 指定 PE 完成 | aclshmemx_udma_quiet(pe) | 当前 PE 发往该 pe 的已提交 UDMA 操作完成 |
| 目标端可消费 | aclshmem_signal_wait_until + signal、AMO counter 或 barrier | 目标端获得明确到达条件，可以读取对应通信段 |

`aclshmemx_udma_quiet(pe)` 适合约束本端资源生命周期，例如本地 buffer 何时可以复用。它不是 team barrier，也不表示其他 PE 的操作已经完成。

目标端是否可以消费数据，应使用独立的完成通知协议表达。常见做法：

- 使用 `aclshmemx_udma_put_signal_nbi(dst, src, elem, sig_addr, signal, pe)` 在数据写入后写远端 signal；目标端用 `aclshmem_signal_wait_until` 等待。

- 多个源 PE 写入同一目标 PE 时，使用 `aclshmemx_udma_atomic_add` 表达到达计数，避免普通写冲突。

> 同步原则：tile 内使用 PE 级完成语义；阶段切换使用 team barrier。只有所有 PE 的数据都需要统一进入下一阶段时，才使用全局 barrier。

## 9. 实践五步：构建 device-side allgather

以下以 allgather 为例（即 shmem 仓 `examples/udma_demo` 实现的形态）。真实业务里 MoE dispatch、KV shuffle、TP alltoall 都是同一组 UDMA 接口的不同 layout 而已，通信主干一致。

### 第一步：Host 初始化 SHMEM 和对称对象

```
aclshmemx_init_attr_t attr;
test_set_attr(my_pe, npes, local_mem_size, ip_port, uid, &attr);

// 把 SHMEM 数据面后端切到 UDMA
attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);

// 对称内存：所有 PE 同步调用、相同 size
//  - 用一个对称对象同时承载 src 段（my_pe 那段）和 dst 段（其它 PE 写来的段）
auto *gva = static_cast<uint8_t *>(aclshmem_malloc(npes * msg_size));

// Host 把本 PE 的输入写到 gva 上属于自己的那段
aclrtMemcpy(gva + my_pe * msg_size, msg_size, host_input,
            msg_size, ACL_MEMCPY_HOST_TO_DEVICE);

```

`option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA` 是把 SHMEM 数据面切到 UDMA 后端的关键开关，没设会与 `aclshmemx_udma_*` 接口不匹配。Host 还需要完成通信域、jetty / EID、device state 等初始化（由 `aclshmemx_init_attr` 一并完成）。

> 强制约定：所有 PE 同步调用 `aclshmem_malloc` 且分配相同 size，否则相邻 heap 的等距步长被破坏，远端写入会落到错位地址（见 `docs/debug/Troubleshooting_FAQs.md`）。

### 第二步：Device 内获取 PE 信息

```
const int64_t my_pe = aclshmem_my_pe();
const int64_t npes  = aclshmem_n_pes();

```

如果使用 SHMEM team，还需要先获取 team 内 rank，再映射到全局 PE。不应混用 team 内编号和全局 PE 编号。

### 第三步：AIV 准备 UDMA work buffer

```
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

```

UDMA 接口要求一段 ≥ 64 B 的 UB 工作区（参见 `aclshmemx_udma_put_nbi` 的 `buf` 参数说明）。本步无需把数据搬到本地——allgather 的 src 已经在 `gva + my_pe * msg_size` 上。

### 第四步：AIV 把本 PE 的段数据push 给所有其他 PE

```
for (int i = 0; i < npes; ++i) {
    if (i == my_pe) {
        continue;                                 // self 不走 UDMA
    }
    aclshmemx_udma_put_nbi(
        gva + msg_size * my_pe,                   // dst：远端 gva 上 my_pe 的那一段
        gva + msg_size * my_pe,                   // src：本端 gva 上 my_pe 的那一段
        (__ubuf__ uint8_t *)ubLocal.GetPhyAddr(),
        msg_size, i);
    aclshmemx_udma_quiet(i);                      // UDMA 限制：避免对同一目标 PE 并发 post
}

```

要点：

- 本端 src 和远端 dst 同名同 offset：因为 allgather 就是把"本PE数据"广播到所有PE相同位置。

- self 分支跳过——同一对象在本 PE 上即为本地数据。

- `udma_put_nbi` 是非阻塞，但 UDMA 头文件明确"concurrent RMA/AMO to the same PE are not supported"，所以参考样例每次 put 之后都跟一个对应 `pe` 的 `udma_quiet`。

### 第五步：阶段 barrier

```
aclshmemx_barrier_all_vec();

```

allgather 完成时所有 PE 的所有段都需要落地，因此用集合 barrier。如果业务只关心"和某个 PE 的对应段已经到达"，可以替换为 `aclshmemx_udma_put_signal_nbi` + `aclshmem_signal_wait_until` 做点对点同步。

完整路径如图 6 所示：

![aiv_urma_complete_data_path](images/aiv_urma_complete_data_path.png)

## 10. 参考示例

下面参考 shmem 仓 `examples/udma_demo/udma_demo_kernel.cpp` 的 all-gather 模式。

代码中：

- `ACLSHMEM_DEVICE`：device 函数修饰宏，标记本函数运行在 NPU 侧。

- `GM_ADDR`：device kernel 入参类型，指向 NPU 全局显存（GM）。

- `__ubuf__`：指针指向 AI Core 内部缓存（UB），UDMA 需要至少 64 B 的 UB 工作区。

### 10.1 device kernel：`udma_put_nbi` 版

每个 PE 把"自己那一段" `gva + msg_len * my_pe` push 到所有其他 PE 的同名段：

```
extern "C"__global__ __aicore__ void udma_all_gather_kernel(
    GM_ADDR gva, GM_ADDR dump, int message_length)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t my_pe   = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    for (int i = 0; i < pe_size; i++) {
        if (i == my_pe) {
            continue;                       // self 不走 UDMA
        }
        aclshmemx_udma_put_nbi(
            gva + message_length * my_pe,    // dst: 远端 gva 上 my_pe 的那一段
            gva + message_length * my_pe,    // src: 本端 gva 上 my_pe 的那一段
            (__ubuf__ uint8_t *)ubLocal.GetPhyAddr(),
            message_length, i);
        aclshmemx_udma_quiet(i);             // 避免对同一目标 PE 并发 post
    }
    aclshmemx_barrier_all_vec();
}

```

要点：

- 本端 `src` 和远端 `dst` 都用同一个 `gva`，因为 SHMEM 对称内存保证两端的同名段一一对应。本端不需要另开本地 buffer——业务数据已经由 Host 写到了 `gva` 上对应位置。

- self 分支 (`i == my_pe`) 跳过 UDMA，因为同一对象在本 PE 上即为本地数据，不需要走 RMA。

- 每次 put 后跟一个 `aclshmemx_udma_quiet(i)`，与 UDMA 头文件中"concurrent RMA/AMO to the same PE are not supported"的限制对齐。

### 10.2 带 signal 的版本：`udma_put_signal_nbi`

如果目标端要在数据到达后立即消费，把 `udma_put_nbi` 换成 `udma_put_signal_nbi`，并提供一段对称的 `sig_addr`：

```
extern "C"__global__ __aicore__ void udma_put_signal_kernel(
    GM_ADDR gva, GM_ADDR sig_addr, GM_ADDR dump_addr, int message_length, uint64_t signal)
{
    int64_t my_pe   = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    for (int i = 0; i < pe_size; i++) {
        if (i == my_pe) {
            continue;
        }
        // 给每个目标 PE 用不同的 signal 槽，避免覆盖
        auto dst_sig_addr = sig_addr + sizeof(uint64_t) * my_pe;
        aclshmemx_udma_put_signal_nbi(
            gva + message_length * my_pe,
            gva + message_length * my_pe,
            message_length,
            (__gm__ uint64_t *)dst_sig_addr,
            signal, i);
        aclshmemx_udma_quiet(i);
    }
    aclshmemx_barrier_all_vec();
}

```

目标端用 `aclshmem_signal_wait_until` 等待对应 PE 的 signal 槽变为 `signal`，即可消费数据。

### 10.3 Host 侧的对应工作

参考 `examples/udma_demo/main.cpp` 的写法：

```
aclInit(nullptr);
aclrtSetDevice(device_id);
aclrtCreateStream(&stream);

aclshmemx_init_attr_t attributes;
test_set_attr(pe_id, n_pes, local_mem_size, ipport, uid, &attributes);
attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

uint8_t *ptr = static_cast<uint8_t *>(aclshmem_malloc(npes * msg_size));

// 把本 PE 的输入写到 ptr 上属于自己的那段
aclrtMemcpy(ptr + my_pe * msg_size, msg_size, host_input,
            msg_size, ACL_MEMCPY_HOST_TO_DEVICE);

// 启动 device kernel
launch_udma_all_gather(block_dim, stream, ptr, dump, msg_size);
aclrtSynchronizeStream(stream);

```

注意 `option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA`——这是把 SHMEM 的数据面后端切到 UDMA 的关键开关。如果不设置，默认走 MTE 后端，调用 `aclshmemx_udma_*` 接口会与初始化不匹配。

## 11. 小结

AIV + UDMA 并非将 Host 通信 API 机械迁移到 Device 上，而是改变通信编排的位置。Host 执行一次性初始化；进入 kernel 后，AIV 按业务 shape 组织数据、发起 RMA、处理局部同步，并继续执行下游计算。

该路径适用于具备以下特征的场景：

- 数据天然按目标 PE 聚合，或者可以用 AIV 低成本组织成"按 PE 聚合的本地段"。

- 通信和计算紧密交错，Host 侧分阶段调度会带来明显空洞。

- shape、tile、PE 关系比较稳定，适合常量化和流水化。

- 正确性协议可以通过 `aclshmemx_udma_put_signal_nbi` + `aclshmem_signal_wait_until`、AMO counter 或阶段 barrier 明确表达。

到达通知的典型模式如图 7 所示：

![arrival_notification_flow](images/arrival_notification_flow.png)

> signal 表示一种协议约定。数据区和 signal 区的顺序、可见性和等待方式要与平台 API 语义保持一致，不应以普通内存可见性观察替代明确的完成语义。

当上述条件成立时，allgather、alltoall、MoE dispatch / combine、KV shuffle 等路径可以避免被拆分为"计算 → 返回 Host → 通信 → 再次返回 Host"的多阶段流程，而是组织为同一个 AIV kernel 内的连续执行路径：完成本地 src 组织后，将数据写入目标 PE 的对称内存。从而优化最终融合算子性能。

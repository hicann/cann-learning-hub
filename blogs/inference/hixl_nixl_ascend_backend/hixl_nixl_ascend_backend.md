# HIXL快速适配NIXL昇腾后端

## 1 背景介绍

HIXL 是昇腾面向高性能数据传输场景提供的通信能力组件，支持内存注册、建链、同步/异步传输、状态查询与通知等关键能力，并提供简洁、易用的 API，便于业务快速集成。当前，vLLM、SGLang 等多个主流 AI 开源框架已通过调用HIXL 接口，在昇腾设备上实现高性能数据传输。

在大模型PD分离部署业务场景中，HIXL 的易用性再次体现出价值。原先业务运行在非昇腾硬件，迁移到昇腾时，节点间KV Cache的传输功能的适配成为主要挑战。此前，业务在非昇腾硬件上KV Cache的高效传输基于开源 NIXL 框架提供的点对点通信服务实现；若放弃 NIXL、直接基于昇腾底层接口重构，改造成本和风险都很高。得益于 NIXL 插件化的后端架构，以及 HIXL 清晰易用的接口能力，项目快速完成了 NIXL 对昇腾后端的适配，最终使客户仅需少量业务改动，即可完成KV Cache传输功能的迁移。

## 2 HIXL使用介绍

### 2.1 接口介绍

| 接口 | 功能 |
| --- | --- |
| Initialize | 初始化 HIXL |
| Finalize | 释放 HIXL 资源 |
| RegisterMem | 注册内存用于后续传输 |
| DeregisterMem | 解注册内存 |
| Connect | 与远端建链 |
| Disconnect | 与远端断链 |
| TransferSync | 同步数据传输 |
| TransferAsync | 异步数据传输 |
| GetTransferStatus | 查询异步请求状态 |
| SendNotify | 发送通知 |
| GetNotifies | 获取并清空已收到通知 |

### 2.2 HIXL点对点远端通信介绍

假设现在存在节点A和B,节点A需要读取或者给B发送数据,使用HIXL接口实现的流程图如下:

![hixl_point_to_point_communication_flow](images/hixl_point_to_point_communication_flow.jpeg)

其中的关键点:

1.由于是节点A发起请求,因此建链以及传输指令都是由节点A执行,节点B只需要在建链之前完成好内存的申请注册.

2.由于使用的异步传输接口TransferAsync,因此节点A需要使用GetTransferStatus循环获取传输状态,直到确认传输成功后将消息通过SendNotify发送给节点B,此时节点A,B都确认传输完成,才能进行后续的资源释放.

### 3 通过HIXL适配NIXL昇腾后端介绍

NIXL 采用模块化插件架构，不同后端都可以通过注册插件的方式接入 NIXL。NIXL 内部由 Plugin Manager 统一管理这些插件，负责插件的发现、加载、卸载以及实例化。

![nixl_plugin_architecture](images/nixl_plugin_architecture.jpeg)

后端插件适配的核心在于实现 NIXL 提供的 SB（SouthBound）API。NIXL 将底层通信能力抽象为一组统一接口，上层调用这些接口时不感知具体后端差异；不同后端只需要按照各自通信库或驱动能力，对这些接口给出对应实现即可。此外，SB API 并不要求全部实现。NIXL 提供了四个能力声明接口，用于描述后端支持的通信能力，Agent 会根据这些能力开关判断该后端需要实现哪些 SB API。也就是说，不同能力组合对应不同的最小实现集合。例如，本次昇腾后端适配的目标是支持远端点对点通信，因此不需要支持节点内通信，对应的 supportsLocal() 可以直接返回 false，从而表明节点内通信相关接口无需实现。

#### 3.1 能力开关接口实现

本次昇腾后端适配面向远端点对点通信场景，因此能力设计上应以“支持跨节点通信、支持通知、不支持节点内通信”为主，并通过 getSupportedMems() 明确声明该后端可处理的内存类型:

| 接口名 | 实现代码 | 接口功能介绍 |
| --- | --- | --- |
| supportsLocal() | bool supportsLocal() const override { return false; } | 声明后端是否支持节点内通信。本次昇腾后端仅面向远端点对点通信，因此这里返回 false，表示不需要实现节点内通信相关能力。 |
| supportsRemote() | bool supportsRemote() const override { return true; } | 声明后端是否支持跨节点通信。昇腾后端本次适配目标就是远端点对点通信，因此这里应返回 true。 |
| supportsNotif() | bool supportsNotif() const override { return true; } | 声明后端是否支持通知能力。对于昇腾后端，通知用于传输完成后的节点消息同步，因此返回 true。 |
| getSupportedMems() | nixl_mem_list_t getSupportedMems() const override { return {VRAM_SEG}; } | 声明该后端支持的内存类型。当前昇腾后端只需支持VRAM_SEG完成设备内存的传输实现远端点对点通信。 |

#### 3.2 SB API实现

根据能力开关接口的设计，SB API 可以分为两类：一类是所有后端都必须实现的基础接口；另一类是由能力开关决定是否需要实现的扩展接口。

对于任意后端，首先需要实现 9 个基础 SB API，这些接口构成了后端接入 NIXL 的最小功能集合。除此之外，NIXL 会根据 4 个能力开关接口所声明的能力，进一步确定该后端是否还需要实现对应的扩展接口。因此，除 9 个基础 SB API 外，还需要额外实现 6 个与远端通信和通知能力相关的 SB API，总计需要实现 15 个接口。

基础 SB API

| 接口名 | 接口功能说明 |
| --- | --- |
| registerMem() | 向后端注册一块本地内存，并生成该内存对应的后端元数据对象，供后续传输使用。 |
| deregisterMem() | 注销已注册内存，并释放 registerMem() 创建的本地元数据资源。 |
| connect() | 与指定 Agent 建立连接。即使后端不需要显式建链，也需要提供该接口，可实现为 no-op。 |
| disconnect() | 断开与指定 Agent 的连接，并释放相关连接资源。 |
| unloadMD() | 释放已经加载的目标侧或远端侧元数据对象。 |
| prepXfer() | 在真正发起传输前完成准备工作，构造并初始化一次传输请求的后端句柄。 |
| postXfer() | 正式提交传输请求，启动异步传输流程。 |
| checkXfer() | 查询当前传输请求的执行状态，例如进行中、完成或失败。 |
| releaseReqH() | 释放传输请求句柄；必要时还需要负责取消或回收未完成请求。 |

由能力开关决定的 SB API

| 能力开关 | 需要额外实现的 SB API | 接口功能说明 |
| --- | --- | --- |
| supportsRemote() | getPublicData() | 导出已注册本地内存的公开元数据，供远端节点访问该内存时使用。 |
| supportsRemote() | getConnInfo() | 导出当前后端实例的连接信息，供远端节点建立通信关系。 |
| supportsRemote() | loadRemoteConnInfo() | 加载远端节点发来的连接信息，为后续远端通信做准备。 |
| supportsRemote() | loadRemoteMD() | 加载远端节点导出的内存元数据，并转换为本地可用的后端元数据对象。 |
| supportsNotif() | getNotifs() | 获取来自远端节点的通知消息，例如传输完成通知或控制通知。 |
| supportsNotif() | genNotif() | 生成并发送通知消息到远端节点。 |

对于 NIXL 而言，通过这些 SB API 实现 2.2 中远端点对点通信的示意流程图如下：

![nixl_sb_api_communication_flow](images/nixl_sb_api_communication_flow.jpeg)

对比 HIXL 的接口流程可以看出，大部分 NIXL SB API 与 HIXL 原生接口在语义上是基本一致的，因此可以直接映射实现；但两者仍存在几个关键差异：

1.NIXL SB API 中没有单独提供初始化和去初始化接口

这是因为 NIXL 将后端的初始化和清理过程收敛到了插件实例的构造函数与析构函数中，而不是作为独立的 SB API 暴露出来。因此在实现昇腾后端时，只需要在后端插件的构造函数中调用 HIXL 的 Initialize()，在析构函数中调用HIXL 的 Finalize() 即可。

2.NIXL 额外定义了远端连接与远端内存元信息相关接口

相较于 HIXL，NIXL 额外定义了 getPublicData()、getConnInfo()、loadRemoteConnInfo()、loadRemoteMD() 和 unloadMD() 这 5 个接口。出现这些接口的根本原因在于两者的抽象层次不同。HIXL 原生接口中，应用直接通过"ip:port" 字符串标识形式识别远端节点，并通过内存地址和内存地址大小完成远端内存访问，使用便捷简单,因此 HIXL 本身并不需要额外维护统一的“远端连接元信息”和“远端内存元信息”抽象。

而 NIXL 作为统一的上层通信框架，需要兼容不同后端的连接模型和远端内存访问模型，因此必须抽象出一套通用的 metadata 导出、交换、加载与释放机制。因此对于昇腾后端来说，这几个接口实现时可以采用轻量化适配方式：

- getConnInfo() / loadRemoteConnInfo() 直接传递加载节点的字符串标识；

- getPublicData() / loadRemoteMD() /unloadMD()空实现；

3.NIXL 额外定义了 prepXfer() 和 releaseReqH() 接口

与第二点类似，这一差异同样来源于 HIXL 与 NIXL 抽象层次的不同。HIXL 更强调接口使用上的直接性，因此通过传输接口直接执行通信；而 NIXL 将“请求准备”和“传输执行”分离，以便更好地兼容多种后端。对于昇腾后端，实现上只需在 prepXfer() 中申请一块内存保存请求信息，在 releaseReqH() 中释放即可。

最后可以得到每个SB API的实现思路:

| 接口名称 | 实现思路 |
| --- | --- |
| registerMem() | 直接调用 HIXL RegisterMem |
| deregisterMem() | 直接调用 HIXL DeregisterMem |
| connect() | 直接调用 HIXL Connect |
| disconnect() | 直接调用 HIXL Disconnect |
| prepXfer() | 申请一块内存记录请求信息 |
| postXfer() | 直接调用HIXL TransferAsync |
| checkXfer() | 直接调用 HIXL GetTransferStatus 查询状态 |
| releaseReqH() | 释放记录请求信息的内存 |
| getPublicData() | 空实现 |
| getConnInfo() | 获取本地的字符串标识 |
| loadRemoteConnInfo() | 加载远端的字符串标识 |
| loadRemoteMD() | 空实现 |
| unloadMD() | 空实现 |
| getNotifs() | 直接调用 HIXL GetNotifies |
| genNotif() | 直接调用 HIXL SendNotify |

### 4.总结

HIXL 凭借简洁易用的 API 设计和高效的数据传输能力，已经在多个实际业务场景和开源框架中落地应用，并取得了良好效果。同时，HIXL 也在持续演进，不断完善功能、提升性能。后续可以持续关注 HIXL 开源仓，获取最新的能力演进和技术动态。

HIXL: https://gitcode.com/cann/hixl/tree/master

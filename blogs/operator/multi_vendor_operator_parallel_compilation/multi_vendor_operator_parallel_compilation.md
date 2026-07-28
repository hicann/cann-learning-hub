# 多Vendor自定义算子并行编译实践

## 1. 背景介绍

随着自定义算子开发规模不断扩大，开发者在一个代码仓中维护多个算子包已经越来越常见。不同算子可能来自不同业务团队，也可能需要按照不同 Vendor 独立交付。每个 Vendor 都有自己的源码目录、芯片型号配置、依赖参数和安装包输出要求。

如果仍然采用逐个目录手动编译的方式，整个流程很容易变得繁琐：

- 每个 Vendor 都要重复执行配置、编译、打包命令。

- 多个安装包输出路径不统一，后续查找和发布成本高。

- 不同算子支持的 Ascend 芯片型号不同，手动配置容易出错。

- 子工程之间缺少统一调度，新增 Vendor 时需要重新整理构建流程。

多 Vendor 并行编译要解决的正是这些问题。它的核心目标不是改变每个算子工程内部的实现方式，而是在外层增加一个统一入口，把多个独立 Vendor 工程组织起来，一次触发、并行构建、分别产出。

除了降低操作复杂度，多 Vendor 并行编译还有一个直接收益：当多个 Vendor 子工程之间没有强依赖时，顶层构建可以让它们并行推进，使总耗时从“各 Vendor 编译耗时累加”接近“最长 Vendor 编译耗时加少量调度开销”。在典型多 Vendor 场景下，相比逐个目录串行编译，整体编译时长可缩短约 40%。实际收益会受 CPU 核数、I/O、依赖下载、子工程规模等因素影响。

简单来说，就是：

> 一个顶层入口统一调度，多个 Vendor 工程独立编译，最终生成各自的安装包。

## 2. 方案介绍

多 Vendor 并行编译可以分成三层理解：顶层构建入口、独立 Vendor 子工程、独立产物输出。

![multi_vendor_parallel_compilation_workflow](images/multi_vendor_parallel_compilation_workflow.png)

多 Vendor 自定义算子并行编译流程

第一层是顶层构建入口。顶层工程负责管理有哪些 Vendor 子工程、每个子工程在哪里、使用什么 Vendor 名称、需要编译哪些 Ascend 芯片型号。开发者执行编译命令时，只需要从顶层入口发起构建。

第二层是独立 Vendor 子工程。每个 Vendor 子工程仍然保留自己的目录结构和 CMake 逻辑。它可以是扁平目录结构，也可以是 `framework`、`op_host`、`op_kernel` 这类分层结构。顶层工程只负责调度，不强行改造子工程内部实现。

第三层是独立产物输出。每个 Vendor 子工程编译完成后，都会生成自己的中间产物和 `custom_opp_*.run` 安装包。不同 Vendor 的输出目录互相隔离，方便后续安装、测试、归档和发布。

这种组织方式适合以下场景：

- 一个仓库中维护多个自定义算子。

- 多个业务团队分别维护自己的 Vendor 算子包。

- 希望一次命令完成多个算子包的编译和打包。

- 希望统一管理芯片型号、依赖参数和安装输出目录。

- 希望不同 Vendor 的产物互不影响，便于定位问题。

## 3. 核心实现

多 Vendor 并行编译的关键在于：把每个 Vendor 子工程当成独立工程处理，再由顶层工程统一调度。

在 CMake 中，可以使用 `ExternalProject_Add` 实现这一点。参考工程中将这部分逻辑封装成了一个函数：

```
function(add_parallel_vendor_project target_name source_dir vendor_name compute_units)

```

这个函数可以理解为“注册一个 Vendor 编译任务”。它主要接收四类信息：

| 参数 | 作用 | 说明 |
| --- | --- | --- |
| target_name | 顶层任务名称 | 用于区分不同 Vendor 构建任务 |
| source_dir | 子工程源码目录 | 指向具体算子工程所在路径 |
| vendor_name | Vendor 名称 | 用于传递给子工程并参与打包 |
| compute_units | 芯片型号配置 | 指定需要编译支持的 Ascend 芯片型号 |

以两个 Vendor 工程为例，顶层 CMake 可以这样注册：

```
add_parallel_vendor_project(
    add_custom
    ${CMAKE_CURRENT_SOURCE_DIR}/add_custom
    add_custom
    "${ADD_CUSTOM_ASCEND_COMPUTE_UNIT}"
)

add_parallel_vendor_project(
    leaky_relu_custom
    ${CMAKE_CURRENT_SOURCE_DIR}/leaky_relu_custom
    leaky_relu_custom
    "${LEAKY_RELU_ASCEND_COMPUTE_UNIT}"
)

```

其中，芯片型号可以在顶层统一配置：

```
set(ADD_CUSTOM_ASCEND_COMPUTE_UNIT ascend910 ascend310p ascend310b ascend910b ascend950)
set(LEAKY_RELU_ASCEND_COMPUTE_UNIT ascend910b)

```

这样做的好处是，每个 Vendor 都可以保留自己的芯片型号范围，顶层又能统一管理这些配置。

## 4. 工程结构

为了更直观地理解这套机制，可以参考如下工程：

parallel_ops_package 多 Vendor 并行编译参考工程：

https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/parallel_ops_package

该工程中包含两个 Vendor 子工程：

- `add_custom`：采用扁平目录组织，Host、Kernel、Tiling 相关代码位于同一层级。

- `leaky_relu_custom`：采用分层目录组织，包含 `framework`、`op_host`、`op_kernel` 等目录。

目录结构如下：

```
parallel_ops_package
├── CMakeLists.txt
├── add_custom
│   ├── CMakeLists.txt
│   ├── add_custom_host.cpp
│   ├── add_custom_kernel.cpp
│   └── add_custom_tiling.h
├── leaky_relu_custom
│   ├── CMakeLists.txt
│   ├── framework
│   ├── op_host
│   └── op_kernel
└── README.md

```

从这个结构可以看到，多 Vendor 并行编译并不要求所有子工程使用完全相同的目录风格。只要子工程自身能够独立完成自定义算子的编译和打包，就可以接入顶层调度。

工程结构这一层关注的是“源码如何组织”。顶层目录保存统一入口，Vendor 子目录保存各自的 Host、Kernel、Tiling、框架适配等实现。这样做的好处是，既能保持各 Vendor 工程的独立性，又能通过顶层工程把它们纳入同一套编译流程。

## 5. 操作实践

在参考工程根目录下，可以通过以下步骤完成多 Vendor 并行编译。

### 5.1 配置环境变量

根据当前环境中 CANN 开发套件包的安装路径配置环境变量：

```
source ${install_path}/cann/set_env.sh

```

其中，`${install_path}` 为 CANN 包安装目录。如果安装时未指定路径，通常位于 `/usr/local/Ascend`。

### 5.2 执行编译和打包

进入下载后的参考工程目录：

```
cd parallel_ops_package

```

执行 CMake 配置和构建：

```
cmake -S . -B build
cmake --build build -j

```

这里的 `-j` 表示并行构建。顶层 CMake 会调度多个 Vendor 子工程，同时推进各自的编译和打包流程。

### 5.3 安装生成的算子包

编译完成后，可以分别安装生成的 `custom_opp_*.run` 包：

```
./build/add_custom/custom_opp_*.run
./build/leaky_relu_custom/custom_opp_*.run

```

执行成功后，终端会显示：

```
SUCCESS

```

## 6. 产物输出

多 Vendor 并行编译完成后，不同 Vendor 会输出到各自独立目录中：

| Vendor 工程 | 输出目录 | 输出内容 |
| --- | --- | --- |
| add_custom | build/add_custom/ | AddCustom 的中间产物与安装包 |
| leaky_relu_custom | build/leaky_relu_custom/ | LeakyReluCustom 的中间产物与安装包 |

这种目录隔离方式可以带来更清晰的工程管理效果：

- 编译产物不会互相覆盖。

- 每个 Vendor 的安装包路径固定，便于自动化脚本处理。

- 某个 Vendor 编译失败时，可以更快定位到对应子工程。

- 后续发布时，可以按 Vendor 独立归档或交付。

## 7. 扩展方式

当需要新增一个 Vendor 工程时，不需要重写整套构建流程，只需要在顶层 CMake 中新增一组配置。

假设新增工程目录为：

```
my_op_custom

```

可以先定义该工程支持的芯片型号：

```
set(MY_OP_ASCEND_COMPUTE_UNIT ascend910b)

```

然后注册新的 Vendor 编译任务：

```
add_parallel_vendor_project(
    my_op_custom
    ${CMAKE_CURRENT_SOURCE_DIR}/my_op_custom
    my_op_custom
    "${MY_OP_ASCEND_COMPUTE_UNIT}"
)

```

完成后重新执行：

```
cmake -S . -B build
cmake --build build -j

```

新的 Vendor 工程就会和已有工程一起参与并行编译。随着 Vendor 数量增加，顶层 CMake 仍然保持统一入口，开发者不需要在多个目录之间反复切换。

## 8. 使用建议

为了让多 Vendor 并行编译流程更稳定，建议关注以下几点。

第一，保证每个 Vendor 子工程可以独立编译。

顶层 CMake 负责统一调度，但不会修复子工程内部的源码、CMake 或依赖问题。在接入顶层工程前，应先确认子工程自身可以完成编译和打包。

第二，保持 Vendor 名称唯一。

`vendor_name` 会影响打包名称和安装路径。多个 Vendor 工程不建议使用相同名称，否则可能造成产物覆盖或问题定位困难。

第三，按实际产品配置芯片型号。

不同算子支持的 Ascend 芯片型号可能不同。例如参考工程中，`add_custom` 支持多个芯片型号，`leaky_relu_custom` 配置为 `ascend910b`。实际开发时，应以目标产品和算子支持范围为准。

第四，统一管理公共依赖。

如果多个 Vendor 子工程依赖相同的第三方资源，建议在顶层统一配置并向子工程传递。参考工程中，`leaky_relu_custom` 使用了 `NLOHMANN_JSON_URL` 参数，顶层 CMake 会在需要时传递给对应子工程。

## 9. 总结

多 Vendor 并行编译提供了一种更适合规模化自定义算子开发的工程组织方式。它通过顶层 CMake 统一调度多个独立 Vendor 子工程，实现一次触发、并行构建、分别打包和独立输出。

相比逐个目录手动编译，这种方式可以减少重复操作，降低参数配置错误概率，并让构建产物更加清晰可控。在多个 Vendor 子工程可并行推进时，还能把串行等待转化为并行执行，进一步缩短整体构建等待时间。

开发者可以先参考 `parallel_ops_package` 工程理解整体流程，再将同样的组织方式应用到自己的多算子、多 Vendor 工程中。

核心价值可以概括为：

> 一个入口统一构建，多个 Vendor 独立产出。

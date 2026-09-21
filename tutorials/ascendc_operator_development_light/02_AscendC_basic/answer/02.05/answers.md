# 2.5 静态 Tensor 课后实践参考答案

完整实现见 [sub_static.asc](./sub_static.asc)。该源码与课堂 Add 使用相同的单 buffer 切分和显式同步，计算指令改为 `AscendC::Sub`，Host 侧用 `x - y` 计算 Golden。

通过唯一的 UB `LocalMemAllocator` 线性分配三个 LocalTensor，不使用 TPipe/TQue/TBuf。

从本章目录可直接编译运行（先加载 CANN 环境；设备编号按实际情况调整）：

```bash
cmake -S src/02.05 -B build/02.05_answer -DKERNEL_SOURCE="$PWD/answer/02.05/sub_static.asc"
cmake --build build/02.05_answer -j2
./build/02.05_answer/demo 0
```

用例固定为 8 核、每核 8 个 tile，共 16384 个 float 元素。预期输出：

```text
[PASS] Sub blocks=8 tiles/core=8 elements=16384 max_error=0
```

`sub_static.asc` 包含完整的 Device 与 Host 代码，CMake 配置来自 `src/02.05`。

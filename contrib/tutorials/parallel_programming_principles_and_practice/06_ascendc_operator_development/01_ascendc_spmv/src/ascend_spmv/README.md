# Ascend C FP32 CSR SpMV

本工程包含一条真实 Ascend C FP32 SpMV 路径，以及若干明确标记的 Host mixed-precision prototypes。两类结果必须分开解释。

## 真实 Device 调用链

`AscendCSpmvBackend` 的调用链为：

```text
aclInit / aclrtSetDevice / aclrtCreateStream
  → aclrtcCreateProg(kernels/spmv_fp32_kernel.cpp)
  → aclrtcCompileProg
  → aclrtcGetBinDataSize / aclrtcGetBinData
  → aclrtBinaryLoadFromData
  → aclrtBinaryGetFunction
  → aclrtMalloc(CSR, x, y)
  → aclrtMemcpy(H2D)
  → aclrtLaunchKernelWithConfig
  → aclrtSynchronizeStream
  → aclrtMemcpy(D2H)
  → CPU reference error
```

Kernel 按行分配任务，每个 AI Core 写互不重叠的 `y[row]`。当前 baseline 使用 FP32 CSR、FP32 输入和 FP32 累加，目标架构沿用课程现有 RTC 示例的 `dav-2201`（Ascend 910B3）。

原来的 FP16、BF16 和 persistent 实现仍由 Host C++/OpenMP 循环执行，类型和 CSV 字段已改为 `HostPrototype*` / `host_prototype_*`。它们只用于数据布局和算法实验，不能记录为 NPU Kernel 性能。

## 正式构建与运行

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPMV_REAL_ASCENDC=ON
cmake --build build -j
./build/bin/spmv_benchmark --matrix U1 --warmup 3 --repeat 10 \
  --results-dir results/current --csv results/current/spmv_benchmark.csv
```

`SPMV_REAL_ASCENDC=ON` 是默认正式模式。找不到 `acl/acl.h`、`libascendcl` 或 `libacl_rtc` 时 CMake 直接失败，不自动回退。只有本地 Host 语法诊断才可显式使用：

```bash
cmake -S . -B build-host -DSPMV_REAL_ASCENDC=OFF
```

该模式生成的 Host Prototype 数据不是 Ascend 实测。

## 验收字段

正式运行必须记录：日志 `Actual Backend=Ascend C RTC`（CSV 中 `actual_backend=ascend_c`）、Device ID、矩阵规模、H2D（`ascendc_transfer_in_ms`）、Kernel+Sync（`ascendc_launch_to_complete_ms`，launch-to-complete 口径）、D2H（`ascendc_transfer_out_ms`）、CPU Reference Error 和退出码。FP32 相对误差阈值为 `1e-6`。仓库历史 CSV 来自旧 Host 模拟实现，仅用于了解旧字段，不代表当前 Kernel 结果；真实数据必须由上述命令重新生成。

Ascend 950 SIMT 内容仍为独立的待验证模板，不能用 910B3 RTC 结果代替。

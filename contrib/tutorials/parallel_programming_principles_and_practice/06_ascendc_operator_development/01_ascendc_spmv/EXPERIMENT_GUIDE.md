# Ascend C FP32 SpMV 与混合精度 Host 原型实验手册

## 1. 当前实现边界

工程现已提供真实 FP32 Ascend C RTC 路径（真实类 `AscendCSpmvBackend`）：`aclrtMalloc` 分配 CSR/x/y，`aclrtMemcpy` 完成 H2D/D2H，RTC 链为 `aclrtcCreateProg` → `aclrtcCompileProg` → `aclrtcGetBinDataSize`/`aclrtcGetBinData` → `aclrtBinaryLoadFromData` → `aclrtBinaryGetFunction` → `aclrtLaunchKernelWithConfig` → `aclrtSynchronizeStream`。原 FP16/BF16/persistent C++ 循环已统一重命名为 **Host Prototype**，不能作为 NPU 混合精度结果。

当前比较包括 CPU reference、Host prototypes 与真实 Ascend C FP32 baseline。只有日志 `Actual Backend=Ascend C RTC`（CSV 中 `actual_backend=ascend_c`）且 Device/H2D/Kernel+Sync/D2H 全链成功的行可标记为 NPU。注意：`ascendc_launch_to_complete_ms` 是 launch-to-complete 口径（kernel 提交到 stream 同步完成），不是纯 kernel 执行时间。

## 2. 构建与当前能力验证

```bash
cd src/ascend_spmv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPMV_REAL_ASCENDC=ON
cmake --build build -j
./build/bin/spmv_benchmark --warmup 1 --repeat 10 \
  --results-dir results/current --csv results/current/spmv_benchmark.csv
```

以 `spmv_csr_reference` 为 reference，记录各路径相对误差；FP32 采用 `1e-6`，BF16/FP16 必须依据课程给定精度阈值单独判定，不得沿用 FP32 结论。

## 3. 真实 Ascend C 路径应包含

```text
Host CSR/x → Device allocation → H2D → Ascend C kernel launch
         → CSR row processing / Vector API → device synchronization → D2H
         → CPU reference verification
```

950 SIMT 内容在没有目标硬件日志和工具链证据时只能作为待验证设计模板，不能声称已完成。学生应在报告中逐项附上 API、编译产物和运行日志证据。

| Path | Actual backend | Error | Time (ms) | Device evidence | Conclusion |
|---|---|---:|---:|---|---|
| CPU reference | CPU | 0 | | N/A | reference |
| baseline prototype | Host CPU | | | none | algorithm prototype |
| optimized prototype | Host CPU | | | none | algorithm prototype |

练习：画出 `AscendCSpmvBackend` 的资源生命周期与 RTC 调用链，并指出当前工程中仍属 Host 原型的环节（FP16/BF16/persistent）。参考答案见 `answer/`。

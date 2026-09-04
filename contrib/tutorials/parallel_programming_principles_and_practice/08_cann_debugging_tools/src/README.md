# 调试实验源码

- `add_sanitizer.asc` + `CMakeLists.txt`：独立 ASC 可执行程序；
- `operator_project_sanitizer/`：标准 `op_host + op_kernel + 算子包 + ACLNN Host` 工程；
- `UPSTREAM.md`、`LICENSE.asc-devkit`：样例来源与许可证。

独立 ASC：

```bash
MODE=memcheck
cmake -S . -B build-$MODE -DLAB05_FAULT_MODE=$MODE -DLAB05_ENABLE_SANITIZER=ON
cmake --build build-$MODE -j
mssanitizer --tool=$MODE ./build-$MODE/lab05_add_sanitizer
```

`MODE` 可替换为 `racecheck`、`initcheck`、`synccheck`。故障程序可能返回非零，必须同时记录退出码并
检查 MODE 对应诊断；匹配时状态是 `EXPECTED_DIAGNOSTIC`，不是正常程序 `PASS`。标准算子工程的
完整直接命令见 `operator_project_sanitizer/README.md`。

msDebug 只读检查 `/proc/debug_switch` 和 `/dev/drv_debug`。课程不执行 `sudo`、`chmod`、`chown`，
也不写系统调试开关；权限不足时记录 `BLOCKED`。

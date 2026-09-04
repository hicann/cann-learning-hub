# 标准算子工程中的 msSanitizer

本目录是 `op_host + op_kernel + 算子包 + ACLNN Host` 工程。四种故障与上级独立 ASC 样例相同，但 Kernel 编译选项通过 CANN 算子工程接口注入：

```cmake
if(COMMAND add_ops_compile_options)
  add_ops_compile_options(ALL OPTIONS -sanitizer)
elseif(COMMAND npu_op_kernel_options)
  npu_op_kernel_options(ascendc_kernels ALL OPTIONS -sanitizer)
endif()
```

## 完整命令

```bash
MODE=memcheck
ROOT="$PWD"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/cann-debug-operator.XXXXXX")"
trap 'chmod -R u+rwX -- "$WORK" 2>/dev/null || true; rm -rf -- "$WORK"' EXIT
BUILD="$ROOT/build-$MODE"
INSTALL="$WORK/install-$MODE"
HOST="$ROOT/test/build-$MODE"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DLAB05_OPERATOR_FAULT_MODE="$MODE" -DLAB05_ENABLE_SANITIZER=ON
cmake --build "$BUILD" --target binary package -j
mkdir -p "$INSTALL"
"$BUILD"/custom_opp_*.run --install-path="$INSTALL"
cmake -S "$ROOT/test" -B "$HOST" -DCMAKE_SKIP_RPATH=TRUE \
  -DLAB05_CUSTOM_OPP_ROOT="$INSTALL"
cmake --build "$HOST" -j
export ASCEND_CUSTOM_OPP_PATH="$INSTALL/vendors/customize"
export LD_LIBRARY_PATH="$INSTALL/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
set +e
timeout 300 mssanitizer --tool="$MODE" "$HOST/execute_add_msanitizer" 2>&1 | tee "$BUILD/mssanitizer.log"
RC="${PIPESTATUS[0]}"
set -e
echo "mssanitizer exit_code=$RC log=$BUILD/mssanitizer.log"
```

把 `MODE` 替换为 `memcheck`、`racecheck`、`initcheck` 或 `synccheck`。预期诊断依次为 illegal
write、Potential RAW hazard、uninitialized read、Unpaired set_flag。故障程序退出码需要记录，但只有
日志同时匹配当前 MODE 的诊断签名时才能写 `EXPECTED_DIAGNOSTIC`。

> 本工程默认同时生成 910B（`ascend910b`）与 A3 系列（`ascend910_93`，Ascend910_9362 对应此编译标识）
> 两个计算单元的算子产物；如需只编某一平台，可在配置阶段追加
> `-DASCEND_COMPUTE_UNIT=ascend910b`（或 `ascend910_93`）覆盖默认值。

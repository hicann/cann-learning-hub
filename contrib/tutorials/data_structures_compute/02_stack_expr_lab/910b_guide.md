# 栈表达式求值实验 — 平台运行指南

## 环境要求

| 项目 | 要求 |
|------|------|
| 芯片 | 华为昇腾 910A3 / 910B / 310B |
| CANN SDK | 9.0.0+ (ascend-toolkit) |
| Python | 3.11+ (含 numpy) |
| 编译工具 | cmake, gcc |

## 确认目标平台

```bash
# 查看 NPU 型号
npu-smi info

# 确认 CANN 环境
source $ASCEND_HOME_PATH/set_env.sh
```

根据 `npu-smi info` 输出确定 TARGET：

| NPU 型号 | TARGET 值 | SoC 版本 |
|----------|-----------|----------|
| Ascend910 (A3) | `ascend910_93` | ascend910_93 |
| Ascend910B | `ascend910b` | ascend910b |
| Ascend310B1 | `ascend310b` | ascend310b |

## 一键运行

```bash
# 1. 设置 CANN 环境
source $ASCEND_HOME_PATH/set_env.sh

# 2. 进入实验目录
cd src/stack_expr_lab

# 3. 编译算子（根据平台设置 TARGET）
TARGET=ascend910_93 bash scripts/build_ops.sh

# 4. 设置自定义 OPP 路径（自动配置 LD_LIBRARY_PATH）
source scripts/env_custom_opp.sh

# 5. 编译 runner
bash scripts/build_runner.sh

# 6. 生成测试数据
python3 scripts/gen_data.py

# 7. 运行 benchmark
aclnn_runner/build/main_stack_benchmark data
```

## 预期输出

```
=== Stack Expression Evaluation Lab Benchmark ===
BLOCK_DIM=8 EXPR_LEN=128 TOKEN_LEN=64

[BracketMatch] time=0.0076 ms
  expr[0]: result=0 ref=0 OK
  expr[1]: result=0 ref=0 OK
  expr[2]: result=0 ref=0 OK
  expr[3]: result=2 ref=2 OK
  expr[4]: result=1 ref=1 OK
  expr[5]: result=3 ref=3 OK
  expr[6]: result=0 ref=0 OK
  expr[7]: result=0 ref=0 OK
  PASS

[SuffixEval]   time=0.0074 ms
  expr[0]: result=11.0000 ref=11.0000 err=0.0000 OK
  expr[1]: result=8.0000 ref=8.0000 err=0.0000 OK
  expr[2]: result=5.0000 ref=5.0000 err=0.0000 OK
  expr[3]: result=14.0000 ref=14.0000 err=0.0000 OK
  expr[4]: result=12.0000 ref=12.0000 err=0.0000 OK
  expr[5]: result=9.0000 ref=9.0000 err=0.0000 OK
  expr[6]: result=30.0000 ref=30.0000 err=0.0000 OK
  expr[7]: result=40.0000 ref=40.0000 err=0.0000 OK
  PASS

[InfixToPostfix] time=0.0080 ms
  expr[0]: abc*+ OK
  expr[1]: ab+c* OK
  expr[2]: abc+* OK
  expr[3]: 42*7+82/- OK
  expr[4]: ab+ OK
  expr[5]: a OK
  expr[6]: ab*cd*+ OK
  expr[7]: ab+cd-* OK
  PASS

=== Result: 3/3 PASS ===
```

## 平台差异

| 项目 | 310B | 910A3 | 910B |
|------|------|-------|------|
| AI Core 数量 | 8 | 8+ | 20+ |
| BLOCK_DIM | 8 | 8 | 20 |
| UB 大小 | ~256KB/核 | ~512KB/核 | ~512KB/核 |
| TARGET 值 | ascend310b | ascend910_93 | ascend910b |

## 常见问题

### 1. `Get regInfo failed, socVersion does not support opType`

**原因**：TARGET 设置与实际 NPU SoC 版本不匹配。

**解决**：用 `npu-smi info` 确认芯片型号，设置正确的 TARGET 值后重新编译算子。

### 2. `array is only valid for Local Memory, not allowed to have __ubuf__`

**原因**：CANN 9.0+ 中 `__ubuf__` 不能用于数组声明。

**解决**：栈数组直接声明为 Kernel 类的普通成员变量（无 `__ubuf__` 前缀），编译器自动分配到 Local Memory。

### 3. `TilingContext has no member named GetAttrValue`

**原因**：CANN 9.0 的 `gert::TilingContext` 不提供 `GetAttrValue` 接口。

**解决**：在 TilingFunc 中通过输入 shape 计算参数（`blockLength = totalLen / blockDim`），无需从 attr 获取。

## 注意事项

1. 本实验的3个算子均为标量操作（逐字符/token处理），适合 AI Core 的 Scalar 单元
2. 栈数组使用 Kernel 类成员变量，编译器自动分配到 Local Memory
3. 每个核独立处理一个表达式，天然 SPMD 并行无通信
4. 栈深度受 MAX_STACK_SIZE 限制（256），足够处理教学级表达式

# 工程部署及性能分析实验 — 平台运行指南

## 环境要求

| 项目 | 要求 |
|------|------|
| 芯片 | 华为昇腾 910B3（Atlas A2 系列） |
| CANN SDK | 9.0.0+ (ascend-toolkit) |
| Python | 3.11+ (含 numpy / torch) |
| 编译工具 | cmake, gcc |

## 确认目标平台

```bash
# 查看 NPU 型号
npu-smi info

# 确认 CANN 环境
source $ASCEND_HOME_PATH/set_env.sh
```

| NPU 型号 | TARGET 值 | SoC 版本 |
|----------|-----------|----------|
| Ascend910B3 (Atlas A2) | `ascend910b` | ascend910b |

## 一键运行

```bash
# 1. 设置 CANN 环境
source $ASCEND_HOME_PATH/set_env.sh

# 2. 进入实验目录
cd src/attention_op

# 3. 编译算子（含打包、安装到 ${HOME}/vendors/customize）
bash scripts/build_ops.sh

# 4. 设置自定义 OPP 路径（自动配置 LD_LIBRARY_PATH）
source scripts/env_custom_opp.sh

# 5. 编译 runner
bash scripts/build_runner.sh

# 6. 生成测试数据（512/1024/2048/4096）
python3 scripts/gen_data.py

# 7. 运行 benchmark（精度验证 + 计时）
aclnn_runner/build/main_attention_benchmark data 512 64
```

## 预期输出

```
[Attention] seq_len=512 dim=64 time=85.0 ms (0.4 GFLOPS)
  maxAbsErr=0.000061 maxRelErr=5.044118
  result: PASS
```

多 seq_len 实测（40 AIV 核）：

| seq_len | 512 | 1024 | 2048 | 4096 |
|---------|-----|------|------|------|
| 耗时 (ms) | 85 | 355 | 1386 | 5506 |

## 性能采集与复杂度演示

```bash
# msProf 上板采集（512，约 1-2 分钟）
bash scripts/run_profiling.sh 512 --output prof
# 报告位于 prof/prof_512/OPPROF_*/，解读 OpBasicInfo.csv / PipeUtilization.csv

# msSanitizer 内存检查（插桩运行）
mssanitizer -t memcheck aclnn_runner/build/main_attention_benchmark data 512 64 1 1
```

## 平台差异说明

| 项目 | 910B3（Atlas A2） |
|------|-------------------|
| AI Core（AIV）数量 | 40 |
| UB 大小 | ~192KB/核（实测 GetCoreMemSize=196352 B） |
| TARGET 值 | ascend910b |
| 多核切分 | 行间独立：`row = blockIdx; row < S; row += blockDim`，blockDim=40 |
| 教学约束 | seq_len ≤ 4096、dim ≤ 64（Kernel 栈帧限制 32KB） |

## 常见问题

### 1. `stack frame size (33312) exceeds limit (32768)`

**原因**：Kernel 内局部/成员数组过大，超出 AIV 栈帧限制。

**解决**：减小数组规模（如 seq_len 上限 4096 时 scoresRow 用 4096×4B=16KB），
复用数组（scoresRow 原地 softmax 变 P），避免同时声明多个大数组。

### 2. `Get regInfo failed, socVersion does not support opType`

**原因**：msOpGen 的 `-c` 参数与实际 NPU SoC 版本不匹配。

**解决**：用 `npu-smi info` 确认芯片型号，910B3 使用 `-c ai_core-ascend910b`。

### 3. msProf 报告中 `aiv_vec_ratio` 为 0

**原因**：本实验为纯标量实现（GM 标量访问 + UB 普通数组），没有向量指令，
属预期现象——这正是性能分析的切入点（瓶颈在标量 GM 访问延迟）。

**解决**：结合 `PipeUtilization.csv` 的 `aiv_time` 与 `Task Duration` 分析流水占用；
向量化改造见 08.03 实践题。

### 4. 编译环境无 msopgen / msprof / mssanitizer

**原因**：未加载 CANN 环境或安装的是精简版。

**解决**：`source $ASCEND_HOME_PATH/set_env.sh` 后确认
`ls $ASCEND_HOME_PATH/bin/ | grep -E "msopgen|msprof|msdebug|mssanitizer"`。

## 注意事项

1. 算子包安装到 `${HOME}/vendors/customize`（用户目录），无需 root 权限
2. 本实验刻意使用纯标量实现 + 多核切分，规避环境对高级特性的限制
   （MIX 任务 / TQue / UB GetValue 与大循环组合），与 02 章节 StackExprOps 模式同构
3. msDebug 需要驱动调试通道（`--full` / `/proc/debug_switch`），云环境可能不可用，
   此时以命令演示与能力清单学习为主
4. 复杂度演示为 O(S²)：seq_len 翻倍耗时约 ×4，采集 4096 单次约 5-6 秒

# 验证状态（2026-09-07）

本课程已完成从仓库根目录 `/home/user/cann-learning-hub` 对扁平化课程目录
`contrib/tutorials/qwen_ops` 的验证。验证环境为 Ascend 910B4、CANN 8.5.0、
Python 3.10.12 与 PyTorch/torch_npu 2.10.0；运行时模型通过宿主机环境变量指定，
未写入课程源码。

## 静态布局与路径验证

以下命令均从仓库根目录执行并返回零：

```bash
bash contrib/tutorials/qwen_ops/scripts/verify_layout.sh
python3 contrib/tutorials/qwen_ops/scripts/test_source_paths.py
```

`verify_layout.sh` 验证了 12 个章节、12 个 notebook、10 个算子工程的 `src/`
布局、24 个 notebook Bash 单元的 dry-run、集成脚本解析、环境设置、notebook 路径/
模型下载/源码同步及 Torch extension 配置。`test_source_paths.py` 验证两套集成入口
解析到下列扁平化章节工程，并确认环境变量覆盖可用：

- 基础版：`01_rmsnorm_baseline`、`02_rope_baseline`、`03_swiglu_baseline`、
  `04_gemm_baseline`、`05_gqa_attention_baseline`。
- 优化版：`06_rmsnorm_optimized`、`07_rope_optimized`、`08_swiglu_optimized`、
  `09_gemm_optimized`、`10_gqa_attention_optimized`。

课程源码树中不存在 `Models/`，Git 也未跟踪 `contrib/tutorials/qwen_ops/Models`
下的文件；`Models/` 是运行时目录。课程根下也不存在旧的
`Qwen2.5cann_ops/` 中间目录。

## 运行时验证

以下完整命令在扁平化路径下执行，`QWEN_OPS_MODEL_PATH` 仅为本次命令环境中的宿主机
模型覆盖：

```bash
BUILD_JOBS=1 bash contrib/tutorials/qwen_ops/scripts/build_all.sh

QWEN_OPS_MODEL_PATH=/home/user/Models/Qwen2.5-0.5B \
  bash contrib/tutorials/qwen_ops/scripts/smoke_test.sh

QWEN_OPS_MODEL_PATH=/home/user/Models/Qwen2.5-0.5B \
  bash contrib/tutorials/qwen_ops/11_baseline_ops_integration/src/Qwen2.5BaselineIntegrationExperiment/run.sh --repeat 1

QWEN_OPS_MODEL_PATH=/home/user/Models/Qwen2.5-0.5B \
  bash contrib/tutorials/qwen_ops/12_optimized_ops_integration/src/Qwen2.5OptimizedIntegrationExperiment/run.sh --repeat 1

QWEN_OPS_MODEL_PATH=/home/user/Models/Qwen2.5-0.5B \
  BUILD_JOBS=1 python3 contrib/tutorials/qwen_ops/scripts/execute_notebooks.py
```

结果如下：

- 10/10 算子工程构建成功（`ascend910b4`）。
- 10/10 `tests/test_torch_op.py` NPU 进程返回零；32/32 具体配置通过。
- `smoke_test.sh` 中嵌入的 2/2 模型集成通过；两条显式 chapter 11/12 集成命令也均返回零。
- 12/12 notebook 通过，原始 `%%bash` 单元共 24/24 通过。

当前机器未安装 Jupyter/nbconvert Python 包，因此没有依赖 Jupyter 执行器。
`scripts/execute_notebooks.py` 是依赖无关的验证路径：它直接解析 notebook JSON，从每个
章节目录执行所有原始 `%%bash` 单元；若遇到非 Bash 代码单元则失败，因而不会跳过可执行
内容。

## 五算子集成数值结果

所有集成均替换 169 个 Linear、49 个 RMSNorm、24 个 SwiGLU、24 个
Attention/RoPE/GQA 模块，且 logits 满足 `torch.allclose(atol=1e-2, rtol=1e-2)`。

| 运行 | 版本 | Native forward (ms) | Custom forward (ms) | 最大绝对误差 | 平均绝对误差 | 结果 |
|---|---|---:|---:|---:|---:|---|
| 显式 `--repeat 1` | 基础版 | 238.114961 | 50600.873694 | 0.001199722 | 0.000128466 | PASS |
| 显式 `--repeat 1` | 优化版 | 271.431053 | 2763.034353 | 0.001213074 | 0.000126100 | PASS |
| notebook `--repeat 3` 中位数 | 基础版 | 233.683090 | 50651.883135 | 0.001199722 | 0.000128466 | PASS |
| notebook `--repeat 3` 中位数 | 优化版 | 249.119934 | 1994.458712 | 0.001213074 | 0.000126100 | PASS |

计时包含 CPU 模型前向以及全部 NPU/CPU wrapper bridge copy，仅证明接入链路；
不得作为纯 kernel 或部署吞吐性能结论。

## 最终检查与可复现命令

最终文档提交前，从仓库根目录执行：

```bash
bash contrib/tutorials/qwen_ops/scripts/verify_layout.sh
python3 contrib/tutorials/qwen_ops/scripts/test_source_paths.py
git diff --check
git status --short
```

复现运行时验证时，以课程根为基础设置 CANN 环境和可选模型覆盖，不要进入已删除的
`Qwen2.5cann_ops` 目录：

```bash
cd /path/to/cann-learning-hub
export ASCEND_HOME_PATH=/path/to/Ascend/cann
export QWEN_OPS_MODEL_PATH=/path/to/Qwen2.5-0.5B
source contrib/tutorials/qwen_ops/scripts/setup_cannlab_env.sh

BUILD_JOBS=1 bash contrib/tutorials/qwen_ops/scripts/build_all.sh
QWEN_OPS_MODEL_PATH="$QWEN_OPS_MODEL_PATH" bash contrib/tutorials/qwen_ops/scripts/smoke_test.sh
QWEN_OPS_MODEL_PATH="$QWEN_OPS_MODEL_PATH" BUILD_JOBS=1 \
  python3 contrib/tutorials/qwen_ops/scripts/execute_notebooks.py
```

## 非阻塞提示

- RoPE 构建报告 `patchelf` 未安装，故未设置 RPATH；课程环境配置库路径后，两种 RoPE 的
  构建、直接运行、smoke 和 notebook 验证均通过。
- `torch_npu` 的 internal-format 警告及 PyTorch 将 `requires_grad=True` tensor 转为标量的
  报告警告不影响数值正确性。

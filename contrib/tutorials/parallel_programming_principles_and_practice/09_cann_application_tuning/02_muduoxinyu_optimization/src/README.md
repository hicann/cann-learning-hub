# MuduoXinyu A/B 工具

| 文件 | 作用 |
|---|---|
| `apply_patch.sh` | 安全应用 FlashAttention 补丁 |
| `run_ab_benchmark.sh` | 运行 Path A/Path B 冒烟和正式比较 |
| `analyze_results.py` | 检查功能并计算 mean、CV、B/A 差异 |
| `patch/muduoxinyu_flashattention_v1.patch` | FlashAttention 候选补丁 |

```bash
npu-smi info
git -C <repo> rev-parse HEAD
bash apply_patch.sh --muduo-root <repo> --check-only
bash apply_patch.sh --muduo-root <repo>
make -C <repo> npu ASCEND_PATH="${ASCEND_HOME:-${ASCEND_HOME_PATH:-${ASCEND_PATH:?请设置 ASCEND_HOME}}}" 2>&1 | tee "$PWD/build.log"
bash run_ab_benchmark.sh --muduo-root <repo> --model <model> \
  --tokenizer <tokenizer> --output-dir <new-output-dir> --build-log "$PWD/build.log"
```

构建命令必须把 stdout/stderr 通过 `tee` 保存到 build log，`run_ab_benchmark.sh` 的
`--build-log <file>` 是必填参数：binary 必须绑定到本次构建的日志与源码证据。

MuduoXinyu 必须是一次性工作副本：HEAD 精确等于课程基线，应用补丁前工作树干净，且没有旧的
`muduoXinyu` 构建产物。脚本不执行 `reset/clean/checkout`，重复应用补丁会明确拒绝。

模型与 tokenizer 不随课程仓库分发；两者必须来自同一套合法实验资产，并在运行前记录绝对路径、
大小和 SHA256。输出目录必须尚不存在。runner 按 smoke A→B、performance B→A 执行，保存：

- `run_manifest.json`：commit、基线 commit、工作树 diff SHA256（与课程 patch 精确一致）、patch SHA256、关键源码
  `src/backend/npuBackend.cpp/.hpp` SHA256、binary 的绝对路径/大小/SHA256/mtime、模型与
  tokenizer 的绝对路径/大小/SHA256、CANN 路径、ATC 版本和执行顺序；
- `commands.log`、`exit_codes.log`、`device_info.txt`：命令、退出码和设备信息；
- `build.log`、`worktree_diff.patch`：本次构建日志与当前 tracked diff 的可审计副本；
- 四份原始日志与 `result.json`：分别判断功能、数据有效性和性能收益。

**构建来源证据门禁（AB_EVIDENCE_GATE）**：运行任何 case 前，runner 先从 `apply_patch.sh`
读取基线 commit、patch SHA256 与候选 `npuBackend.cpp` SHA256 并强制校验——HEAD 必须等于
基线提交、`patch/muduoxinyu_flashattention_v1.patch` 哈希必须匹配、`npuBackend.cpp` 哈希必须
等于已验证候选；当前 tracked diff（`git diff --binary --no-ext-diff`）必须与课程 patch
byte-for-byte 一致（任何额外 tracked 改动都失败）；binary 与 `--build-log` 的 mtime 都必须
不早于 patch 涉及的最新 tracked 源码 mtime（防止明显旧构建产物通过）。`--build-log <file>`
为必填且必须非空，构建日志与 diff 会复制到输出目录。任一校验失败都会在运行前以非零退出，
绝不输出 `RUN_AB=PASS`；`build_evidence.evidence_gate=PASS` 只会在全部门禁通过后写出。
这是“构建来源证据绑定”，不声称数学上可复现的构建证明。

本实验比较的是两套实现包：Path A 为 FP32 多算子，Path B 为 FP16 FlashAttentionV4。它不是只改变
“是否融合”的纯单变量实验，结论只能归因于整体实现包。

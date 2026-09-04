# UPSTREAM：MuduoXinyu 上游信息与候选溯源

## 上游仓库

- 仓库：`https://gitcode.com/fffjjjlll1005/muduoXinyu`
- 许可证：MIT（见本目录 `LICENSE-MuduoXinyu`，Copyright (c) 2025 BNU-SYS）
- 基线提交（本课程 A/B 对比的 Path A 版本）：
  `60c6371cd30894d9896dfa979b86c6f892b6cbda`

## 候选改动溯源

- 候选归档 SHA256：
  `6c33c9a296d6b12ddc041099327fc8c443709c314f02bf7ccedb615c3c477a33`
- 候选关键文件（应用补丁后）`src/backend/npuBackend.cpp` SHA256：
  `3dbe660a5933584fc9d434100ea4dc5c1600186a494b46df96da6aea55612126`
- 当前 `patch/muduoxinyu_flashattention_v1.patch` SHA256：
  `ecb725364fb9ecfae8ee1b4530890438af88164c244fd730b778a19773e6223a`
- 补丁内容源于经过真机验证的 MuduoXinyu 候选工作树。2026-08-15 发现精简版补丁缺少
  先前暂存的前置差异，无法直接应用到声明的基线提交，因此使用本地保留的原始 Git blob
  重建为“基线提交 → 已验证候选”的自包含 6 文件补丁。重建后已在全新基线 worktree 执行
  `git apply --check`、实际应用并复核上述 `npuBackend.cpp` SHA256；该操作只验证补丁一致性，
  不等同于重新完成 NPU 性能验收。

## 补丁覆盖内容（6 个文件）

| 文件 | 变更要点 |
|--|--|
| `src/backend/npuBackend.cpp` | Path B：`prepareFlashAttention` / `attentionAllHeadsFlashDevice`；FP16 持久 KV cache；每层每步 4 次 `aclnnCast`；调用计数与 marker |
| `src/backend/npuBackend.hpp` | 新接口声明与私有成员 |
| `src/infer/infer.cpp` | Path B 初始化挂接与参数组合门禁（`--useNpuFlashAttention` 要求 `--backend npu --enableDeviceOpt`） |
| `src/model/model.cpp` | `useNpuFlashAttention` 默认 false |
| `src/model/model.hpp` | `useNpuFlashAttention` 成员 |
| `src/model/modelForwardOpt.cpp` | forward 内 A/B 分支选择（Path B 跳过 FP32 KV 写入与分解 attention） |

## 验证边界

- 本课程提供补丁、执行脚本与日志分析器；性能结论由学生本次真实运行产生；
- 不包含模型、tokenizer、构建产物或原始大日志；
- 补丁应用后 `npuBackend.cpp` 的 SHA256 可通过 `apply_patch.sh` 校验；
- `run_ab_benchmark.sh` 在运行任何 case 前执行构建来源证据门禁：从 `apply_patch.sh` 读取
  基线提交 `60c6371cd30894d9896dfa979b86c6f892b6cbda`、patch SHA256 与候选 `npuBackend.cpp`
  SHA256，并强制校验仓库 HEAD、`patch/muduoxinyu_flashattention_v1.patch` 与
  `src/backend/npuBackend.cpp` 均与上述值一致；当前 tracked diff（`git diff --binary
  --no-ext-diff`）必须与课程 patch byte-for-byte 一致；必填的 `--build-log`（非空）与 binary
  的 mtime 都必须不早于 patch 涉及的最新 tracked 源码 mtime。binary、模型、tokenizer 以绝对
  路径、大小与 SHA256 绑定进 `run_manifest.json`，构建日志与 diff 复制到输出目录。任一证据
  缺失或失配都会在运行前非零退出，不会输出 `RUN_AB=PASS`；这是构建来源证据绑定，不是数学
  上的可重复构建证明；
- 性能数据绑定候选哈希与参考环境（Ascend 910B3 / CANN 9.0.0，2026-08-07 验证），
  不保证其他环境得到相同结果。

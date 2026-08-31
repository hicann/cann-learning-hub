# 第 8 章知识测验参考答案

## 选择题

| 题号 | 答案 | 解析 |
| -- | -- | -- |
| 题目 1 | **A** | `msopgen gen -i ops.json -c ai_core-ascend910b -out <dir>` 基于原型定义生成算子工程 |
| 题目 2 | **A** | 用户目录部署：`bash xxx.run --install-path=${HOME}`，安装到 `${HOME}/vendors/customize`，无需 root |
| 题目 3 | **B** | `Task Duration(us)` 为算子总执行时间（08.02 实测 ~85ms，与 benchmark 计时一致） |
| 题目 4 | **B** | seq_len 翻倍 → 耗时约 ×4（85 → 355ms），验证 O(S²) 复杂度 |
| 题目 5 | **C** | `mssanitizer -t memcheck` 检测内存越界（08.02 注入实验检出 `illegal write of size 2`） |

## 填空题

| 题号 | 答案 | 解析 |
| -- | -- | -- |
| 题目 6 | **aclnn** | 通过 aclnn 接口（`aclnnAttentionCustom`）单算子调用：GetWorkspaceSize → 执行 |
| 题目 7 | **4·S²·D** | QKᵀ 与 AV 各 2·S²·D FLOPs，总计 4·S²·D，复杂度 O(S²) |
| 题目 8 | **调试（debug_switch）** | msDebug 依赖驱动调试通道（`--full`/`/proc/debug_switch`），CANNLab 云环境大概率不可用，可降级为命令演示 |

## 实践题评分要点

1. 向量化改造（阶段 B 内层循环 DataCopy + Cast + Muls + Add）——代码正确性
2. 回归 PASS（maxAbsErr < 1e-2）
3. msProf 前后对比分析（`aiv_vec_ratio` 从 ≈0 上升）
4. 环境限制的说明与替代处理（如实记录，不算扣分项）

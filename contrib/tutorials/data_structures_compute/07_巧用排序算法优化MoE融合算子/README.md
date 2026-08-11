# 07_巧用排序算法优化MoE融合算子

本实验围绕 MoE（Mixture of Experts）路由阶段展开，使用 Ascend C 实现并比较 TopK、QuickSort 和 HeapSort 三种专家选择策略，并串联 Token Permute / Unpermute，观察排序算法对融合路径的影响。

## 支持平台

| 平台 | 芯片 | 状态 |
|------|------|------|
| 昇腾 310B | Ascend310B1 | 已验证 |
| 昇腾 910B | Ascend910B | 已验证 |

## 章节内容

| 章节 | 主题 | 学习入口 |
|------|------|----------|
| 07.01 | MoE 路由、排序算法与 Ascend C 执行模型 | [章节导论](./07.01_chapter_intro.ipynb) |
| 07.02 | 三种路由策略与完整融合路径 | [动手实验](./07.02_moe_sort_lab.ipynb) |
| 07.03 | 选择题、填空题和实践任务 | [章节测试](./07.03_chapter_test.ipynb) |

建议按 `07.01 → 07.02 → 07.03` 的顺序学习。参考答案位于 `answer/`，可复现实验源码位于 `src/moe_sort_lab/`。

## 实验内容

1. 用固定 `top_k=2` 的路由选择问题理解 MoE 的专家分派。
2. 对比基线 TopK、完整 QuickSort 和 Heap Extract TopK 的复杂度与代价。
3. 使用排序后的 token 顺序完成 Permute，并用专家输出和路由权重完成 Unpermute。
4. 区分 AI Core kernel 时间、Host 构造排序顺序时间和端到端时间。
5. 通过 `TARGET=ascend910b` / `TARGET=ascend310b` 选择目标芯片，Host 侧将对应核数写入 tiling。

## 环境要求

- Ascend 910B 或 Ascend 310B 设备
- CANN 8.0+（建议 8.3+/9.0+）
- Python 3.8+、NumPy、CMake、GCC
- Jupyter Notebook

## 快速开始

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cd src/moe_sort_lab

# 910B：默认目标，BLOCK_DIM=16
TARGET=ascend910b bash scripts/build_ops.sh

# 310B：兼容目标，BLOCK_DIM=8
# TARGET=ascend310b bash scripts/build_ops.sh

source scripts/env_custom_opp.sh
bash scripts/build_runner.sh
python3 scripts/gen_data.py --num_tokens 1024 --num_experts 64 --hidden_size 128 --top_k 2
aclnn_runner/build/main_benchmark data 1024 128 2
aclnn_runner/build/main_full_pipeline_benchmark data 1024 128 2
```

310B 与 910B 均已完成设备编译、安装和运行验证。构建时 `TARGET=ascend910b` 使用 `BLOCK_DIM=16`，`TARGET=ascend310b` 使用 `BLOCK_DIM=8`。

## 目录结构

```text
MoE_310B4_Lite_Experiment/
├── 07.01_chapter_intro.ipynb
├── 07.02_moe_sort_lab.ipynb
├── 07.03_chapter_test.ipynb
├── 910b_guide.md
├── answer/
├── images/
└── src/moe_sort_lab/
    ├── aclnn_runner/
    ├── custom_ops/json/
    ├── custom_ops/src/
    └── scripts/
```

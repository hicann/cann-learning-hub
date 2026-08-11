# 第4章：Reduce算子与优先队列模拟堆

基于树形规约与优先队列（小根堆）的 Ascend C / CANN Reduce / TopK 算子实验。

## 支持平台

| 平台 | 芯片 | 状态 |
|------|------|------|
| 昇腾 310B | Ascend310B1 | ✅ 已验证 |
| 昇腾 910B | Ascend910B | ✅ 已验证 |
| 昇腾 A2/A3 | Atlas 推理卡 | ✅ 已适配 |

## 3个算子

| 算子 | 功能 | 核心数据结构 |
|------|------|------------|
| ReduceSumLite | 求和规约 | 树形规约 → 累加器 |
| ReduceMaxLite | 最大值规约 | 树形规约 → 比较 |
| TopKReduceLite | TopK选择 | 小根堆 |

## 核心教学目标

1. 树结构的数组化表示 (parent/left/right)
2. 递归规约 → 迭代规约的转化
3. GM → UB → Compute → CopyOut 数据流
4. 多核切分与分层规约（动态获取核数，跨芯片适配）
5. 堆结构在 TopK 中的应用

## 快速开始

```bash
# 设置环境（路径根据实际安装位置调整）
source /home/developer/Ascend/cann-9.0.0/set_env.sh
# 或
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子（默认 910B，310B 需指定 TARGET=ascend310b）
cd src/reduce_lab
TARGET=ascend910b bash scripts/build_ops.sh  # 910B
# 或
TARGET=ascend310b bash scripts/build_ops.sh  # 310B

# 编译 runner + 运行 benchmark
source scripts/env_custom_opp.sh
bash scripts/build_runner.sh
python3 scripts/gen_data.py --num_tokens 1024 --top_k 4
aclnn_runner/build/main_reduce_benchmark data 1024 4
```

## 跨芯片适配说明

算子通过 tiling 数据中的 `outputSize` / `blockDim` 字段将 BLOCK_DIM 动态传递给 kernel，避免硬编码。Host 侧的 `BLOCK_DIM` 常量会根据 `TARGET` 环境变量被 `build_ops.sh` 自动 patch（310B=8, 910B=20）。

## 目录结构

```
04_Reduce算子与优先队列模拟堆/
├── 04.01_chapter_intro.ipynb   # 章节介绍
├── 04.02_reduce_lab.ipynb      # 动手实验
├── 04.03_chapter_test.ipynb    # 课后测试
├── 910b_guide.md               # 910B 平台运行指南
├── answer/                     # 参考答案
├── images/                     # 拓扑图  
└── src/reduce_lab/             # 算子源码

# 第2章：栈的表达式求值实验

基于栈结构（LIFO）的 Ascend C / CANN 表达式求值算子实验。

## 支持平台

| 平台 | 芯片 | SoC 版本 | 状态 |
|------|------|----------|------|
| 昇腾 910A3 | Ascend910 | ascend910_93 | ✅ 已验证 |
| 昇腾 910B | Ascend910B | ascend910b | ✅ 已适配 |
| 昇腾 310B | Ascend310B1 | ascend310b | ✅ 已适配 |

## 在线体验环境

本课程已在 **CANNLab 云开发环境** 中完成运行验证：

| 环境 | 状态 | 说明 |
| -- | -- | -- |
| CANNLab 云开发环境 | ✅ 已验证 | NPU 镜像模板：`cann_9.0.0_py3.11-A2-arm`，规格：`1*NPU 910B3 16vCPUs 32GiB`，Python 内核：Python 3.11.15 |

CANNLab 环境创建与使用方法请参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。

## 3个算子

| 算子 | 功能 | 核心数据结构 |
|------|------|------------|
| BracketMatchLite | 括号匹配检验 | 顺序栈（Local Memory 数组+top） |
| SuffixEvalLite | 后缀表达式求值 | 操作数栈 |
| InfixToPostfixLite | 中缀转后缀 | 运算符栈 |

## 核心教学目标

1. 栈的 LIFO 特性在 NPU 上的直接实现（Kernel 类成员数组 + top 指针）
2. Push/Pop → 数组的 top 指针操作
3. 双栈协同 → 多 Queue 协同的智算映射
4. GM → Kernel → CopyOut 数据流
5. 多核并行：独立表达式分核处理（SPMD）

## 快速开始

```bash
# 设置环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
# 或 source $ASCEND_HOME_PATH/set_env.sh

# 确认目标平台（通过 npu-smi info 查看芯片型号）
# 910A3: TARGET=ascend910_93
# 910B:  TARGET=ascend910b
# 310B:  TARGET=ascend310b

# 编译算子
cd src/stack_expr_lab
TARGET=ascend910_93 bash scripts/build_ops.sh

# 编译 runner + 运行 benchmark
source scripts/env_custom_opp.sh
bash scripts/build_runner.sh
python3 scripts/gen_data.py
aclnn_runner/build/main_stack_benchmark data
```

## 关键技术要点

### 栈空间声明

在 Ascend C Kernel 中，栈数组声明为 Kernel 类的成员变量，编译器自动分配到 **Local Memory**：

```cpp
class KernelBracketMatchLite {
    // ...
private:
    char ubStack[MAX_STACK_SIZE];  // Local Memory 上的栈空间
    uint32_t top = 0;              // 栈顶指针
};
```

> **注意**：`__ubuf__` 关键字在 CANN 9.0+ 中仅用于指针类型转换，不能用于数组声明。
> 局部/成员数组默认在 Local Memory 中，无需额外关键字。

### Tiling 参数传递

TilingFunc 中通过输入 shape 计算每个核的处理长度：

```cpp
uint32_t blockDim = 8;
uint32_t blockLength = totalLen / blockDim;
tiling->exprLength = blockLength;  // 每核处理 exprLength 个字符
```

### 多算子工程生成

JSON 中包含多个算子时，需使用 `-op` 参数逐个生成：

```bash
msopgen gen -i ops.json -c ai_core-ascend910_93 -op BracketMatchLite -m 0
msopgen gen -i ops.json -c ai_core-ascend910_93 -op SuffixEvalLite -m 1
msopgen gen -i ops.json -c ai_core-ascend910_93 -op InfixToPostfixLite -m 1
```

## 目录结构

```
02_stack_expr_lab/
├── 02.01_chapter_intro.ipynb   # 章节介绍
├── 02.02_stack_expr_lab.ipynb  # 动手实验
├── 02.03_chapter_test.ipynb    # 课后测试
├── answer/                     # 参考答案
├── images/                     # 示意图
└── src/stack_expr_lab/         # 算子源码
```

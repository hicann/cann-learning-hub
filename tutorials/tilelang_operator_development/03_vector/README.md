# 第3章 Vector 算子：从功能实现到性能优化

前置：完成前两章及第1.4节环境配置。本目录保留六节教学 Notebook、代码、练习答案和插图。

| 小节 | 教学文档 |
|---|---|
| 3.1 章节介绍 | [打开](03.01_intro.ipynb) |
| 3.2 从一个简单的 vector 算子实现开始 | [打开](03.02_single_vector.ipynb) |
| 3.3 tiling 分核 | [打开](03.03_tiling.ipynb) |
| 3.4 双 buffer 流水排布 | [打开](03.04_pipeline.ipynb) |
| 3.5 SimdVF 与 SimtVF 对照 | [打开](03.05_execution_modes.ipynb) |
| 3.6 行 Softmax | [打开](03.06_softmax.ipynb) |

## 阅读方式

直接打开上方 Notebook 阅读正文、代码与保留的运行输出；自行执行需要按第1.4节准备950环境。折叠仅用于方便阅读，不影响代码单元的执行顺序。

## 运行方法

### 在终端运行

在已配置课程环境的容器内，从仓库根目录执行：

```bash
cd tutorials/tilelang_operator_development/03_vector
source src/env.sh
python src/run_relu_basic.py relu
python src/run_vector.py tiling
python src/run_pipeline.py
python src/run_add_modes.py
python src/run_softmax.py all
```

`src/env.sh` 默认使用当前 Python 环境中安装的课程兼容 TileLang 后端。如需指定已编译的源码目录，在 source 前将 `TILELANG_COURSE_FORK` 设置为该目录。

### 在 Notebook 中运行

逐页执行 Notebook 时，从上述环境启动 Jupyter，选择安装了 PyTorch、torch_npu 和课程后端的 Python 内核，工作目录设为本章目录。每页从上到下执行；设备程序由独立子进程运行。Markdown 中的 Python 代码块用于讲解：标明片段的代码需放入对应核函数；完整实现通过同页的 src/answer 入口运行。

### 输出文件

运行产生的缓存、日志和性能采集文件默认写入系统临时目录，不属于课程提交文件；可在 source 前设置 `COURSE_OUTPUT_DIR` 指定输出位置。CANN 安装位置不同时，可用 `CANN_ENV_SCRIPT` 指定环境脚本。

## 练习与参考代码

- `answer/` 保存完整可执行答案，运行方法为 `python answer/对应文件.py`。
- `src/exercises/03.03_tiled_exp.py` 是待补全索引的练习，未填写 TODO 时会明确退出，避免启动不完整的核函数。补全后再运行；完整答案是 `answer/03.03_tiled_exp.py`。
- `src/controlled_error.py` 是容量反例，预期在 Host 端报 248 KiB 预算错误并非零退出，用于解释 `(65536,)` 用例。
- `src/original/` 是后端参考示例，计算表达以各文件 `ref_program` 为准。默认验证 `N=2**20`；可用 `--n` 指定满足整除条件的输入，`--benchmark` 开启额外性能采集。正文的两模式对比使用 `src/add_modes.py`，不混用参考脚本。

## 性能与文件说明

正文性能表保留此前的实测记录，课程目录不包含当时的原始采集附件，不将这些数值标作读者当前环境的测量。按各节 msprof 命令可重新采集；任何未通过完整正确性校验的结果都不能用于性能结论。

`images/` 保存教学插图。

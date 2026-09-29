# 第1章 TileLang认识与环境准备

本章面向具备Python编程基础和昇腾950基本架构知识、初次学习算子开发的读者。先认识TileLang的特点与适用场景，再结合950硬件理解数据分块和软件分工，最后运行一个加法算子，尝试修改分块参数并检查结果。

## 学习顺序

建议按1.1—1.5的顺序阅读。先建立基本认识，再完成运行与练习。

| 小节 | 你将学习和完成的内容 |
| --- | --- |
| [1.1 章节介绍](01.01_chapter_intro.ipynb) | 了解课程目标、前置基础和第1—7章学习路线。 |
| [1.2 TileLang初识](01.02_tilelang_intro.ipynb) | 认识TileLang的四个特点、与AscendC的对比及应用场景。 |
| [1.3 TileLang与昇腾950](01.03_ascend_runtime.ipynb) | 认识950的计算与存储硬件，理解tile及TileLang在软件体系中的位置。 |
| [1.4 环境准备与第一个算子](01.04_first_operator.ipynb) | 准备运行环境，分清计算描述、编译与调用，运行加法算子并核对结果。 |
| [1.5 章节练习](01.05_chapter_practice.ipynb) | 完成概念自测，修改一个分块参数，预测变化并核对结果。 |

学习中需要查询SIMD接口时，可以按需查阅[附录A：SIMD API参考](../appendix_a_simd_lab/A.01_simd_lab.ipynb)。

## 阅读与运行

各节Notebook包含正文和示例代码，1.4与1.5保留对应的真实运行输出。阅读课程时，可以直接在本页对照代码与结果；练习的参考答案可以按需展开。

<p><small>线上课程环境不提供昇腾950（A5）设备。没有设备时，可以先阅读课程并完成自测和预测；实际运行需要自行准备匹配的950环境，具体方法见1.4。</small></p>

准备好环境并加载CANN环境变量后，在课程仓库根目录进入本章目录，直接运行示例脚本：

```bash
cd tutorials/tilelang_operator_development/01_overview
python src/01.04_first_operator.py
python src/01.05_grid_practice.py
```

1.4采用`N = 1024`、`TILE = 256`；1.5保持输入和计算不变，只将`TILE`改为512。两份程序都会比较全部1024项输出，通过后打印`Verification passed!`。

<p><small>程序已通过<code>tilelang.disable_cache()</code>关闭编译缓存。也可使用Notebook中的执行入口：以本章目录为工作目录，先执行保存程序的单元，再执行运行单元。</small></p>

## 文件说明

| 文件或目录 | 用途 |
| --- | --- |
| `01.01_*.ipynb`—`01.05_*.ipynb` | 本章五节课程，包含正文、示例及相应的运行输出。 |
| `src/` | 可在950环境中直接运行的示例程序。 |
| `answer/` | 练习参考答案；1.5的主要答案也可在课程页面中查看。 |
| `images/` | 课程配图。 |

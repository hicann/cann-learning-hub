# TileLang 昇腾算子开发教程

本课程介绍如何使用 **TileLang，在昇腾950（A5）上编写自定义算子**。从第一个加法算子出发，逐步学习向量计算、矩阵乘法、融合算子、调试与性能测量，最后学习借助AI Agent完成算子开发与交付。

课程共 **7章32节正文，以及附录A**。建议按第1～7章顺序学习，附录按需查阅。

## 从哪里开始

- **第一次学习：** 从[1.1「章节介绍」](01_overview/01.01_chapter_intro.ipynb)开始，了解课程目标与学习路线。
- **前置基础：** 具备Python编程基础和昇腾950基本架构知识，不要求算子开发经验；[1.3](01_overview/01.03_ascend_runtime.ipynb)会回顾Cube、Vector及常用存储位置。使用过PyTorch张量会更容易理解示例。
- **准备动手：** 按[1.4「环境准备与第一个算子」](01_overview/01.04_first_operator.ipynb)准备环境并运行加法示例，软件要求与版本说明也集中在该节。

## 章节目录

<table style="margin-left: 0; margin-right: auto; text-align: left;">
<thead>
<tr>
<th style="text-align: left; vertical-align: top;">章节</th>
<th style="text-align: left; vertical-align: top;">你将学习的内容</th>
<th style="text-align: left; vertical-align: top;">节数</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left; vertical-align: top;"><a href="01_overview/README.md">第1章：认识TileLang与环境准备</a></td>
<td style="text-align: left; vertical-align: top;">了解语言用途及其与950的关系，运行第一个算子。</td>
<td style="text-align: left; vertical-align: top;">5</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top;"><a href="02_basics/README.md">第2章：编程基础</a></td>
<td style="text-align: left; vertical-align: top;">理解kernel结构、数据搬运、编译与调用，编写简单算子。</td>
<td style="text-align: left; vertical-align: top;">5</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top;"><a href="03_vector/README.md">第3章：Vector算子</a></td>
<td style="text-align: left; vertical-align: top;">从向量计算走向分块、分核和软件流水，完成行Softmax。</td>
<td style="text-align: left; vertical-align: top;">6</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top;"><a href="04_gemm/README.md">第4章：GEMM算子</a></td>
<td style="text-align: left; vertical-align: top;">编写矩阵乘法，学习分块、流水与任务分配。</td>
<td style="text-align: left; vertical-align: top;">5</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top;"><a href="05_fusion/README.md">第5章：融合算子</a></td>
<td style="text-align: left; vertical-align: top;">连接Cube与Vector计算，实现Matmul + Bias并处理动态行数。</td>
<td style="text-align: left; vertical-align: top;">4</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top;"><a href="06_tuning/README.md">第6章：调试与性能测量</a></td>
<td style="text-align: left; vertical-align: top;">定位计算错误，使用工具测量耗时、查看结果。</td>
<td style="text-align: left; vertical-align: top;">4</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top;"><a href="07_agent/README.md">第7章：AI Agent辅助开发与交付</a></td>
<td style="text-align: left; vertical-align: top;">理解算子交付工作流，查看阶段产物并判断结果是否可靠。</td>
<td style="text-align: left; vertical-align: top;">3</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top;"><a href="appendix_a_simd_lab/README.md">附录A：SIMD API参考</a></td>
<td style="text-align: left; vertical-align: top;">查阅13个接口的功能、参数、约束和简短示例。</td>
<td style="text-align: left; vertical-align: top;">—</td>
</tr>
</tbody>
</table>

## 学习阶段与能力目标

<table style="margin-left: 0; margin-right: auto; text-align: left;">
<thead>
<tr>
<th style="text-align: left; vertical-align: top; white-space: nowrap; min-width: 5em;">阶段</th>
<th style="text-align: left; vertical-align: top; white-space: nowrap; min-width: 5em;">覆盖章节</th>
<th style="text-align: left; vertical-align: top;">学完后能做什么</th>
<th style="text-align: left; vertical-align: top;">典型任务</th>
<th style="text-align: left; vertical-align: top;">检查方式</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left; vertical-align: top; white-space: nowrap; min-width: 5em;">入门</td>
<td style="text-align: left; vertical-align: top; white-space: nowrap; min-width: 5em;">第1—2章</td>
<td style="text-align: left; vertical-align: top;">区分Host与kernel，理解数据搬运，运行并修改简单算子</td>
<td style="text-align: left; vertical-align: top;">Add、Mul、SAXPY</td>
<td style="text-align: left; vertical-align: top;">概念练习与完整数值校验</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top; white-space: nowrap; min-width: 5em;">进阶</td>
<td style="text-align: left; vertical-align: top; white-space: nowrap; min-width: 5em;">第3—5章</td>
<td style="text-align: left; vertical-align: top;">实现Vector、GEMM和融合算子，处理分块与动态行数</td>
<td style="text-align: left; vertical-align: top;">行Softmax、分块GEMM、动态M融合算子</td>
<td style="text-align: left; vertical-align: top;">索引推导、代码实践与完整数值校验</td>
</tr>
<tr>
<td style="text-align: left; vertical-align: top; white-space: nowrap; min-width: 5em;">综合应用</td>
<td style="text-align: left; vertical-align: top; white-space: nowrap; min-width: 5em;">第6—7章</td>
<td style="text-align: left; vertical-align: top;">定位计算错误，读取耗时数据，检查Agent交付材料</td>
<td style="text-align: left; vertical-align: top;">修复GEMM并测量耗时，检查Softmax案例的交付证据</td>
<td style="text-align: left; vertical-align: top;">调试说明、性能数据解读与交付检查</td>
</tr>
</tbody>
</table>

## 阅读与运行

每节只维护一份Notebook，可直接阅读正文、代码和已有运行输出。第7章通过案例产物学习开发与交付过程；附录A用于查阅API，无需执行代码。

<small>线上课程环境不提供昇腾950设备。实际运行请自行准备1.4所述的环境，在对应章节目录中按页面说明运行脚本；也可以使用Notebook，从第一个代码单元按顺序执行。不要求安装Jupyter才能运行算子。</small>

第6章的[data目录](06_tuning/data)保留了正文使用的msprof报告，可直接用表格软件查看。

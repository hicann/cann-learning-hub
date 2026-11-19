# 4.2 PyPTO 算子编程（L4-50～60，共 11 门）

> 对应 [原子课程总纲](../../README.md) 4.2 节 PyPTO 算子编程方向与 [directory.md](../../directory.md) 规划；Pro 教材与课件位于 [pypto_pro/](./pypto_pro/)。

本目录包含已完成的 PyPTO Pro 教材（L4-50～60，11 章、65 个 Notebook）和配套的 11 份章节 PPT（[pypto_pro/slides/](./pypto_pro/slides/)），以及存量的 PyPTO 基础课件。Pro 教材从异构计算与首个 Kernel 起步，介绍 Tile/Reg、SIMD/SIMT、内存与同步、融合和调优，最后以单头 SDPA、Gated-KV SDPA 串联开发过程。教学目标是理解算子的数据流，完成正确的实现，再依据测量结果优化。

适合具备 Python、张量运算和 Linux 基础的算子开发者。课程独立引入 Pro 所需概念，不要求先修 Tensor 编程课程；已有 Tensor 课程见 [PyPTO 算子开发教程](../../../../tutorials/pypto_development/README.md)。本课程不依赖 pypto-gym 的运行环境，延伸阅读中的案例用于源码对照。

## 目标环境

<table style="margin-left:0; margin-right:auto; text-align:left;">
<tr><th style="text-align:left;">项目</th><th style="text-align:left;">目标</th></tr>
<tr><td>硬件</td><td>Ascend 950PR / 950DT</td></tr>
<tr><td>CANN / PyPTO</td><td>CANN 9.2.0 及以上、包内配套 PyPTO（0.3.0 及以上）</td></tr>
<tr><td>Python</td><td>3.12以上</td></tr>
<tr><td>CANNLab</td><td>950 尝鲜体验环境（A5）</td></tr>
<tr><td>gitcode 在线 Notebook</td><td>暂不适用（950 Only）</td></tr>
</table>

环境创建参见 [CANNLab 环境体验指南](../../../../docs/CANNLab_env_experience_guide.md)。具备配套环境后，从对应章节目录打开 Notebook，使 `answer/`、`images/` 的相对路径有效；按章节概述中的导航阅读，各小节的环境准备单元需先执行。

## 在线体验环境

本课程支持以下在线体验环境：

<table style="margin-left:0; margin-right:auto; text-align:left;">
<tr><th style="text-align:left;">体验环境</th><th style="text-align:left;">镜像模板 / 版本</th><th style="text-align:left;">Python 内核</th><th style="text-align:left;">说明</th></tr>
<tr><td>CANNLab 950 尝鲜体验环境（A5）</td><td>cann_9.0.0-beta.2-py3.12-a5</td><td>Python 3.12</td><td>参考 <a href="../../../../docs/CANNLab_env_experience_guide.md">CANNLab 环境体验指南</a> 创建环境运行 Notebook；gitcode 在线 Notebook 暂不适用（950 Only）</td></tr>
</table>

## Pro 章节导航

<table style="margin-left:0; margin-right:auto; text-align:left;">
<tr><th style="text-align:left;">课程编号</th><th style="text-align:left;">章节入口</th><th style="text-align:left;">主要内容</th></tr>
<tr><td>L4-50</td><td><a href="./pypto_pro/01_pypto_pro_introduction/01.01_chapter_intro.ipynb">01 快速入门</a></td><td>异构计算、950 架构、执行域与首个 Kernel</td></tr>
<tr><td>L4-51</td><td><a href="./pypto_pro/02_pypto_pro_simd_programming_model/02.01_chapter_intro.ipynb">02 SIMD 编程模型</a></td><td>核函数、Tensor/Tile、内存搬运、缓冲、同步与 JIT</td></tr>
<tr><td>L4-52</td><td><a href="./pypto_pro/03_pypto_pro_simd_vector_operator/03.01_chapter_intro.ipynb">03 SIMD 矢量算子</a></td><td>Reg/Tile 计算、归约、广播与模式选择</td></tr>
<tr><td>L4-53</td><td><a href="./pypto_pro/04_pypto_pro_simd_matmul_operator/04.01_chapter_intro.ipynb">04 SIMD 矩阵算子</a></td><td>Cube、分形布局、多核与动态尾块</td></tr>
<tr><td>L4-54</td><td><a href="./pypto_pro/05_pypto_pro_simd_fused_operator/05.01_chapter_intro.ipynb">05 SIMD 融合算子</a></td><td>Cube/Vector 数据流、融合模式、同步与缓冲</td></tr>
<tr><td>L4-55</td><td><a href="./pypto_pro/06_pypto_pro_simt_programming_model/06.01_chapter_intro.ipynb">06 SIMT 编程模型</a></td><td>线程层次、索引、启动、共享数据与分支</td></tr>
<tr><td>L4-56</td><td><a href="./pypto_pro/07_pypto_pro_simt_operator/07.01_chapter_intro.ipynb">07 SIMT 算子</a></td><td>Gather、原子操作与调试</td></tr>
<tr><td>L4-57</td><td><a href="./pypto_pro/08_pypto_pro_debug_and_optimization/08.01_chapter_intro.ipynb">08 调试与优化</a></td><td>功能定位、sanitizer、性能分析与流水</td></tr>
<tr><td>L4-58</td><td><a href="./pypto_pro/09_pypto_pro_simd_simt_hybrid/09.01_chapter_intro.ipynb">09 SIMD/SIMT 协同</a></td><td>分段设计、不规则选择与协同开销</td></tr>
<tr><td>L4-59</td><td><a href="./pypto_pro/10_pypto_pro_sdpa_practice/10.01_chapter_intro.ipynb">10 SDPA 实战</a></td><td>单头规格、online softmax、切块与精度验收</td></tr>
<tr><td>L4-60</td><td><a href="./pypto_pro/11_pypto_pro_gated_kv_sdpa_practice/11.01_chapter_intro.ipynb">11 Gated-KV SDPA 实战</a></td><td>确定性块选择、原始位置 causal mask 与稀疏/稠密协同</td></tr>
</table>

## 课件

Pro 章节 PPT（一门一份，共 11 份）位于 [pypto_pro/slides/](./pypto_pro/slides/)，命名与章目录一致：

<table style="margin-left:0; margin-right:auto; text-align:left;">
<tr><th style="text-align:left;">序号</th><th style="text-align:left;">主题</th><th style="text-align:left;">主要内容</th><th style="text-align:left;">课件</th></tr>
<tr><td>01</td><td>快速入门</td><td>异构计算、950 架构、执行域与首个 Kernel</td><td><a href="./pypto_pro/slides/01_pypto_pro_introduction.pptx">01_pypto_pro_introduction.pptx</a></td></tr>
<tr><td>02</td><td>SIMD 编程模型</td><td>核函数、Tensor/Tile、内存搬运、缓冲、同步与 JIT</td><td><a href="./pypto_pro/slides/02_pypto_pro_simd_programming_model.pptx">02_pypto_pro_simd_programming_model.pptx</a></td></tr>
<tr><td>03</td><td>SIMD 矢量算子</td><td>Reg/Tile 计算、归约、广播与模式选择</td><td><a href="./pypto_pro/slides/03_pypto_pro_simd_vector_operator.pptx">03_pypto_pro_simd_vector_operator.pptx</a></td></tr>
<tr><td>04</td><td>SIMD 矩阵算子</td><td>Cube、分形布局、多核与动态尾块</td><td><a href="./pypto_pro/slides/04_pypto_pro_simd_matmul_operator.pptx">04_pypto_pro_simd_matmul_operator.pptx</a></td></tr>
<tr><td>05</td><td>SIMD 融合算子</td><td>Cube/Vector 数据流、融合模式、同步与缓冲</td><td><a href="./pypto_pro/slides/05_pypto_pro_simd_fused_operator.pptx">05_pypto_pro_simd_fused_operator.pptx</a></td></tr>
<tr><td>06</td><td>SIMT 编程模型</td><td>线程层次、索引、启动、共享数据与分支</td><td><a href="./pypto_pro/slides/06_pypto_pro_simt_programming_model.pptx">06_pypto_pro_simt_programming_model.pptx</a></td></tr>
<tr><td>07</td><td>SIMT 算子</td><td>Gather、原子操作与调试</td><td><a href="./pypto_pro/slides/07_pypto_pro_simt_operator.pptx">07_pypto_pro_simt_operator.pptx</a></td></tr>
<tr><td>08</td><td>调试与优化</td><td>功能定位、sanitizer、性能分析与流水</td><td><a href="./pypto_pro/slides/08_pypto_pro_debug_and_optimization.pptx">08_pypto_pro_debug_and_optimization.pptx</a></td></tr>
<tr><td>09</td><td>SIMD/SIMT 协同</td><td>分段设计、不规则选择与协同开销</td><td><a href="./pypto_pro/slides/09_pypto_pro_simd_simt_hybrid.pptx">09_pypto_pro_simd_simt_hybrid.pptx</a></td></tr>
<tr><td>10</td><td>SDPA 实战</td><td>单头规格、online softmax、切块与精度验收</td><td><a href="./pypto_pro/slides/10_pypto_pro_sdpa_practice.pptx">10_pypto_pro_sdpa_practice.pptx</a></td></tr>
<tr><td>11</td><td>Gated-KV SDPA 实战</td><td>确定性块选择、原始位置 causal mask 与稀疏/稠密协同</td><td><a href="./pypto_pro/slides/11_pypto_pro_gated_kv_sdpa_practice.pptx">11_pypto_pro_gated_kv_sdpa_practice.pptx</a></td></tr>
</table>

- 存量基础课件：[PyPTO 基础](./01_pypto_basic_3.1.pptx)（PyPTO Tensor 编程基础课件素材）。

# 第6章 算子调试与性能测量

本章介绍如何定位数值错误，以及如何使用do_bench和msprof取得算子耗时。

<p class="course-note"><small>线上课程环境不提供昇腾950（A5）设备。每节Notebook包含正文、完整程序和已有实测结果，可直接对照阅读；自行运行请按1.4准备环境。</small></p>

- [6.1 算子调试与性能测量](06.01_introduction.ipynb)：根据问题找到对应的学习内容。
- [6.2 功能调试：输出不一致时怎样逐步定位](06.02_debugging.ipynb)：沿数据路径定位差异，修复后比较完整输出。
- [6.3 性能测量](06.03_measure_and_tune.ipynb)：使用do_bench和msprof，读取耗时与调用记录。
- [6.4 综合练习：修复并测量一个GEMM](06.04_capstone.ipynb)：依次完成修复、do_bench计时、msprof报告阅读三个任务。

## 阅读与自行运行

正文保留关键代码和结果解释。完整程序、Notebook运行辅助代码和采集日志可展开查看；耗时与CSV结果表保持可见。

**使用终端：** 进入本章目录后，运行对应小节给出的Python脚本或msprof命令。

**使用Notebook：** 以本章目录为工作目录，从准备单元开始顺序执行。代码折叠只影响显示，不改变执行顺序。

<p class="course-note"><small><code>%%writefile</code>保存代码，<code>%%writefile -a</code>追加代码，后续运行单元才执行程序；重新运行时仍需从头按顺序执行。程序已调用<code>tilelang.disable_cache()</code>关闭编译缓存。</small></p>

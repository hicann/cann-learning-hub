![ACL Python 算子库调用](./images/readme_cover.png)

---

# ACL Python 算子库调用（CANN 模块 4.2）

## 课程简介

本实验对应 CANN 教学体系模块 4.2（ACL Python 算子库调用），使用 **pyACL 直接算子 API**：GEMM 调用 `acl.blas.gemm_ex`，SpMV 调用
`acl.op.execute_v2("SparseTensorDenseMatMul", ...)`。ATC 先生成适配目标设备的单算子 OM，
运行时通过 `acl.op.set_model_dir` 查找对应实现；程序不使用模型推理接口加载并执行整图。

## 适用人群与前置要求

面向具备 Python、NumPy、矩阵乘和 COO 稀疏矩阵基础的学习者；建议先了解 ACL Runtime 的
初始化、设备内存和 Stream 概念。

## 学习目标

- 完成 ACL 初始化、设备内存、数据搬运、异步算子提交、同步和资源释放；
- 理解 GEMM 的稠密矩阵合同与 SpMV 的 COO 稀疏合同；
- 用 NumPy reference 检查 NPU 输出。

## 课程支持的硬件产品与已验证的在线体验环境

<table style="text-align:left; margin-left:0;">
<tr><th>项目</th><th>说明</th></tr>
<tr><td>支持硬件</td><td>Atlas A3（SoC Ascend910_9362）；本轮已实测验证，不代表所有 A3 子型号；历史记录另有 Ascend 910B3</td></tr>
<tr><td>CANNLab 环境</td><td>CANN 9.0，镜像模板 <code>cann_9.0.0-py3.11-A3-arm-20260829</code></td></tr>
<tr><td>Notebook 内核</td><td><code>Python 3.11.4 (CANN)</code>，kernelspec 为 <code>python3</code></td></tr>
<tr><td>CANNLab 指南</td><td><a href="../../../../docs/CANNLab_env_experience_guide.md">CANNLab 环境体验指南</a></td></tr>
<tr><td>GitCode 在线 Notebook</td><td>-</td></tr>
<tr><td>实验验证情况</td><td>2026-09-02 A3 验证：GEMM 与 SpMV 均完成 Device 执行并通过 NumPy 校验；SpMV 为 AI CPU/tf_kernel 能力路径（<code>aicore_accelerated=false</code>），不是 AI Core 加速</td></tr>
</table>

## 执行边界

<table style="text-align:left; margin-left:0;">
<tr><th>算子</th><th>实际调用入口</th><th>CANN 9.0 实测边界</th></tr>
<tr><td>GEMM</td><td><code>acl.blas.gemm_ex</code></td><td>设备执行路径；具体 Kernel 类别以当前运行证据为准</td></tr>
<tr><td>SpMV</td><td><code>acl.op.execute_v2("SparseTensorDenseMatMul", ...)</code></td><td>AI CPU/tf_kernel 能力验证，<code>aicore_accelerated=false</code>，不得写成 AI Core 加速</td></tr>
</table>

两条路径都先用 ATC 生成单算子 OM，再通过 `acl.op.set_model_dir` 注册模型目录。OM 文件存在、
`acl` 可以导入或历史日志通过，都不能单独作为当前 NPU 验收结论。

## 课程章节目录

<table style="text-align:left; margin-left:0;">
<tr><th>Notebook</th><th>内容</th><th>在线体验</th></tr>
<tr><td><a href="./05.01_chapter_intro.ipynb">05.01_chapter_intro.ipynb</a></td><td>API 路线、实验环境说明</td><td>-</td></tr>
<tr><td><a href="./05.02_gemm.ipynb">05.02_gemm.ipynb</a></td><td><code>acl.blas.gemm_ex</code></td><td>-</td></tr>
<tr><td><a href="./05.03_spmv.ipynb">05.03_spmv.ipynb</a></td><td><code>acl.op.execute_v2</code></td><td>-</td></tr>
<tr><td><a href="./05.04_chapter_test.ipynb">05.04_chapter_test.ipynb</a></td><td>客观题与简单/中等/困难三档实践</td><td>-</td></tr>
</table>

进入 `本目录` 后启动 Jupyter，并按顺序运行 Notebook。

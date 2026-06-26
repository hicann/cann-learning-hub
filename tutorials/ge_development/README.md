# GE 图引擎开发系列教程

本教程面向 GE 图引擎使用者，包括模型部署工程师、推理应用开发者、算子/Pass 扩展开发者。内容围绕 GE 对外暴露的能力与接口展开，覆盖 GE 基础概念、AscendIR 构图、ATC 离线编译、ACL 推理部署、GeSession 在线执行、动态 Shape、自定义算子入图、自定义融合 Pass、执行优化、故障定位与框架生态对接，帮助开发者掌握"如何用好 GE"。

教程按章节划分，每个章节均包含以下内容：
- Notebooks：包含课程知识点与练习题，适用于自主学习或讲师引导式教学，可在 gitcode 提供的轻量级 notebook 上运行。也可自行搭建 jupyter lab，在本地环境执行使用。
- answer：章节练习参考答案。

>- **注意：**
>- 本教程面向 GE 用户侧能力学习，重点关注模型编译、执行、部署与扩展开发实践，不深入展开 GE 内部实现细节。
>- 课程内容以 CANN/昇腾社区公开文档、GE 图开发指南、ATC 使用指南、ACL 推理开发文档及相关示例为参考。
>- 本教程当前仅针对 Atlas A2 系列产品进行验证，其它产品使用存在问题，欢迎开发者提出 issue 或 PR 进行共建。环境要求：CANN 8.5.0 及以上，Linux 已部署 CANN 开发环境（参考 [CANN 下载页面](https://www.hiascend.com/cann/download)）；「基础概念入门」无需 NPU，其余动手实践需 NPU 或昇腾云环境。

## 能力分层设计

本课程围绕"用好 GE"的能力成长路径设计，按初级、中级、高级三个层级逐级递进。各层级覆盖章节、核心能力目标与考核方式如下，作为后续不同层级能力认证考试的依据：

<div align="left">
<table style="margin-left: 0; margin-right: auto;">
<thead>
<tr><th>阶段</th><th>章节范围</th><th>能力目标</th><th>考核方式</th></tr>
</thead>
<tbody>
<tr><td>初级（快速入门）</td><td>第 1–2 章</td><td>理解 GE 全栈定位、两大职责与 AscendIR 构图概念；掌握离线（ATC+ACL）与在线（GeSession）两条推理路径，能跑通端到端推理。</td><td>选择/判断题检验概念 + 跑通一次离线/在线推理的综合编程实践。</td></tr>
<tr><td>中级（进阶开发）</td><td>第 3–4 章</td><td>掌握图编译配置与产物（融合、精度、OM、模型缓存），自定义算子入图与融合 Pass；掌握静态/动态 Shape 执行流程与优化技术。</td><td>编译配置与执行优化的综合编程实践。</td></tr>
<tr><td>高级（高阶应用）</td><td>第 5 章及以后</td><td>掌握 GE 对接 PyTorch（TorchAir）、TensorFlow 等框架，掌握常见问题定位方法。</td><td>框架接入与问题定位综合实践。</td></tr>
</tbody>
</table>
</div>

## GE 图引擎开发系列（快速入门）

### 第一章：基础概念入门

<div align="left">
<table style="margin-left: 0; margin-right: auto;">
<thead>
<tr><th>Notebook</th><th>Link</th><th>状态</th></tr>
</thead>
<tbody>
<tr><td>1.1 章节介绍</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/01_basic_concepts/01.01_chapter_intro.ipynb&amp;npuCnt=2">在线体验</a></td><td>✅ 已发布</td></tr>
<tr><td>1.2 GE 定位与核心概念</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/01_basic_concepts/01.02_ge_overview.ipynb&amp;npuCnt=2">在线体验</a></td><td>✅ 已发布</td></tr>
<tr><td>1.3 AscendIR 基础概念</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/01_basic_concepts/01.03_ascend_ir.ipynb&amp;npuCnt=2">在线体验</a></td><td>✅ 已发布</td></tr>
<tr><td>1.4 章节练习</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/01_basic_concepts/01.04_chapter_practice.ipynb&amp;npuCnt=2">在线体验</a></td><td>✅ 已发布</td></tr>
</tbody>
</table>
</div>

### 第二章：推理流程介绍

<div align="left">
<table style="margin-left: 0; margin-right: auto;">
<thead>
<tr><th>Notebook</th><th>Link</th><th>状态</th></tr>
</thead>
<tbody>
<tr><td>2.1 章节介绍</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_inference_workflow/02.01_chapter_intro.ipynb&amp;npuCnt=2">在线体验</a></td><td>✅ 已发布</td></tr>
<tr><td>2.2 离线推理流程：ATC 编译与 ACL 推理</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_inference_workflow/02.02_offline_inference.ipynb&amp;npuCnt=2">在线体验</a></td><td>✅ 已发布</td></tr>
<tr><td>2.3 在线执行流程：GeSession 构图与执行</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_inference_workflow/02.03_online_execution.ipynb&amp;npuCnt=2">在线体验</a></td><td>✅ 已发布</td></tr>
<tr><td>2.4 章节练习</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_inference_workflow/02.04_chapter_practice.ipynb&amp;npuCnt=2">在线体验</a></td><td>✅ 已发布</td></tr>
</tbody>
</table>
</div>

## GE 图引擎开发系列（进阶开发）

### 第三章：图编译

<div align="left">
<table style="margin-left: 0; margin-right: auto;">
<thead>
<tr><th>Notebook</th><th>Link</th><th>状态</th></tr>
</thead>
<tbody>
<tr><td>3.1 章节介绍</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/03_graph_compilation/03.01_chapter_intro.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>3.2 图的构建与输入：AscendIR 构图与 Parser 解析</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/03_graph_compilation/03.02_graph_build_and_input.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>3.3 编译配置：融合、精度、buffer、stream 等选项</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/03_graph_compilation/03.03_compilation_config.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>3.4 编译产物：OM 结构、外置权重、SO in OM、模型缓存</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/03_graph_compilation/03.04_compilation_output.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>3.5 图编译扩展能力：自定义算子入图</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/03_graph_compilation/03.05_custom_op_integration.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>3.6 图编译扩展能力：自定义融合 Pass</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/03_graph_compilation/03.06_custom_fusion_pass.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>3.7 章节练习</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/03_graph_compilation/03.07_chapter_practice.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
</tbody>
</table>
</div>

### 第四章：模型执行与优化

<div align="left">
<table style="margin-left: 0; margin-right: auto;">
<thead>
<tr><th>Notebook</th><th>Link</th><th>状态</th></tr>
</thead>
<tbody>
<tr><td>4.1 章节介绍</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/04_model_execution_optimization/04.01_chapter_intro.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>4.2 静态 Shape 执行流程：整图下沉</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/04_model_execution_optimization/04.02_static_shape_execution.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>4.3 动态 Shape 执行流程：Host 调度</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/04_model_execution_optimization/04.03_dynamic_shape_execution.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>4.4 静态 Shape 执行优化技术</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/04_model_execution_optimization/04.04_static_shape_optimization.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>4.5 动态 Shape 执行优化技术</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/04_model_execution_optimization/04.05_dynamic_shape_optimization.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>4.6 章节练习</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/02_advanced_development/04_model_execution_optimization/04.06_chapter_practice.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
</tbody>
</table>
</div>

## GE 图引擎开发系列（高阶应用）

### 第五章：实践与问题定位

<div align="left">
<table style="margin-left: 0; margin-right: auto;">
<thead>
<tr><th>Notebook</th><th>Link</th><th>状态</th></tr>
</thead>
<tbody>
<tr><td>5.1 章节介绍</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/03_advanced_application/05_practice_and_troubleshooting/05.01_chapter_intro.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>5.2 GE 对接 PyTorch：TorchAir 图模式</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/03_advanced_application/05_practice_and_troubleshooting/05.02_pytorch_integration.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>5.3 GE 对接 TensorFlow</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/03_advanced_application/05_practice_and_troubleshooting/05.03_tensorflow_integration.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>5.4 常见问题定位方法</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/03_advanced_application/05_practice_and_troubleshooting/05.04_troubleshooting.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
<tr><td>5.5 章节练习</td><td><a href="https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&amp;repoUrl=https://gitcode.com/cann/cann-learning-hub.git&amp;ttl=120&amp;diskSize=40Gi&amp;path=tutorials/ge_development&amp;scanFilePath=tutorials/ge_development/03_advanced_application/05_practice_and_troubleshooting/05.05_chapter_practice.ipynb&amp;npuCnt=2">在线体验</a></td><td>🚧 开发中</td></tr>
</tbody>
</table>
</div>

# Ascend C 算子开发系列教程
本教程将带你学习面向昇腾 NPU 的Ascend C高性能算子开发全流程教程，包含 Tiling 模板化编程、算子调试、性能优化等核心内容。

教程按章节划分，每个章节均包含以下内容：
- Notebooks：包含课程知识点与练习题，适用于自主学习或讲师引导式教学，可在 gitcode提供的轻量级 notebook 上运行。也可自行搭建jupyter lab，在本地环境执行使用。
- SRC：包含课程中所有的源码，供开发者自行下载及修改。
- PPT（建设中）：包含各课时的授课内容。  

>- **注意：**
>- 本教程当前仅针对Atlas A2系列产品进行验证，其它产品使用存在问题，欢迎开发者提出issue或PR进行共建。

## Notebooks

### Ascend C：面向昇腾NPU的高性能算子开发语言

| Notebook | Link | 状态 |
|--|--|--|
| 1.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/01_basic_overview/01.01_chapter_intro.ipynb) | ✅ 已发布 |
| 1.2 人工智能与算子基础 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/01_basic_overview/01.02_ai_and_operator_basics.ipynb) | ✅ 已发布 |
| 1.3 CANN架构与昇腾NPU原理 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb) | ✅ 已发布 |
| 1.4 Ascend C算子开发的基本概念 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/01_basic_overview/01.04_ascend_c_op_dev_basic_concepts.ipynb) | ✅ 已发布 |
| 1.5 章节练习 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/01_basic_overview/01.05_chapter_practice.ipynb) | ✅ 已发布 |

### Ascend C算子开发基础知识

| Notebook | Link | 状态 |
|--|--|--|
| 2.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/02_AscendC_basic/02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 2.2 Ascend C的"Hello world" | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/02_AscendC_basic/02.02_HelloWorld.ipynb) | ✅ 已发布 |
| 2.3 AscendC编程范式介绍与API介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/02_AscendC_basic/02.03_ascendc_programming_paradigm_and_api_introduction.ipynb) | ✅ 已发布 |
| 2.4 基于Add算子的核函数介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/02_AscendC_basic/02.04_introduction_to_kernel_functions_based_on_add_operator.ipynb) | ✅ 已发布 |
| 2.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/02_AscendC_basic/02.05_chapter_test.ipynb) | ✅ 已发布 |

### Vector算子开发

| Notebook | Link | 状态 |
|--|--|--|
| 3.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/03_intermediate_vector_operator_development/03.01_chapter_intro.ipynb) | ✅ 已发布 |
| 3.2 工程化算子开发介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/03_intermediate_vector_operator_development/03.02_operator_engineering_intro.ipynb) | ✅ 已发布 |
| 3.3 aclnn和pybind调用 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/03_intermediate_vector_operator_development/03.03_acl_pybind_call.ipynb) | ✅ 已发布 |
| 3.4 泛化Tiling设计 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/03_intermediate_vector_operator_development/03.04_generalized_tiling_design.ipynb) | ✅ 已发布 |
| 3.5 tiling模板化编程、属性、Tbuf、workspace的使用 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/03_intermediate_vector_operator_development/03.05_tiling_template_attr_tbuf_workspace.ipynb) | ✅ 已发布 |
| 3.6 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/03_intermediate_vector_operator_development/03.06_chapter_practice.ipynb) | ✅ 已发布 |

### 矩阵算子开发

| Notebook | Link | 状态 |
|--|--|--|
| 4.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/04_matmul_basic/04.01_chapter_intro.ipynb) | ✅ 已发布 |
| 4.2 矩阵乘法介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/04_matmul_basic/04.02_matrix_multiplication_introduction.ipynb) | ✅ 已发布 |
| 4.3 基于高阶API的矩阵乘算子开发 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/04_matmul_basic/04.03_matrix_multiplication_operator_development_with_high_level_api.ipynb) | ✅ 已发布 |
| 4.4 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/04_matmul_basic/04.04_chapter_test.ipynb) | ✅ 已发布 |

### 融合算子开发

| Notebook | Link | 状态 |
|--|--|--|
| 5.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/05_fused_operator_development/05.01_chapter_intro.ipynb) | ✅ 已发布 |
| 5.2 融合算子概念介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/05_fused_operator_development/05.02_fused_operator_concept_intro.ipynb) | ✅ 已发布 |
| 5.3 VV融合算子开发 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/05_fused_operator_development/05.03_vv_fused_operator_development.ipynb) | ✅ 已发布 |
| 5.4 CV融合算子开发 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/05_fused_operator_development/05.04_cv_fused_operator_development.ipynb) | ✅ 已发布 |
| 5.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/05_fused_operator_development/05.05_chapter_practice.ipynb) | ✅ 已发布 |

### CANN开源算子仓：昇腾NPU异构计算全场景算子开发、验证与贡献一站式平台

| Notebook | Link | 状态 |
|--|--|--|
| 6.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.01_chapter_intro.ipynb) | ✅ 已发布 |
| 6.2 开源仓介绍及验证 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.02_opensource_repo_intro_and_verification.ipynb) | ✅ 已发布 |
| 6.3 基于开源仓的算子开发 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.03_operator_development_based_on_opensource_repo.ipynb) | ✅ 已发布 |
| 6.4 UT的编写及验证 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.04_ut_st_writing_and_verification.ipynb) | ✅ 已发布 |
| 6.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/06_opensource_repo_operator_intro_and_contribution/06.05_chapter_practice.ipynb) | ✅ 已发布 |

### Ascend C算子功能调试

| Notebook | Link | 状态 |
|--|--|--|
| 7.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/07_Troubleshooting/07.01_chapter_intro.ipynb) | ✅ 已发布 |
| 7.2 CPU域孪生调试 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/07_Troubleshooting/07.02_CPU_Debugging_Overview.ipynb) | ✅ 已发布 |
| 7.3 NPU域上板调试 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/07_Troubleshooting/07.03_NPU_On-Board_Debugging.ipynb) | ✅ 已发布 |
| 7.4 Ascend C算子开发典型问题 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/07_Troubleshooting/07.04_Typical_Issues_in_AscendC_Operator_Development.ipynb) | ✅ 已发布 |
| 7.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/07_Troubleshooting/07.05_chapter_test.ipynb) | ✅ 已发布 |


### Ascend C算子性能优化

| Notebook | Link | 状态 |
|--|--|--|
| 8.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/08_performance_optimization/08.01_chapter_intro.ipynb) | ✅ 已发布 |
| 8.2 msProf工具使用技巧 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/08_performance_optimization/08.02_profiling_tool_usage.ipynb) | ✅ 已发布 |
| 8.3 仿真分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/08_performance_optimization/08.03_simulation_analysis.ipynb) | ✅ 已发布 |
| 8.4 Ascend C算子性能优化典型示例 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/08_performance_optimization/08.04_ascendc_op_perf_optimization_demo.ipynb) | ✅ 已发布 |
| 8.5 章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?imageId=online-ubuntu22-cann8.5-python3.11-jupyter%3Av1.0.1&repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development&scanFilePath=tutorials/ascendc_operator_development/08_performance_optimization/08.05_chapter_practice.ipynb) | ✅ 已发布 |

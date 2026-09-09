[![数据结构与算法设计](./images/READMEImage.png)](https://e.huawei.com/cn/talent/learning/#/zone?customizedZoneId=qS4mF6lijbGojmzxSlKlhX4ccow)

------

## 课程整体简介

本课程以常见数据结构为主线，将数组、栈、队列、优先队列、图、链表和排序算法等基础知识，与 Ascend C 算子开发及昇腾工程实践相结合。课程通过 12 个循序渐进的实验，引导学习者理解数据结构在异构计算中的组织方式、访问特征和工程价值，并将抽象的数据结构知识转化为可编译、可运行、可验证的算子程序。

课程前半部分从开发环境、VectorAdd 算子和 Tiling 数据切分入手，随后通过括号匹配、表达式求值和双缓冲数据流水线，掌握栈、队列及生产者—消费者模型的实际应用。课程中段进一步学习 Reduce 算子与优先队列、图邻接表与稀疏表示、链表与连续存储，以及排序算法在数据重排任务中的应用。课程后半部分通过工程部署、装填因子与线性探测、树形任务队列、位图集合交和 MoE 融合算子，进一步理解数据结构与访存、流水线调度及算子性能之间的关系，形成从数据结构设计到 Ascend C 工程交付的完整实践路径。

## 课程适用学习人群

本课程适合已经学习过 C/C++ 程序设计和基础数据结构，希望通过可运行实验入门 Ascend C 算子开发的学习者。开始前建议具备以下基础：

- 理解一维数组、队列、整数除法和左闭右开区间；
- 能够使用 Linux 命令行、CMake 和 Jupyter Notebook；
- 能够阅读简单的 C/C++ 代码；
- 无需具备 Ascend C 自定义算子开发经验。

## 整体学习目标

完成本课程后，学习者应能够：

1. 理解数组、栈、队列、优先队列、图、链表和排序算法的核心特性，并分析不同数据结构的存储方式、访问模式和适用场景。
2. 建立数据结构与昇腾计算任务之间的联系，能够根据计算需求选择合适的数据组织形式和处理方法。
3. 掌握 Ascend C 算子开发的基础流程，能够完成开发环境检查、Kernel 直调工程构建、编译运行和结果验证。
4. 根据数据规模、核数和 Tile 数量设计基本的 Tiling 方案，理解多核切分、32 Byte 对齐、UB 空间占用和数据覆盖等约束。
5. 使用 TPipe、TQue、GlobalTensor 和 LocalTensor 组织数据搬运与计算，并利用队列和双缓冲机制实现计算与搬运流水。
6. 将栈、优先队列、邻接表、稀疏表示、链式结构和排序等方法应用于表达式处理、Reduce、图数据组织和数据重排等典型任务。
7. 使用非法参数检查、边界用例和 CPU Golden 对算子进行正确性验证，并能够定位常见的数据切分、内存访问和流水调度问题。
8. 完成算子生成、Host 与 Kernel 联调、打包部署和性能数据采集，能够阅读性能报告并对算子的执行效率进行初步分析。
9. 形成从问题分析、数据结构选择、算法设计、算子实现到测试部署的完整工程实践能力。

## 课程支持的硬件产品

| 硬件产品 | 验证状态 |
| --- | --- |
| Atlas A2 | ✅ 已验证 |

## 在线体验环境

- gitcode 在线体验 Notebook
- CANNLab 云开发环境
  - NPU 镜像模板：`cann_9.0.0_py3.11-A2-arm`
  - 规格：`1*NPU 910B3 16vCPUs 32GiB`
  - Python 内核：Python 3.11.4

CANNLab 环境创建与使用方法请参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。


## 课程章节目录

### 第一章：基础实操

| Notebook | Link | 状态 |
| --- | --- | --- |
| 01.01 章节介绍：基础实操 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/01_basic_operations/01.01_chapter_intro.ipynb) | ✅ 已发布 |
| 01.02 环境与工程一键启动 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/01_basic_operations/01.02_environment_and_project.ipynb) | ✅ 已发布 |
| 01.03 VectorAdd 算子实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/01_basic_operations/01.03_vector_add_operator.ipynb) | ✅ 已发布 |
| 01.04 Tiling 可视化交互实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/01_basic_operations/01.04_tiling_visualization.ipynb) | ✅ 已发布 |
| 01.05 章节实践：独立完成 VectorAdd | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/01_basic_operations/01.05_chapter_test.ipynb) | ✅ 已发布 |

### 第二章：括号匹配及表达式求值

| Notebook | Link | 状态 |
| --- | --- | --- |
| 02.01 章节介绍：括号匹配及表达式求值 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/02_stack_expr_lab/02.01_chapter_intro.ipynb) | ✅ 已发布 |
| 02.02 括号匹配及表达式求值动手实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/02_stack_expr_lab/02.02_stack_expr_lab.ipynb) | ✅ 已发布 |
| 02.03 章节实践与测试：括号匹配及表达式求值 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/02_stack_expr_lab/02.03_chapter_test.ipynb) | ✅ 已发布 |

### 第三章：双缓冲数据流水线

| Notebook | Link | 状态 |
| --- | --- | --- |
| 03.01 章节介绍：双缓冲数据流水线 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/03_double_buffer_pipeline/03.01_chapter_intro.ipynb) | ✅ 已发布 |
| 03.02 TQue 队列基础 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/03_double_buffer_pipeline/03.02_queue_basics.ipynb) | ✅ 已发布 |
| 03.03 双缓冲 VectorAdd | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/03_double_buffer_pipeline/03.03_double_buffer_vector_add.ipynb) | ✅ 已发布 |
| 03.04 章节实践：独立完成双缓冲调度 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/03_double_buffer_pipeline/03.04_chapter_test.ipynb) | ✅ 已发布 |

### 第四章：Reduce算子与优先队列模拟堆

| Notebook | Link | 状态 |
| --- | --- | --- |
| 04.01 章节介绍：Reduce算子与优先队列模拟堆 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/04_reduce_priority_queue_heap/04.01_chapter_intro.ipynb) | ✅ 已发布 |
| 04.02 Reduce算子与优先队列模拟堆动手实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/04_reduce_priority_queue_heap/04.02_reduce_lab.ipynb) | ✅ 已发布 |
| 04.03 课后测试：Reduce算子与优先队列模拟堆 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/04_reduce_priority_queue_heap/04.03_chapter_test.ipynb) | ✅ 已发布 |

### 第五章：图邻接表向CSR稀疏张量的格式转换

| Notebook | Link | 状态 |
| --- | --- | --- |
| 05.01 图邻接表与并行 SSSP 章节概述 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/05_graph_adj_to_csr_tensor/05.01_chapter_intro.ipynb) | ✅ 已发布 |
| 05.02 基于图邻接表向 CSR 转换的并行 SSSP 算子开发 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/05_graph_adj_to_csr_tensor/05.02_parallel_sssp.ipynb) | ✅ 已发布 |
| 05.03 章节实践：图邻接表向CSR稀疏张量的格式转换 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/05_graph_adj_to_csr_tensor/05.03_chapter_practice.ipynb) | ✅ 已发布 |

### 第六章：链表与连续数组的访存对比

| Notebook | Link | 状态 |
| --- | --- | --- |
| 06.01 链表与连续数组的访存对比章节概述 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/06_linked_list_vs_contiguous_array/06.01_chapter_intro.ipynb) | ✅ 已发布 |
| 06.02 链表与连续数组访存对比实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/06_linked_list_vs_contiguous_array/06.02_memory_access_compare.ipynb) | ✅ 已发布 |
| 06.03 章节实践：链表与连续数组的访存对比 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/06_linked_list_vs_contiguous_array/06.03_chapter_practice.ipynb) | ✅ 已发布 |

### 第七章：巧用排序算法优化MoE融合算子

| Notebook | Link | 状态 |
| --- | --- | --- |
| 07.01 章节介绍：巧用排序算法优化MoE融合算子 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/07_sort_for_moe_fusion/07.01_chapter_intro.ipynb) | ✅ 已发布 |
| 07.02 MoE 排序路由动手实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/07_sort_for_moe_fusion/07.02_moe_sort_lab.ipynb) | ✅ 已发布 |
| 07.03 章节测试：巧用排序算法优化MoE融合算子 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/07_sort_for_moe_fusion/07.03_chapter_test.ipynb) | ✅ 已发布 |

### 第八章：工程部署及性能分析

| Notebook | Link | 状态 |
| --- | --- | --- |
| 08.01 章节介绍：工程部署及性能分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/08_engineering_deployment_and_perf_analysis/08.01_chapter_intro.ipynb) | ✅ 已发布 |
| 08.02 Attention 算子工程部署动手实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/08_engineering_deployment_and_perf_analysis/08.02_attention_operator_lab.ipynb) | ✅ 已发布 |
| 08.03 章节实践与测试：工程部署及性能分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/08_engineering_deployment_and_perf_analysis/08.03_chapter_test.ipynb) | ✅ 已发布 |

### 第九章：装填因子与线性探测的访存优化

| Notebook | Link | 状态 |
| --- | --- | --- |
| 09.01 装填因子与线性探测的访存优化章节概述 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/09_load_factor_and_linear_probing/09.01_chapter_intro.ipynb) | ✅ 已发布 |
| 09.02 开放寻址批量查询实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/09_load_factor_and_linear_probing/09.02_open_addressing_lookup.ipynb) | ✅ 已发布 |
| 09.03 装填因子与线性探测的访存优化章节实践 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/09_load_factor_and_linear_probing/09.03_chapter_practice.ipynb) | ✅ 已发布 |

### 第十章：树形任务队列与流水线调度

| Notebook | Link | 状态 |
| --- | --- | --- |
| 10.01 章节介绍：树形任务队列与流水线调度 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/10_tree_queue_pipeline/10.01_chapter_intro.ipynb) | ✅ 已发布 |
| 10.02 树形任务队列与流水线调度动手实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/10_tree_queue_pipeline/10.02_tree_queue_lab.ipynb) | ✅ 已发布 |
| 10.03 章节测试：树形任务队列与流水线调度 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/10_tree_queue_pipeline/10.03_chapter_test.ipynb) | ✅ 已发布 |

### 第十一章：位图压缩与集合交运算的带宽优化

| Notebook | Link | 状态 |
| --- | --- | --- |
| 11.00 章节介绍：位图压缩与集合交运算的带宽优化 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/11_bitmap_set_bandwidth/11.00_chapter_intro.ipynb) | ✅ 已发布 |
| 11.01 位图集合的数据结构设计 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/11_bitmap_set_bandwidth/11.01_bitmap_set_structure.ipynb) | ✅ 已发布 |
| 11.02 集合交算子与带宽实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/11_bitmap_set_bandwidth/11.02_bitmap_and_operator.ipynb) | ✅ 已发布 |
| 11.03 章节实践：独立完成位图集合交 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/11_bitmap_set_bandwidth/11.03_chapter_test.ipynb) | ✅ 已发布 |

### 第十二章：MoE融合算子与性能分析

| Notebook | Link | 状态 |
| --- | --- | --- |
| 12.01 章节介绍：MoE融合算子与性能分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/12_moe_fused/12.01_chapter_intro.ipynb) | ✅ 已发布 |
| 12.02 MoE Router 融合算子动手实验 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/12_moe_fused/12.02_moe_router_fused_lab.ipynb) | ✅ 已发布 |
| 12.03 章节实践与测试：MoE融合算子与性能分析 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/data_structures_compute&scanFilePath=contrib/tutorials/data_structures_compute/12_moe_fused/12.03_chapter_test.ipynb) | ✅ 已发布 |

[![并行计算：基于鲲鹏与昇腾的实践](./images/readme_cover.png)](https://e.huawei.com/cn/talent/ict-academy/#/ict-courses-detail?courseId=Nyj5J_JYIT2qMx-6Q88ToU0iJPw)

---

# 并行计算：基于鲲鹏与昇腾的实践

## 课程简介

本课程是紧跟"通算与智算融合"产业发展趋势的新工科专业课程，以鲲鹏（Kunpeng）通用算力与昇腾（Ascend）智算算力为软硬件底座，按照"单核 → 多核 → 异构"的算力演进主线组织内容：从并行软硬件架构与存储层次出发，依次学习 ARM NEON SIMD 向量化编程、Pthreads 多线程编程、OpenMP 共享内存编程，再进入 Ascend C 自定义算子开发与 AscendCL 异构应用开发，最终完成一个真实检测模型的端到端部署与调优。

课程总计 48 学时（理论 26 学时、实验 22 学时）。本仓库提供其中的实验部分，覆盖第二章至第七章，共 **41 个 Jupyter Notebook 实验**，参考时长约 24～32 小时。每个实验均按"基准版本 → 单点改动 → 实测对比 → 归因分析"的受控对照方式设计：先验证正确性，再讨论性能，把每一次性能变化都归因到具体的硬件机制或代码改动上。

## 适合人群与前置要求

面向计算机科学与技术、人工智能、软件工程等相关专业大二以上学生，以及对并行计算与异构计算感兴趣的开发人员。

要求具备 C/C++ 程序设计基础，了解计算机组成原理与操作系统的基本概念，熟悉基本的程序调试方法。不要求预先掌握 SIMD、多线程或昇腾开发经验。建议按下表顺序学习，各章的具体前置要求见对应章节 README。

## 学习目标

完成课程后，学习者能够：

- 运用 Roofline 模型与性能度量指标定量分析程序的性能瓶颈，并实测存储层次的各项参数；
- 使用 NEON intrinsic 完成向量化改写，综合运用寄存器分块、Cache 分块与内存打包优化单核性能；
- 使用 Pthreads 完成任务分解、线程同步与线程池设计，并诊断数据竞争、死锁与伪共享等并发缺陷；
- 使用 OpenMP 完成循环并行、归约、负载均衡调度与任务并行，识别并消除循环携带依赖；
- 使用 Ascend C 完成矢量类、规约类、融合类与矩阵类自定义算子的开发与调优；
- 使用 AscendCL 完成资源管理、内存传输、多流流水与设备侧性能度量；
- 使用 ATC 与 AscendCL 完成深度学习模型从权重导出到端侧推理的全链路部署与性能剖析。

## 课程支持的硬件产品

| 项目             | 要求                                                       |
| -------------- | -------------------------------------------------------- |
| CPU 侧实验（第二～五章） | 鲲鹏等 AArch64 多核处理器；需支持 NEON，需真实硬件                         |
| NPU 侧实验（第六～七章） | Atlas A2 / A3 训练推理系列产品                                   |
| CANN 版本        | 9.0.0 及以上                                                |
| 编译器            | CPU 侧：GCC（需 OpenMP 支持）；Ascend C：`bisheng`；AscendCL：`g++` |
| Python         | 3.11                                                     |

## 已验证的在线体验环境
| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| cann-learning-hub 在线体验 notebook | cann_9.0.0_py3.11-A2-arm | Python 3.11.15 | 各 Notebook 表格中的"在线体验"链接可直接打开运行 |
| CANNLab 云开发环境 | cann_9.0.0_py3.11-A2-arm | Python 3.11.4 |参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)创建CANNLab环境运行notebook |

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)，选择对应CANN版本文档。
## 课程章节目录

### 第二章：并行软硬件架构

| Notebook | Link | 状态 |
| --- | --- | --- |
| 02.01 综合实训 · 存储层次实测与访存优化 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/02_arch/02.01_arch_perf.ipynb) | ✅ 已发布 |

### 第三章：ARM NEON SIMD 编程

| Notebook | Link | 状态 |
| --- | --- | --- |
| 03.01 AXPY 向量数乘累加 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/03_simd/03.01_axpy.ipynb) | ✅ 已发布 |
| 03.02 矩阵-向量乘 GEMV | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/03_simd/03.02_gemv.ipynb) | ✅ 已发布 |
| 03.03 RGB → BGR 通道重排 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/03_simd/03.03_rgb2bgr.ipynb) | ✅ 已发布 |
| 03.04 白平衡 Gray World | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/03_simd/03.04_whitebalance.ipynb) | ✅ 已发布 |
| 03.05 RGB → Gray 灰度转换 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/03_simd/03.05_rgb2gray.ipynb) | ✅ 已发布 |
| 03.06 YUV420 → RGB 综合案例 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/03_simd/03.06_yuv2rgb.ipynb) | ✅ 已发布 |
| 03.07 综合实训 · 通用矩阵乘法 GEMM | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/03_simd/03.07_gemm.ipynb) | ✅ 已发布 |

### 第四章：Pthreads 多线程编程

| Notebook | Link | 状态 |
| --- | --- | --- |
| 04.01 Hello World：Fork-Join 与非确定性 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.01_helloworld.ipynb) | ✅ 已发布 |
| 04.02 矩阵向量乘法：数据分解与线程传参 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.02_gemv.ipynb) | ✅ 已发布 |
| 04.03 π 估算：数据竞争与锁粒度 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.03_pi.ipynb) | ✅ 已发布 |
| 04.04 哲学家就餐：死锁、活锁与资源分级 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.04_philosophers.ipynb) | ✅ 已发布 |
| 04.05 消息传递：四种同步策略的递进 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.05_sendmsg.ipynb) | ✅ 已发布 |
| 04.06 生产者-消费者：有界缓冲区的四种实现 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.06_prodcons.ipynb) | ✅ 已发布 |
| 04.07 向量归一化与屏障 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.07_normalize.ipynb) | ✅ 已发布 |
| 04.08 并发链表：读多写少场景的锁策略 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.08_linkedlist.ipynb) | ✅ 已发布 |
| 04.09 伪共享：当正确的程序依然很慢 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.09_falsesharing.ipynb) | ✅ 已发布 |
| 04.10 综合实训 · 基于 Pthreads 的线程池及其应用 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/04_pthread/04.10_threadpool.ipynb) | ✅ 已发布 |

### 第五章：OpenMP 共享内存编程

| Notebook | Link | 状态 |
| --- | --- | --- |
| 05.01 Hello World 与 Fork-Join 模型 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.01_hello.ipynb) | ✅ 已发布 |
| 05.02 梯形积分法 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.02_trapezoid.ipynb) | ✅ 已发布 |
| 05.03 Fibonacci 与 Leibniz 级数 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.03_dependency.ipynb) | ✅ 已发布 |
| 05.04 三角形负载：循环调度与负载均衡 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.04_schedule.ipynb) | ✅ 已发布 |
| 05.05 奇偶换位排序 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.05_oddeven_sort.ipynb) | ✅ 已发布 |
| 05.06 直方图统计 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.06_histogram.ipynb) | ✅ 已发布 |
| 05.07 矩阵向量乘法：嵌套循环并行与 SIMD 协同 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.07_gemv_simd.ipynb) | ✅ 已发布 |
| 05.08 递归归并排序：task 构造与任务并行 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.08_task_mergesort.ipynb) | ✅ 已发布 |
| 05.09 综合实训 · OpenMP × NEON 协同优化矩阵乘法 GEMM | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/05_openmp/05.09_omp_gemm.ipynb) | ✅ 已发布 |

### 第六章：Ascend C 算子开发

| Notebook | Link | 状态 |
| --- | --- | --- |
| 06.01 Ascend C HelloWorld | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/06_ascendc/06.01_hello_world.ipynb) | ✅ 已发布 |
| 06.02 向量加法 Add | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/06_ascendc/06.02_vector_add.ipynb) | ✅ 已发布 |
| 06.03 规约算子 ReduceSum | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/06_ascendc/06.03_reduce_sum.ipynb) | ✅ 已发布 |
| 06.04 激活函数 Sigmoid | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/06_ascendc/06.04_sigmoid.ipynb) | ✅ 已发布 |
| 06.05 融合算子 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/06_ascendc/06.05_fusion.ipynb) | ✅ 已发布 |
| 06.06 矩阵乘法 MatMul | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/06_ascendc/06.06_matmul.ipynb) | ✅ 已发布 |
| 06.07 Softmax | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend&scanFilePath=contrib/tutorials/parallel_computing_practices_on_kunpeng_ascend/06_ascendc/06.07_softmax.ipynb) | ✅ 已发布 |

### 第七章：AscendCL 应用开发

<table>
  <thead>
    <tr>
      <th>Notebook</th>
      <th>状态</th>
      <th>在线体验</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>07.01 应用框架与运行时资源</td>
      <td>✅ 已发布</td>
      <td rowspan="7">
        在CANNLab中运行（
        <a href="https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md">
          CANNLab运行教程指导书
        </a>
        ）
      </td>
    </tr>
    <tr>
      <td>07.02 Host 与 Device 的内存和数据传输</td>
      <td>✅ 已发布</td>
    </tr>
    <tr>
      <td>07.03 Stream 与任务级并行</td>
      <td>✅ 已发布</td>
    </tr>
    <tr>
      <td>07.04 同步管理与性能度量</td>
      <td>✅ 已发布</td>
    </tr>
    <tr>
      <td>07.05 单算子调用：aclnn 两段式接口</td>
      <td>✅ 已发布</td>
    </tr>
    <tr>
      <td>07.06 模型推理：从 ATC 到 aclmdl</td>
      <td>✅ 已发布</td>
    </tr>
    <tr>
      <td>07.07 综合实训：检测算法推理部署</td>
      <td>✅ 已发布</td>
    </tr>
  </tbody>
</table>

> 第一章"并行计算概论"为纯理论章节（并行的概念与层次、Amdahl 与 Gustafson 定律、Roofline 模型、PCAM 方法论），不设配套实验，故本仓库不含 `01_` 目录。

每章的最后一个 Notebook 为该章的**综合实训**，把本章的单项技术组装为一个完整任务：第二章的存储层次实测、第三章的 GEMM 分步优化、第四章的线程池、第五章的 OpenMP × NEON 协同 GEMM、第六章的 Softmax、第七章的检测模型端到端部署。

## 目录结构说明

```
parallel_computing_practices_on_kunpeng_ascend/
├── README.md                 # 本文件：课程总览
├── 02_arch/                  # 第二章 并行软硬件架构
│   ├── README.md
│   ├── 02.01_arch_perf.ipynb
│   └── answer/               # 扩展任务参考实现
├── 03_simd/                  # 第三章 ARM NEON SIMD 编程
├── 04_pthread/               # 第四章 Pthreads 多线程编程
├── 05_openmp/                # 第五章 OpenMP 共享内存编程
├── 06_ascendc/               # 第六章 Ascend C 算子开发
└── 07_ascendcl/              # 第七章 AscendCL 应用开发
```

- 实验源码以 `%%writefile` 的方式内嵌在各 Notebook 中，随讲随写，无需另行准备工程目录。
- `answer/` 目录存放**扩展任务**的参考实现，正文中的动手练习建议先独立完成后再对照。

## 学习建议

1. **按章顺序学习**。第三～五章共享同一条性能优化主线（GEMM），第五章的综合实训直接建立在第三章 NEON 实现之上；第六、七章的实验依次建立在前一实验的编译与运行流程之上。
2. **在真机上运行**。本课程的绝大多数结论来自实测数据，模拟器环境下缓存层次、多核并行与 NPU 行为均不成立。
3. **先正确、后性能**。每个实验都带有正确性校验，请在校验通过后再解读性能数据。
4. **关注趋势而非绝对值**。参考耗时会随硬件型号、编译器版本与系统负载变化，学习时应关注版本之间的相对变化及其成因。

## 作者团队

孔畅（深圳职业技术大学 人工智能学院）

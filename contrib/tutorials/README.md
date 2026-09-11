# 外部贡献课程

本目录用于存放社区开发者贡献的 CANN 学习课程和教程。

## 贡献指南

### 目录结构

每个贡献的课程应放在独立文件夹中，结构示例：

```
contrib/tutorials/
├── your_course_name/
│   ├── README.md          # 课程介绍
│   ├── chapter_1/          # 章节内容
│   │   └── notebook.ipynb  # 章节notebook
│   └── chapter_2/
│       └── notebook.ipynb  # 章节notebook
└── README.md              # 本文件
```

### 提交要求

1. 每个课程需要包含 `README.md` 介绍课程内容和目标
2. 课程介绍中可以署上作者相关信息（姓名、联系方式等）
3. Notebook 文件需使用 UTF-8 编码
4. 代码中避免硬编码路径，使用相对路径或环境变量
5. 提交前请在本地验证 Notebook 可以正常执行

### 审核流程

1. Fork 本仓库
2. 在 `contrib/tutorials/` 下创建您的课程目录
3. 提交 Pull Request 到 `cann/cann-learning-hub` 的 `master` 分支
4. 等待审核通过后合并

## 贡献者列表

| 贡献者 | 课程名称 | 贡献日期 | 状态 |
| --- | --- | --- | --- |
| Datawhale / Torch-RecHub 社区 | [Torch-RecHub 推荐系统实战教程](./torch-rechub) | 2026.05 | ✅ 已迁移 |
| SwanLab / CANN 社区 | [Swan LLM 大模型实战课程](./swan_llm_course) | 2026.05 | 持续建设 |
| SwanLab / CANN 社区 | [SwanLab 共建训练实战案例](./swanlab_examples) | 2026.08 | 持续建设 |
| 北京林业大学 | [高性能计算数据结构课程](./data_structure_for_hpc) | 2026.09 | ✅ 已发布 |
| 华南理工大学 | [嵌入式智能计算课程](./ai_computing_embedded_system) | 2026.09 | ✅ 已发布 |
| 南京信息工程大学 | [《人工智能安全》项目化实践案例](./ai_security_nuist) | 2026.09 | ✅ 已发布 |
| 哈尔滨工业大学（威海） | [基于昇腾处理器的深度学习实验](./ascend_ai_lab) | 2026.09 | ✅ 已发布 |
| 上海交通大学 | [昇腾 AI 端云协同与多模态综合实验](./ascend_multimodal_practice) | 2026.09 | ✅ 已发布 |
| 华中科技大学 | [计算机组成原理与体系结构](./computer_composition_and_architecture) | 2026.09 | ✅ 已发布 |
| 中国科学技术大学 | [数据结构与算法设计（计算实践）](./data_structures_compute) | 2026.09 | ✅ 已发布 |
| 北京航空航天大学 | [智能计算系统（机器学习系统）](./machine_learning_system) | 2026.09 | ✅ 已发布 |
| 华中科技大学 | [星闪短距通信系统仿真](./nearlink_sdr_sim) | 2026.09 | ✅ 已发布 |
| 华东师范大学 | [算子开发与应用](./operator_development_and_application) | 2026.09 | ✅ 已发布 |
| 北京师范大学 | [并行编程原理与实践](./parallel_programming_principles_and_practice) | 2026.09 | ✅ 已发布 |
| 哈尔滨工业大学 | [PyAsc 编译原理课程](./pyasc_compiler_development) | 2026.09 | ✅ 已发布 |
| 东南大学 | [面向国产智算平台的智能计算系统（Qwen 算子开发）](./qwen_ops) | 2026.09 | ✅ 已发布 |

## 课程列表

| 教程名称 | 教程描述 | 访问链接 | 状态 |
| --- | --- | --- | --- |
| Torch-RecHub 推荐系统实战教程 | 基于 Torch-RecHub 的推荐系统端到端实战教程，覆盖 CTR 精排、序列兴趣建模、召回、多任务学习、实验跟踪与模型导出 | [torch-rechub](./torch-rechub) | ✅ 已迁移 |
| Swan LLM 大模型实战课程 | 面向高校学生的大语言模型实战课程，覆盖大模型基础理论、SFT/LoRA 微调、强化学习、推理部署及 Ascend C 性能优化 | [swan_llm_course](./swan_llm_course) | 持续建设 |
| SwanLab 共建训练实战案例 | SwanLab 与 CANN 社区共建的模型训练实践案例，涵盖 MNIST、Qwen2.5 数学解题 LoRA 微调、Qwen3 医学领域 SFT 等场景 | [swanlab_examples](./swanlab_examples) | 持续建设 |
| 高性能计算数据结构课程 | 基于昇腾 NPU 与 CANN 的高性能计算数据结构实践课程，覆盖基础数据结构、并行计算、分布式计算及 Ascend C 算子开发与优化 | [data_structure_for_hpc](./data_structure_for_hpc) | ✅ 已迁移 |
| 嵌入式智能计算课程 | 以华为昇腾 AI 芯片为核心平台的嵌入式 AI 全栈开发课程，覆盖系统部署、模型推理与性能分析 | [ai_computing_embedded_system](./ai_computing_embedded_system) | ✅ 已发布 |
| 《人工智能安全》项目化实践案例 | 围绕人工智能安全主题的项目化实践，覆盖环境配置、智能威胁检测、AI 模型后门攻防与人脸伪造检测 | [ai_security_nuist](./ai_security_nuist) | ✅ 已发布 |
| 基于昇腾处理器的深度学习实验 | 基于昇腾处理器的深度学习实验课程，覆盖图像分类训练、YOLO 训练调优、边缘端推理、NMS 自定义算子开发与 DeepSeek LoRA 微调评估 | [ascend_ai_lab](./ascend_ai_lab) | ✅ 已发布 |
| 昇腾 AI 端云协同与多模态综合实验 | 覆盖视觉（YOLO 目标检测）、语音（ASR/TTS）、语言（LLM LoRA 微调）与多模态机械臂的端云协同综合实验 | [ascend_multimodal_practice](./ascend_multimodal_practice) | ✅ 已发布 |
| 计算机组成原理与体系结构 | 计算机组成原理与体系结构核心课程，配套 CANN 异构环境验证、ATC 模型转换、pyACL 推理与性能分析实验 | [computer_composition_and_architecture](./computer_composition_and_architecture) | ✅ 已发布 |
| 数据结构与算法设计（计算实践） | 以常见数据结构为主线，将数组、栈、队列、图、排序等基础内容与 Ascend C 算子开发结合的计算实践课程 | [data_structures_compute](./data_structures_compute) | ✅ 已发布 |
| 智能计算系统（机器学习系统） | 智能计算系统方向的系统化课程，覆盖 PyTorch 开发验证、Ascend C 算子开发、多卡混合并行训练、模型微调与部署、高并发推理服务 | [machine_learning_system](./machine_learning_system) | ✅ 已发布 |
| 星闪短距通信系统仿真 | 星闪（NearLink）短距通信系统 SDR 仿真实验课程 | [nearlink_sdr_sim](./nearlink_sdr_sim) | ✅ 已发布 |
| 算子开发与应用 | Ascend C 算子开发全流程入门实验，覆盖环境认知、算子原型、Host 侧体验与 Kernel 开发及性能分析 | [operator_development_and_application](./operator_development_and_application) | ✅ 已发布 |
| 并行编程原理与实践 | 围绕国产昇腾平台的并行编程实践，覆盖毕昇工具链、OpenMP、MPI、AscendCL、Ascend C、HCCL 与集群作业部署 | [parallel_programming_principles_and_practice](./parallel_programming_principles_and_practice) | ✅ 已发布 |
| PyAsc 编译原理课程 | 以 PyAsc 开源编译框架为载体的编译原理课程，覆盖 Python 前端、MLIR（ASC-IR）到 Ascend C 代码生成的完整编译链路 | [pyasc_compiler_development](./pyasc_compiler_development) | ✅ 已发布 |
| 面向国产智算平台的智能计算系统（Qwen 算子开发） | 围绕 Qwen2.5 五类核心算子（RMSNorm、RoPE、SwiGLU、GEMM、GQA）的 Ascend C 开发与优化实验，并统一接入 Qwen2.5-0.5B 模型验证 | [qwen_ops](./qwen_ops) | ✅ 已发布 |

---

如果您有任何问题，欢迎提交 Issue 或联系社区维护者。

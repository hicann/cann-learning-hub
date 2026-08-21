# AutoFuse 自动融合开发系列教程

本教程面向希望理解并掌握昇腾 AutoFuse 自动融合能力的开发者，基于 CANN 官方资料整理。课程设计兼顾普通使用者与进阶开发者需求：从零开始建立认知，逐步深入到融合原理与实战应用。

## 教程结构

教程按章节组织，每个章节包含以下内容：

- **Notebooks**：涵盖课程知识点、可执行示例与练习题，适用于自主学习或讲师引导式教学。可在 gitcode 提供的轻量级 Notebook 环境中直接运行，也可自行搭建 JupyterLab 在本地执行。
- **answer**：提供课后习题与章节练习的参考答案，便于自测与验证。
- **images**：存放章节所需的图示资源，Notebook 中通过相对路径引用。

> **注意事项**
>
> - AutoFuse 自动融合特性仅支持 Atlas 350 加速卡、Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品。
> - AutoFuse 的使能方式取决于对接路线。采用 GE 图编译路线时，需在模型图编译前通过环境变量 `AUTOFUSE_FLAGS` 开启自动融合；采用 PyTorch Inductor 对接路线时，当前无需额外配置环境变量，只需在 Python 脚本中导入 `inductor_npu_ext`。使能后，AutoFuse 会在编译阶段自动识别可融合算子模式、生成融合内核并完成相关优化，无需用户手工编写融合代码。

## 软硬件配套说明

| 项目      | 要求                                                                     |
| --------- | ------------------------------------------------------------------------ |
| 支持硬件  | Atlas 350 加速卡、Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上                                                             |
| Python    | 3.11                                                                     |

## 在线体验环境

本教程支持以下在线体验环境：

| 体验环境                            | 镜像模板 / 版本          | Python 内核    | 说明                                                                                                                                              |
| ----------------------------------- | ------------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| cann-learning-hub 在线体验 notebook | cann_9.0.0_py3.11-A2-arm | Python 3.11.15 | 各 Notebook 表格中的"在线体验"链接可直接打开运行                                                                                                  |
| CANNLab 云开发环境                  | cann_9.0.0_py3.11-A2-arm | Python 3.11.4  | 参考[CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)创建CANNLab环境运行notebook |

> **注意：** 如在本地环境离线体验，需自行安装配套的 CANN 软件，具体请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/softwareinstall/instg/atlasdeploy_03_0001.html)。

## 课程目录

### 第一章 AutoFuse 基础介绍

帮助建立 AutoFuse 的整体认知，掌握基础使能方式

| Notebook             | 链接           | 状态      |
| -------------------- | -------------- | --------- |
| 1.1 章节介绍 | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/autofusion_development&scanFilePath=tutorials/autofusion_development/01_basic_overview/01.01_chapter_intro.ipynb) | ✅ 已发布 |
| 1.2 AutoFuse简介     | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/autofusion_development&scanFilePath=tutorials/autofusion_development/01_basic_overview/01.02_autofuse_introduction.ipynb) | ✅ 已发布 |
| 1.3 AutoFuse使能基础 |  [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/autofusion_development&scanFilePath=tutorials/autofusion_development/01_basic_overview/01.03_enable_autofusion.ipynb) | ✅ 已发布 |
| 1.4 章节练习         |  [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/autofusion_development&scanFilePath=tutorials/autofusion_development/01_basic_overview/01.04_chapter_practice.ipynb) | ✅ 已发布 |

### 第二章 AutoFuse 自动融合原理

深入理解融合条件判断与策略求解的核心机制

| Notebook         | 链接           | 状态       |
| ---------------- | -------------- | ---------- |
| 2.1 章节介绍     | 在线体验建设中 | ⏳ 未发布 |
| 2.2 融合条件判断 | 在线体验建设中 | ⏳ 未发布 |
| 2.3 融合策略求解 | 在线体验建设中 | ⏳ 未发布 |
| 2.4 章节练习     | 在线体验建设中 | ⏳ 未发布 |

### 第三章 AutoFuse 项目实践与问题定位

面向实际项目开发，掌握高级配置与问题排查方法

| Notebook                   | 链接           | 状态      |
| -------------------------- | -------------- | --------- |
| 3.1 章节介绍               | 在线体验建设中 | ⏳ 未发布 |
| 3.2 AutoFuse使能进阶       | 在线体验建设中 | ⏳ 未发布 |
| 3.3 对接PyTorch项目实践    | 在线体验建设中 | ⏳ 未发布 |
| 3.4 对接TensorFlow项目实践 | 在线体验建设中 | ⏳ 未发布 |
| 3.5 性能分析方法           | 在线体验建设中 | ⏳ 未发布 |
| 3.6 问题定位方法           | 在线体验建设中 | ⏳ 未发布 |
| 3.7 章节练习               | 在线体验建设中 | ⏳ 未发布 |

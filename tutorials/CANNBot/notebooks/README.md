# CANNBot Notebook 实操课程（零基础入门 + 进阶）

本系列是 CANNBot 课程的 **Notebook 实操篇**，按「认知 → 体验 → 延伸 → 进阶」六册展开：零基础入门篇（1~3 册）带你从「什么是编程 Agent」走到亲手交付第一个经过真机验证的算子；进阶篇（4~6 册）用一个 **xlog1py 算子贯穿「完整版工作流开发 → 性能调优 → 协作上库」全链路**。全部实操记录来自真实端到端运行。

> **与后续课程的关系**：本系列是 [CANNBot 系列课程](./../README.md)（9 课，需算子开发基础）的前置入口。推荐路径：**零基础篇 3 册 → 进阶篇 3 册 → 初级课程第 1～4 课 → 中级课程第 5～9 课**。

## 课程内容

每册 Notebook 包含知识讲解、可运行的演示代码与章末练习（练习答案见 `answer/` 目录）。

### 🟢 零基础入门篇

面向具备 NPU/CANN 基础认知但从未接触过 AI Agent 的开发者。

| 序号 | 主题 | 主要内容 | 前置 | 建议时长 | 状态 |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | [初识 CANNBot](./01_cannbot_basics.ipynb) | 聊天 AI 与编程 Agent 的区别；Agent / Skill / 多 Agent 协作；三层架构与工作流；六仓生态全景 | 无（需 NPU 基础认知） | ~10 min | ✅ |
| 2 | [第一个算子](./02_flash_workflow.ipynb) | 环境自查；Flash 版四步安装；需求五要素；add_custom 算子端到端实操（与手动开发课程 02.04 节同规格对照，真实实测记录）；产物走读与精度验收；sigmoid_custom 分层课后实践 | 第 1 册 | ~40 min | ✅ |
| 3 | [玩转与共建](./03_capabilities_and_community.ipynb) | 调试求助与性能优化；五条开发路径选择；cann-bench 评测；参与贡献的四个入口；学习路线图 | 第 2 册 | ~10 min | ✅ |

### 🔵 进阶篇

面向已完成零基础篇（或具备 1 个以上算子开发经验）的开发者，一个算子贯穿三册。

| 序号 | 主题 | 主要内容 | 前置 | 建议时长 | 状态 |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 4 | [完整版工作流](./04_complete_workflow.ipynb) | 两套工作流的选型原理与三大设计思想（分权制衡 / 决策权留给人的三个点 / 过程产物分离）；决策者视角使用指南；xlog1py 2.5 小时实录精编（QA 门禁拦下 Inf 缺陷、47/47） | 第 1~3 册 | ~40 min | ✅ |
| 5 | [性能调优](./05_performance_optimization.ipynb) | 调优三设计思想（数据先于观点 / 分档读法 / 流量思维）；行动模板；16 用例实录精编与 y_compact 流量发现（866 万倍）；需求演进与停止条件 | 第 4 册 | ~40 min | ✅ |
| 6 | [协作上库](./06_gitcode_collaboration.ipynb) | 收束卷：最后一公里为何适合自动化与五条红线；PR 八步一图速览（POST ≠ 完成）；AI 边界框架与全课程总结；Issue→PR 课后实操 | 第 5 册 | ~10 min | ✅ |

## 软硬件配套说明

| 项目 | 要求 |
| --- | --- |
| 支持硬件 | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| CANN 版本 | 9.0.0 及以上（课程实测记录基于 CANN 9.1） |
| CLI 工具 | OpenCode / Claude Code / Codex / Trae 等任选其一（课程以 OpenCode 为例） |
| Python | 3.11 |

## 在线体验环境

| 体验环境 | 镜像模板 / 版本 | Python 内核 | 说明 |
| --- | --- | --- | --- |
| cann-learning-hub 在线体验 notebook | cann_9.0.0_py3.11-A2-arm | Python 3.11.15 | 各册可直接打开运行（[示例：第 1 册](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/CANNBot/notebooks&scanFilePath=tutorials/CANNBot/notebooks/01_cannbot_basics.ipynb)） |
| CANNLab 云开发环境 | cann_9.0.0_py3.11-A2-arm | Python 3.11 | 参考 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md) 创建环境运行 notebook；第 2 册的 OpenCode 与 Flash 插件按 2.2 节四步安装 |

> **说明**：第 1、3 册为概念/导航册，任何环境可学；第 2 册与进阶篇涉及真实安装、算子生成与性能采集，推荐在 CANNLab 云开发环境（预装 NPU 与 CANN）或本地昇腾环境实践。进阶篇实操沿用第 2 册克隆的 cannbot-skills 仓库。CANNBot 仓库地址：[cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)。

## 反馈与建议

若在学习过程中遇到问题，欢迎：

- 到 [cannbot-skills 仓库](https://gitcode.com/cann/cannbot-skills) 提 Issue 或参与 Discussions
- 在课程仓库提交 Issue
- 加入 CANNBot 微信交流群（见[仓库 Discussions](https://gitcode.com/cann/cannbot-skills/discussions/2)）

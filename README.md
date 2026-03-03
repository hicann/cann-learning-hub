# cann-learning-hub

## 🔥Latest News
- [2026/03] cann-learning-hub项目首次上线。

## 🚀概述
本代码库是与[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）相关的开源学习资料集合。你可以在这里找到用户指南、教程及其他学习资源，免费面向所有对 CANN异构架构感兴趣的学习者开放。

本项目旨在打造一个持续更新的动态项目，让各类文档、示例、最佳实践、优化方案与新功能能够快速、便捷地呈现给用户并供其使用。

## 📝版本配套
本项目源码会基于CANN软件的非beta版本进行全量验证，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明，已验证情况如下下表所示。

| 已验证支持CANN版本 | 验证日期 |
|------------------------|-----------|
| 8.5.0 | 2026.03.02 |

## 📖教程与课程大纲
| 课程 | 课程描述 | 状态 | 支持产品 |
|---------|----------------|----------------|---------------|
| [Ascend C算子开发](tutorials/ascend_c_operator_development) | 基于Ascend C的aicore算子开发教程 | 开发中 | Atlas A2 训练系列产品<br>Atlas A2 推理系列产品 |
| [pyPTO算子开发]() | 基于pyPTO的aicore算子开发教程 | 规划中 | |

## ⚡️快速体验

- **在线体验（建设中）**：开发者根据Readme中的跳转链接基于gitcode提供的轻量级notebook直接体验。

- **gitcod环境体验**：参考[gitcode环境体验指南](./docs/gitcode_env_experience_guide.md)，基于Gitcode提供的notebook进行体验。

## 🔍目录结构
关键目录如下。
```
├── quick_start                        # 快速入门系列教程
│   ├── first_custom_operator          # 基于CANN快速跑通第一个自定义算子
│   ├── first_operator_api_call        # 基于CANN快速跑通第一个算子API调用
│   └── ...                            # 待扩展
├── tutorials                          # CANN各场景开发教程
│   ├── ascend_c_operator_development  # Ascend C算子开发系列教程
│   ├── PyPTO_ operator_development    # PyPTO算子开发系列教程
│   └── ...                            # 待扩展
├── contrib                            # 用户贡献
│   └── ...                            # 待扩展
└── README.md
```

## 💬相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/doc)

## 🤝联系我们
本项目功能和文档正在持续更新和完善中，欢迎您持续关注。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/cann-learning-hub/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/cann-learning-hub/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/cann-learning-hub/wiki)获取技术文章。
# 第5章：Ascend C 调试与性能分析

## 章节概述

围绕 NPU 上板调试、上板性能分析和仿真调优，学习观察运行时数据、采集性能指标并定位性能热点。通过对比调试打印开关前后的数据，理解调试信息对执行耗时的影响。

## 前置要求

- 完成第二章，掌握 Add 算子与 CMake 编译、Host 直调流程。
- 上板实验需要可用 NPU；仿真实验需要匹配的仿真库与 msProf simulator 工具。

## 学习目标

- 使用 DumpTensor 与 printf 辅助定位 Kernel 问题。
- 使用 msProf 采集并阅读上板性能数据。
- 结合仿真 trace 和源码行信息分析性能热点。

## Notebook 与在线体验

| Notebook | 在线体验 |
| --- | --- |
| [05.01 章节介绍](05.01_chapter_intro.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_debug_and_profile/05.01_chapter_intro.ipynb) |
| [05.02 NPU上板调试](05.02_NPU_On-Board_Debugging.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_debug_and_profile/05.02_NPU_On-Board_Debugging.ipynb) |
| [05.03 算子上板调优](05.03_profiling_tool_usage.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_debug_and_profile/05.03_profiling_tool_usage.ipynb) |
| [05.04 算子仿真调优](05.04_simulation_analysis.ipynb) | [在线体验](https://ai.gitcode.com/user/username/notebookcann?repoUrl=https://gitcode.com/cann/cann-learning-hub.git&ttl=120&diskSize=40Gi&path=tutorials/ascendc_operator_development_light&scanFilePath=tutorials/ascendc_operator_development_light/05_debug_and_profile/05.04_simulation_analysis.ipynb) |

## 文件与运行说明

Notebook 按文件名前缀顺序学习，`.01` 为章节介绍。请在本章目录启动或运行 Notebook。本章`images/` 保存课程图片，`src/` 保存配套源码，`answer/` 保存参考答案。运行生成的 `Sources/` 或 `output/` 与仓库配套资源分开保存。练习中的待补充代码需完成后再执行验证。

[返回教程总目录](../README.md)

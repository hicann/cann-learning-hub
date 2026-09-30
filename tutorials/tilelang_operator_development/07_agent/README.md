# 第 7 章 AI Agent 辅助算子开发与交付

**本章约90分钟，建议学完前六章后学习。** 本章以第3章的行Softmax为案例，介绍基于Skills编排的算子交付工作流，学习怎样引导Agent开发TileLang算子，并判断各阶段的结果是否可靠。

学习路线是：认识整体架构 → 学习各阶段的方法 → 跟随同一任务的演进作判断。

| 小节 | Notebook | 时间 |
|---|---|---:|
| 7.1 看懂 Agent 整体架构 | [打开](07.01_overview.ipynb) | 15 分钟 |
| 7.2 Skills 编排原理 | [打开](07.02_agent_workflow.ipynb) | 35 分钟 |
| 7.3 一个 Softmax 算子的交付过程 | [打开](07.03_practice.ipynb) | 40 分钟 |

## 怎样阅读

7.1 沿架构图认识五个阶段。7.2 讲需求怎样形成设计、设计怎样落实代码、ST 怎样形成可信测试，以及调优怎样产生有依据的候选。7.3 沿实际 Softmax 任务展开，每阶段抓一个焦点，前一阶段的选择和结果引出下一阶段的问题。

代码节选、真实结果和图示直接呈现在正文中。完整设计、pytest、候选与日志供进一步查阅；课堂不需要逐行朗读这些文件。

## 当前案例与资料

- [任务输入](src/workflow_run/course_inputs/task.md) 与 [全部九项指定案例](src/workflow_run/course_inputs/cases.json)。
- [初始设计](src/workflow_run/operators/row_softmax/evidence/design/design_v1.md)、[生成初版](src/workflow_run/operators/row_softmax/evidence/generation/initial/row_softmax.py) 与 [指定案例结果](src/workflow_run/operators/row_softmax/evidence/generation/result.json)。
- [ST 测试](src/workflow_run/operators/row_softmax/test_row_softmax.py)、[ST 报告](src/workflow_run/operators/row_softmax/st_report.md) 与 [完整执行记录](src/workflow_run/operators/row_softmax/evidence/st/level1.json)。
- [最终交付文件与验证说明](src/workflow_run/operators/row_softmax_20260916_002620/delivery_reviewed/README.md)。
- [本轮过程与交付导览](src/workflow_run/README.md)。
- [课后完整流程练习](src/exercise_task.md)、[练习解析](answer/07.03_answer.md)。

本轮案例已完成设计、生成、ST、性能调优及最终格式检查：交付副本完整 ST 为 90/90；以整次 kernel `Task Duration` 为主指标，九项时间比的几何均值为 0.93322，最差一项为 1.00792，满足本任务约定。具体材料见本轮导览。

## 课后自主运行

自主运行时，将 [CANNBot Skills 仓库](https://gitcode.com/cann/cannbot-skills/)中的 TileLang 算子开发插件安装到前六章使用的 TileLang 仓库，并在该仓库启动自己使用的 Agent。具体操作见[课后练习](src/exercise_task.md)。

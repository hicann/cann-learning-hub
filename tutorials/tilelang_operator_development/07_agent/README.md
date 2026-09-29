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

自主运行直接使用课程开始时已经拉取的 TileKernels 代码仓。工作流安装脚本的相对路径为 `agent/init.sh`，安装后的启动目录为 `agent/`；完整命令见[课后练习](src/exercise_task.md)。此外还需要完整任务与合适环境：

- Linux 与可用的 Ascend 950 系列 NPU，Docker 可选。
- 与 CANN 匹配的 PyTorch、torch_npu 和 TileLang Ascend 环境；本例采用 AscendC + SimdVF。
- Codex 与工程已安装的 Skills；本练习推荐选择 Flash 模式，完整案例和目标在运行前给清楚。

具体任务与启动方式见 [课后练习](src/exercise_task.md)。换设备、软件或输入要求后，需要按新条件取得自己的结果。课件里的记录不能代替自己的执行。

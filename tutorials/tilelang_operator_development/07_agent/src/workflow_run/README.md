# 第 7.3 节工作流产物导览

本目录保留稳定行 Softmax 案例从任务输入、设计、生成、ST、调优到最终交付的关键产物。课程正文只选取能够说明阶段关系和判断依据的材料。

## 建议阅读顺序

1. 阅读[任务输入](course_inputs/task.md)与[九个指定案例](course_inputs/cases.json)，明确功能、精度和整次 kernel duration 目标。
2. 对照[初始设计](operators/row_softmax/evidence/design/design_v1.md)和[生成初版](operators/row_softmax/evidence/generation/initial/row_softmax.py)，检查“一行一个逻辑 block”怎样落实到代码。
3. 阅读[ST 报告](operators/row_softmax/st_report.md)和[实际 pytest](operators/row_softmax/test_row_softmax.py)，观察测试怎样发现缓存前参数校验缺失。
4. 查看[源码成本分析](operators/row_softmax_20260916_002620/semantic_cost_model.md)和[课程调优报告](operators/row_softmax_20260916_002620/flash-report.md)，理解候选怎样从诊断事实产生，又怎样按整体 kernel 耗时裁决。
5. 最后核对[最终设计与交付说明](operators/row_softmax_20260916_002620/delivery_reviewed/README.md)。

## 最终结果

- 最终实现采用候选 C4。
- 最终交付副本完整 ST 为 90/90。
- 以整次 kernel `Task Duration` 为主指标，九项性能比值的几何均值为 0.93321841，最差一项为 1.00791803，满足总体与逐项门槛。
- 格式检查后重新执行完整 ST，最终代码、测试和证据完成对应。

课程对性能的讲解使用同批真实样本中已经保存的 `Task Duration`，没有重新运行、删除样本或修改原始测量值。

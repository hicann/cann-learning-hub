# 03.04 章节实践参考说明

章节实践只需修改 <code>student_pipeline.h</code>：把物理 Buffer 数量改为 2，并完成预装、预取、上一结果写回和最终排空。完整参考实现见同目录下的 <code>student_pipeline.h</code>。

参考实现应同时通过：

- Host 侧 MockPipeline 调度检查，覆盖 <code>tileCount=1、2、7</code>；
- Ascend C 干净构建；
- 四类非法参数拒绝；
- 四组 NPU 全量精度检查；
- 指标字段检查，其中 <code>buffer_num=2</code>、<code>queue_depth=1</code>、<code>schedule=prefetch</code>。

/*
 * TreeQueuePipelineLite — 910B 教学版 Host/Tiling 实现
 *
 * parent 和 order 的依赖控制由 Host 侧的 Python/reference 实现完成。
 * 设备侧负责按 order 模拟两个缓冲槽和两个 Compute lane 的三阶段时序，
 * 并输出每个任务的 CopyOut 完成时间及依赖校验结果。
 */
#include "tree_queue_pipeline_lite_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t BLOCK_DIM = 1;
static constexpr uint32_t QUEUE_DEPTH = 2;
static constexpr uint32_t COMPUTE_LANES = 2;
static constexpr float COPY_IN = 1.0f;
static constexpr float COPY_OUT = 1.0f;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TreeQueuePipelineLiteTilingData tiling;
    const gert::StorageShape *parentShape = context->GetInputShape(0);
    uint32_t taskCount = static_cast<uint32_t>(parentShape->GetStorageShape().GetShapeSize());

    // 调度存在跨任务依赖，使用一个 control block 保持时序确定；
    // 910B 的 Compute lane 并行度在 kernel 内用 COMPUTE_LANES 表示。
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_taskCount(taskCount);
    tiling.set_queueDepth(QUEUE_DEPTH);
    tiling.set_computeLanes(COMPUTE_LANES);
    tiling.set_copyIn(COPY_IN);
    tiling.set_copyOut(COPY_OUT);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                       context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(taskCount) * sizeof(int32_t);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *parentShape = context->GetInputShape(0);
    int64_t taskCount = parentShape->GetDim(0);
    *context->GetOutputShape(0) = gert::Shape({taskCount});
    *context->GetOutputShape(1) = gert::Shape({1});
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class TreeQueuePipelineLite : public OpDef {
public:
    explicit TreeQueuePipelineLite(const char *name) : OpDef(name)
    {
        this->Input("parent").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cost").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("order").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("stage_end").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dependency_ok").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend310b");
    }
};
OP_ADD(TreeQueuePipelineLite);
}

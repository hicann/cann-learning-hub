#include "../op_kernel/attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // ---------- 1. 从输入 shape 读取序列长度 S 与头维度 D ----------
    const gert::StorageShape* qShape = context->GetInputShape(0);
    int32_t seqLen = qShape->GetStorageShape().GetDim(0);
    int32_t dim = qShape->GetStorageShape().GetDim(1);

    AttentionCustomTilingData *tiling = context->GetTilingData<AttentionCustomTilingData>();
    tiling->seqLen = static_cast<uint32_t>(seqLen);
    tiling->dim = static_cast<uint32_t>(dim);
    tiling->scale = 1.0f / std::sqrt(static_cast<float>(dim));

    // ---------- 2. 多核执行：按行切分（纯向量核，行间无依赖）----------
    auto ubPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivCoreNum = ubPlatform.GetCoreNumAiv();
    uint32_t blockDim = aivCoreNum > 0 ? aivCoreNum : ubPlatform.GetCoreNum();
    if (blockDim > static_cast<uint32_t>(seqLen)) {
        blockDim = static_cast<uint32_t>(seqLen);
    }
    context->SetBlockDim(blockDim);

    // ---------- 3. 工作空间 ----------
    // 本算子不使用 workspace（中间结果分批暂存于输出缓冲）
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    // BISECT: 打印核数与 UB 大小
    uint64_t ubSize = 0;
    ubPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    fprintf(stderr, "[DEBUG] UB size=%llu AIV cores=%u blockDim=%u\n",
            (unsigned long long)ubSize, aivCoreNum, blockDim);

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(0);
    gert::Shape* oShape = context->GetOutputShape(0);
    *oShape = *qShape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AttentionCustom : public OpDef {
public:
    explicit AttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AttentionCustom);
}

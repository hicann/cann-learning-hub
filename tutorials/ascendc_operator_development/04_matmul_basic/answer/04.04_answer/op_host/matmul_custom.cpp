
#include "matmul_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t M = 1024;
    int32_t N = 640;      // B的列数
    int32_t K = 256;      // A的列数 / B的行数
    int32_t baseM = 128;
    int32_t baseN = 128;
    
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);   // 使用2个核并行计算
    
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    MatmulCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;   // Tiling生成失败
    }
    uint64_t localMemSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    tiling.set_localMemSize(localMemSize);   // 存入Tiling数据，供核函数分配工作空间
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        context->SetBlockDim(2);
        context->SetTilingKey(2);
    } else {
        context->SetBlockDim(1);
        context->SetTilingKey(1);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
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
class MatmulCustom : public OpDef {
public:
    explicit MatmulCustom(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(MatmulCustom);
}

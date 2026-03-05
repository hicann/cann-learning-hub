
#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"
#include "../op_kernel/tiling_key_add_custom_template.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
  ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
  ge::DataType dtype_y = context->GetOutputDesc(0)->GetDataType();
  uint32_t D_T_X = static_cast<int>(dtype_x);

  uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
  float max_value = *context->GetAttrs()->GetFloat(0);
  float min_value = *context->GetAttrs()->GetFloat(1);
  tiling->totalLength = totalLength;
  tiling->tileNum = 8;
  tiling->max = max_value;
  tiling->min = min_value;
  context->SetBlockDim(8);
  ASCENDC_TPL_SEL_PARAM(context, D_T_X); 
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
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
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("max").AttrType(OPTIONAL).Float(0);
        this->Attr("min").AttrType(OPTIONAL).Float(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(AddCustomTemplate);
}

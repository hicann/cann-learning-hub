#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferDataTypeForAddCustom(gert::InferDataTypeContext* context)
{
    OP_LOGI("Begin InferDataTypeForAddCustom");
    const ge::DataType x1DataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1DataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AddCustom).InferDataType(InferDataTypeForAddCustom);
} // namespace ops
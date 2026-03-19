#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static ge::graphStatus InferShapeSigmoid(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSigmoid");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 填充输出shape大小
    auto xShapeSize = xShape->GetDimNum();
    yShape->SetDimNum(xShapeSize);
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeSigmoid");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sigmoid).InferShape(InferShapeSigmoid);
} // namespace ops
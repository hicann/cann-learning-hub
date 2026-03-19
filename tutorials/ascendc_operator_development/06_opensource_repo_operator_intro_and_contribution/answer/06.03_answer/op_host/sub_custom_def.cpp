
#include "register/op_def_registry.h"

namespace ops {
class SubCustom : public OpDef {
public:
    explicit SubCustom(const char* name) : OpDef(name)
    {
        // 输入参数说明
        this->Input("x1") 
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT}) 
            .Format({ge::FORMAT_ND}) 
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous(); 
        this->Input("x2") 
            .ParamType(REQUIRED) 
            .DataType({ge::DT_FLOAT}) 
            .Format({ge::FORMAT_ND}) 
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y") 
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SubCustom); // 添加算子信息库
} // namespace ops

#include "register/op_def_registry.h"

namespace ops {
class Sigmoid : public OpDef {
public:
    explicit Sigmoid(const char* name) : OpDef(name)
    {
        // 输入参数说明
        this->Input("x")                                       // 输入x1定义
            .ParamType(REQUIRED)                                // 必选输入
            .DataType({ge::DT_FLOAT, ge::DT_INT32})             // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})             // 支持format格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小shape对应format格式
            .AutoContiguous();                                  // 内存自动连续化
        this->Output("y") // 输出y定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Sigmoid); // 添加算子信息库
} // namespace ops
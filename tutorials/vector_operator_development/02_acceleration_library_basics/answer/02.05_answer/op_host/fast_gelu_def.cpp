#include "register/op_def_registry.h"
#include "tiling/tiling_data.h"

namespace ge {

class FastGeluOp : public OpDef {
public:
    explicit FastGeluOp(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        OpAICoreConfig cfg;
        cfg.ExtendCfgInfo("opFile.value", "fast_gelu_kernel");
        cfg.ExtendCfgInfo("tilingFile.value", "fast_gelu_tiling");
        this->AICore().AddConfig("ascend950", cfg);
    }
};

IMPL_OP_INFERSHAPE(FastGelu).InferShape(Ops::Base::InferShape4Elewise).InferDataType(Ops::Base::InferDtype4Elewise);

REG_OP(FastGelu)
    .Input("x", DT_FLOAT)
    .Output("y", DT_FLOAT)
    .OP_END();

}  // namespace ge

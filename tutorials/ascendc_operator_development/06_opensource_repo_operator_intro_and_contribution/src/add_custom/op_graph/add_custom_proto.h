#ifndef ADD_CUSTOM_PROTO_H_
#define ADD_CUSTOM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(AddCustom)
    .INPUT(x1, TensorType({ DT_FLOAT, DT_INT32}))
    .INPUT(x2, TensorType({ DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(AddCustom)
} // namespace ge
#endif // ADD_CUSTOM_PROTO_H_
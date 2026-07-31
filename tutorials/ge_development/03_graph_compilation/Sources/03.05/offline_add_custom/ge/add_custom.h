/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_TUTORIAL_0305_OFFLINE_ADD_CUSTOM_PROTO_H_
#define GE_TUTORIAL_0305_OFFLINE_ADD_CUSTOM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(AddCustom)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddCustom)
}  // namespace ge

#endif  // GE_TUTORIAL_0305_OFFLINE_ADD_CUSTOM_PROTO_H_

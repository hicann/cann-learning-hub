/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#include <cstdint>

struct AddCustomTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};

#endif  // ADD_CUSTOM_TILING_H

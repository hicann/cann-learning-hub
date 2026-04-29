/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include "function.h"
using variable_list = std::vector<at::Tensor>;

// 为NPU设备注册前向实现
at::Tensor add_custom_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    // 创建输出内存
    at::Tensor result = at::empty_like(self);

    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnAddCustomTemplate, self, other, result);
    return result;
}


// 为NPU设备注册前反向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("add_custom", &add_custom_impl_npu);
}



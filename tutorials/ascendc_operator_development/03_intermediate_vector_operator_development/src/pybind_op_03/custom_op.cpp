/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


at::Tensor npu_add_custom_template(const at::Tensor &x, const at::Tensor &y) {
    at::Tensor z = at::empty_like(x);
    EXEC_NPU_CMD(aclnnAddCustomTemplate, x, y, z);
    return z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("npu_add_custom_template", &npu_add_custom_template, "torch add");
}

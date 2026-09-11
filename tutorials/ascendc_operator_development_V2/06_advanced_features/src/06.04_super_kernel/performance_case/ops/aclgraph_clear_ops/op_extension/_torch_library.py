# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch
import torch.library as library

from . import custom_ops_lib

_LIBRARY = None
_REGISTERED = False


def register_torch_ops():
    global _LIBRARY, _REGISTERED
    if _REGISTERED:
        return

    _LIBRARY = library.Library("ascendc_ops", "FRAGMENT")
    _LIBRARY.define("clear_ops(Tensor a) -> Tensor")

    @library.impl(_LIBRARY, "clear_ops", "Meta")
    def clear_ops_meta(a):
        return torch.empty_like(a)

    @library.impl(_LIBRARY, "clear_ops", "PrivateUse1")
    def clear_ops_impl(a):
        return custom_ops_lib.run_clear_ops(a)

    _REGISTERED = True

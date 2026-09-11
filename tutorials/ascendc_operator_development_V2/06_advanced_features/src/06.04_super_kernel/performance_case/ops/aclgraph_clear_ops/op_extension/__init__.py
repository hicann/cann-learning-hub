# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from pathlib import Path

from . import custom_ops_lib
from ._torch_library import register_torch_ops

LOADED_LIBRARY_PATH = Path(custom_ops_lib.__file__).as_posix()
run_clear_ops = custom_ops_lib.run_clear_ops
register_torch_ops()

__all__ = ["LOADED_LIBRARY_PATH", "run_clear_ops", "register_torch_ops"]

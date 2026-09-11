# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import importlib
from importlib import util as importlib_util
import time

import mstx
import numpy as np
import torch
import torch_npu


AUTO_CUSTOM_OP_PACKAGES_BY_MODE = {
    "aclgraph": ("op_extension",),
    "aclgraph+sk": ("op_extension",),
}
MSTX_MESSAGE = "custom_sk_model_steps"


def load_custom_op_packages(run_mode):
    loaded_package_names = []
    for package_name in AUTO_CUSTOM_OP_PACKAGES_BY_MODE.get(run_mode, ()):
        if importlib_util.find_spec(package_name) is None:
            print(
                f"Optional custom op package {package_name} is not installed; skip loading it."
            )
            continue
        module = importlib.import_module(package_name)
        loaded_path = getattr(module, "LOADED_LIBRARY_PATH", "<installed package>")
        print(f"Loaded installed custom op package {package_name}: {loaded_path}.")
        loaded_package_names.append(package_name)
    return set(loaded_package_names)


def setup_device(device_id):
    torch_npu.npu.set_device(f"npu:{device_id}")
    torch_npu.npu.set_op_timeout_ms(10000)


def setup_seed(seed=1236):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_with_msprof_range(model, x, args):
    """Run warmup outside, then mark one formal step for external msprof."""
    if args.skip_first < 0:
        raise ValueError("skip_first must be non-negative")
    if args.step_cnt <= 0:
        raise ValueError("step_cnt must be positive")

    y = None
    for _ in range(args.skip_first):
        y = model(x)
    torch.npu.synchronize()

    stream = torch_npu.npu.current_stream()
    range_id = mstx.range_start(MSTX_MESSAGE, stream.npu_stream)
    if range_id == 0:
        raise RuntimeError("MSTX range start failed; run this mode under msprof")

    start_time = time.perf_counter()
    try:
        for _ in range(args.step_cnt):
            y = model(x)
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
    finally:
        mstx.range_end(range_id)

    print(
        f"#### {MSTX_MESSAGE}: {elapsed_ms:.3f} ms total, "
        f"{elapsed_ms / args.step_cnt:.3f} ms/step ####"
    )
    return y

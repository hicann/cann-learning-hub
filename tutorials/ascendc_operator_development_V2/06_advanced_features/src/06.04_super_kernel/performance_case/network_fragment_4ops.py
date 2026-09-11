# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
SuperKernel example code using PyTorch with NPU backend.
python3 torch_npu_example.py
inputs: [x]
layer1: [grouped_matmul, dequant_swiglu_quant, grouped_matmul, dynamic_quant]
layer2: [grouped_matmul, dequant_swiglu_quant, grouped_matmul, dynamic_quant]
...layerN: [grouped_matmul, dequant_swiglu_quant, grouped_matmul, dynamic_quant]
outputs: [y]
"""

import argparse
import torch
import torch.nn as nn
import torch_npu

from util import (
    load_custom_op_packages,
    run_with_msprof_range,
    setup_device,
    setup_seed,
)


def _compile_aclgraph_model(model, run_mode):
    options = {
        "static_kernel_compile": True,
        "super_kernel_optimize": run_mode.endswith("+sk"),
    }
    if run_mode.endswith("+sk"):
        options["super_kernel_optimize_options"] = {
            "aggressive_opt_strategies": {
                "event_breaker_bypass": 0b00,
                "value_breaker_bypass": 0b01,
                "task_breaker_bypass": 0b00,
            },
            "preload_code": 0,
        }
    return torch.compile(
        model,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
        options=options,
    )


def _run_custom_clear_op(x, run_mode, custom_op_packages):
    if "op_extension" in custom_op_packages:
        return torch.ops.ascendc_ops.clear_ops(x)
    return x


class NpuAscModule(nn.Module):
    def __init__(
        self, epsilon, out_dim=7168, hidden_dim=4096, tokens_num=9216, group_num=1
    ):
        super(NpuAscModule, self).__init__()
        self.epsilon = epsilon
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.tokens_num = tokens_num
        self.group_num = group_num
        # todo : need check it
        # self.group_list = nn.Parameter(
        #     torch.tensor([tokens_num // group_num] * group_num, device="npu", dtype=torch.int64), requires_grad=False
        # )
        self.group_list = nn.Parameter(
            torch.tensor([1] * group_num, device="npu", dtype=torch.int64),
            requires_grad=False,
        )
        self.gmm_config1 = nn.ParameterDict(
            {
                "weight": nn.Parameter(
                    torch.rand(
                        [self.group_num, self.out_dim, self.hidden_dim],
                        device="npu",
                        dtype=torch.int8,
                    ),
                    requires_grad=False,
                ),
            }
        )
        self.gmm_config2 = nn.ParameterDict(
            {
                "weight": nn.Parameter(
                    torch.rand(
                        [self.group_num, self.hidden_dim // 2, self.out_dim],
                        device="npu",
                        dtype=torch.int8,
                    ),
                    requires_grad=False,
                ),
                "scale": nn.Parameter(
                    torch.rand(
                        [self.group_num, self.out_dim],
                        device="npu",
                        dtype=torch.bfloat16,
                    ),
                    requires_grad=False,
                ),
                "per_token_scale": nn.Parameter(
                    torch.rand([self.tokens_num], device="npu", dtype=torch.float32),
                    requires_grad=False,
                ),
            }
        )
        self.dequant_swiglu_quant_config = nn.ParameterDict(
            {
                "weight_scale": nn.Parameter(
                    torch.rand(
                        [self.group_num, self.hidden_dim],
                        device="npu",
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                ),
                "activation_scale": nn.Parameter(
                    torch.rand([self.tokens_num], device="npu", dtype=torch.float32),
                    requires_grad=False,
                ),
                "quant_scale": nn.Parameter(
                    torch.rand(
                        [self.group_num, self.hidden_dim // 2],
                        device="npu",
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                ),
            }
        )

    def grouped_matmul_pre(self, x):
        y = torch_npu.npu_grouped_matmul(
            [x],
            [self.gmm_config1["weight"]],
            group_list=self.group_list,
            group_type=0,
            group_list_type=1,
            split_item=3,
            output_dtype=torch.int32,
        )[0]
        return y

    def dequant_swiglu_quant(self, x):
        y = torch_npu.npu_dequant_swiglu_quant(
            x,
            weight_scale=self.dequant_swiglu_quant_config["weight_scale"],
            activation_scale=self.dequant_swiglu_quant_config["activation_scale"],
            quant_scale=self.dequant_swiglu_quant_config["quant_scale"],
            group_index=self.group_list,
            quant_mode=1,
        )[0]
        return y

    def grouped_matmul_after(self, x):
        y = torch_npu.npu_grouped_matmul(
            [x],
            [self.gmm_config2["weight"]],
            scale=[self.gmm_config2["scale"]],
            per_token_scale=[self.gmm_config2["per_token_scale"]],
            group_list=self.group_list,
            group_type=0,
            group_list_type=1,
            split_item=3,
            output_dtype=torch.bfloat16,
        )[0]
        return y

    def dynamic_quant(self, x):
        y = torch_npu.npu_dynamic_quant(x)[0]
        return y

    def forward(self, x):
        x = self.grouped_matmul_pre(x)
        x = self.dequant_swiglu_quant(x)
        x = self.grouped_matmul_after(x)
        x = self.dynamic_quant(x)
        return x


class RunModel(nn.Module):
    def __init__(self, layer_cnt, run_mode, custom_op_packages):
        super(RunModel, self).__init__()
        self.layer_cnt = layer_cnt
        self.run_mode = run_mode
        self.custom_op_packages = custom_op_packages
        self.layers = nn.ModuleList([NpuAscModule(epsilon=1e-5)] * layer_cnt)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            if self.run_mode == "aclgraph+sk":
                scope_name = f"sk_test_layer_{layer_id}"
                torch.npu.super_kernel_scope_begin(scope_name)
                x = layer(x)
                x = _run_custom_clear_op(x, self.run_mode, self.custom_op_packages)
                torch.npu.super_kernel_scope_end(scope_name)
            else:
                x = layer(x)
                x = _run_custom_clear_op(x, self.run_mode, self.custom_op_packages)
        return x


def get_args():
    parser = argparse.ArgumentParser(description="SuperKernel NPU Example")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--layer_cnt", type=int, default=50, help="Number of layers in the model"
    )
    parser.add_argument(
        "--skip_first", "--warmup",
        dest="skip_first",
        type=int,
        default=1,
        help="Number of initial steps to skip in profiling",
    )
    parser.add_argument(
        "--step_cnt", type=int, default=1, help="Number of steps to run in profiling"
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="aclgraph",
        choices=["aclgraph", "aclgraph+sk"],
    )
    parser.add_argument(
        "--msprof",
        action="store_true",
        help="Use an external msprof session and mark the formal step with MSTX.",
    )

    return parser.parse_args()


def create_input():
    x_shape = [9216, 7168]
    x = torch.ones(x_shape, device="npu", dtype=torch.int8)
    # x_shape = [9216, 4096]
    # x = torch.ones(x_shape, device="npu", dtype=torch.int32)
    return x


def print_run_config(args, x):
    print("#### model init done ####")
    print(f"#### compile config: run_mode={args.run_mode} ####")
    print(f"#### model layer cnt: {args.layer_cnt} ####")
    print(
        f"#### profiler step cnt: {args.step_cnt}, skip_first: {args.skip_first} ####"
    )
    print(f"#### input x: {x.dtype} {x.shape} ####")


def main():
    args = get_args()
    if not args.msprof:
        raise ValueError("Run this case through run_case.sh under external msprof")

    custom_op_packages = load_custom_op_packages(args.run_mode)
    setup_device(args.device_id)
    setup_seed()

    x = create_input()
    model = RunModel(
        layer_cnt=args.layer_cnt,
        run_mode=args.run_mode,
        custom_op_packages=custom_op_packages,
    ).npu()
    print_run_config(args, x)
    model = _compile_aclgraph_model(model, args.run_mode)

    y = run_with_msprof_range(model, x, args)

    print(f"#### input x: {x.dtype} {x.shape} ####")
    print(f"#### output y: {y.dtype} {y.shape} ####")
    print("run done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

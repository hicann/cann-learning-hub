#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
import torch_npu
import op_extension  # Registers the bundled ACLGraph clear_ops.


TOKENS_NUM = 9216
OUT_DIM = 7168
HIDDEN_DIM = 4096
GROUP_NUM = 1


def frozen_rand(shape: tuple[int, ...], dtype: torch.dtype) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.rand(shape, device="npu", dtype=dtype),
        requires_grad=False,
    )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group_list = torch.nn.Parameter(
            torch.ones(GROUP_NUM, device="npu", dtype=torch.int64),
            requires_grad=False,
        )
        self.gmm_pre_weight = frozen_rand(
            (GROUP_NUM, OUT_DIM, HIDDEN_DIM), torch.int8
        )
        self.gmm_after_weight = frozen_rand(
            (GROUP_NUM, HIDDEN_DIM // 2, OUT_DIM), torch.int8
        )
        self.gmm_after_scale = frozen_rand(
            (GROUP_NUM, OUT_DIM), torch.bfloat16
        )
        self.gmm_after_per_token_scale = frozen_rand(
            (TOKENS_NUM,), torch.float32
        )
        self.swiglu_weight_scale = frozen_rand(
            (GROUP_NUM, HIDDEN_DIM), torch.float32
        )
        self.swiglu_activation_scale = frozen_rand(
            (TOKENS_NUM,), torch.float32
        )
        self.swiglu_quant_scale = frozen_rand(
            (GROUP_NUM, HIDDEN_DIM // 2), torch.float32
        )

    def grouped_matmul_pre(self, x: torch.Tensor) -> torch.Tensor:
        y = torch_npu.npu_grouped_matmul(
            [x],
            [self.gmm_pre_weight],
            group_list=self.group_list,
            group_type=0,
            group_list_type=1,
            split_item=3,
            output_dtype=torch.int32,
        )[0]
        return y

    def dequant_swiglu_quant(self, x: torch.Tensor) -> torch.Tensor:
        y = torch_npu.npu_dequant_swiglu_quant(
            x,
            weight_scale=self.swiglu_weight_scale,
            activation_scale=self.swiglu_activation_scale,
            quant_scale=self.swiglu_quant_scale,
            group_index=self.group_list,
            quant_mode=1,
        )[0]
        return y

    def grouped_matmul_after(self, x: torch.Tensor) -> torch.Tensor:
        y = torch_npu.npu_grouped_matmul(
            [x],
            [self.gmm_after_weight],
            scale=[self.gmm_after_scale],
            per_token_scale=[self.gmm_after_per_token_scale],
            group_list=self.group_list,
            group_type=0,
            group_list_type=1,
            split_item=3,
            output_dtype=torch.bfloat16,
        )[0]
        return y

    def dynamic_quant(self, x: torch.Tensor) -> torch.Tensor:
        y = torch_npu.npu_dynamic_quant(x)[0]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grouped_matmul_pre(x)
        x = self.dequant_swiglu_quant(x)
        x = torch.ops.ascendc_ops.gelu_custom(x)
        x = self.grouped_matmul_after(x)
        x = self.dynamic_quant(x)
        x = torch.ops.ascendc_ops.clear_ops(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question 1 ACLGraph sample.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.steps < 1:
        parser.error("--steps must be positive")
    return args


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    return torch.compile(
        model,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
        options={
            "static_kernel_compile": True,
            "super_kernel_optimize": True,
        },
    )


def run_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    steps: int,
) -> torch.Tensor:
    output = None
    with torch.no_grad():
        for _ in range(warmup):
            output = model(x)
            torch.npu.synchronize()

        print(f"warmup complete: {warmup} step(s)")
        for step in range(steps):
            output = model(x)
            torch.npu.synchronize()
            print(f"profile step complete: {step + 1}/{steps}")

    if output is None:
        raise RuntimeError("The model did not execute")
    return output


def main() -> None:
    args = parse_args()
    torch_npu.npu.set_device(f"npu:{args.device_id}")
    torch_npu.npu.set_op_timeout_ms(10000)
    torch.manual_seed(1236)

    model = compile_model(Model().npu().eval())
    x = torch.ones((TOKENS_NUM, OUT_DIM), device="npu", dtype=torch.int8)

    output = run_model(model, x, args.warmup, args.steps)

    print(f"output: dtype={output.dtype}, shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()

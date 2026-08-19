#!/usr/bin/env python3
import argparse
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


@torch.no_grad()
def run_forward(model, inputs, repeat: int):
    _ = model(**inputs).logits
    torch.npu.synchronize()

    times = []
    logits = None
    for _ in range(repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        logits = model(**inputs).logits
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return logits, sum(times) / len(times)


def patch_qwen_linear_gemm(model):
    load_torch_ops()
    gemm = torch.ops.gemm_custom.gemm
    patched = 0

    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        def custom_forward(self, x):
            original_shape = x.shape[:-1]
            in_features = x.shape[-1]
            x2d = x.reshape(-1, in_features).contiguous()
            weight_kn = self.weight.t().contiguous()
            y2d = gemm(x2d, weight_kn)
            if self.bias is not None:
                y2d = y2d + self.bias
            return y2d.reshape(*original_shape, self.weight.shape[0])

        module.forward = types.MethodType(custom_forward, module)
        patched += 1

    if patched == 0:
        raise RuntimeError("No torch.nn.Linear modules were patched")

    print(f"[PATCH] patched {patched} torch.nn.Linear modules with custom GEMM")


def load_local_or_remote(model_path: str, attn_implementation: str):
    local_only = Path(model_path).exists()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    config._attn_implementation = attn_implementation
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        local_files_only=local_only,
    ).eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get(
        "QWEN_OPS_MODEL_PATH",
        str(Path(__file__).resolve().parents[2].parent / "Models" / "Qwen2.5-0.5B"),
    ))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()

    device = "npu"

    model, tokenizer = load_local_or_remote(args.model, args.attn_implementation)
    model = model.to(device)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("[RUN] native Qwen")
    logits_native, time_native = run_forward(model, inputs, args.repeat)

    print("[RUN] Qwen with custom baseline GEMM Linear")
    patch_qwen_linear_gemm(model)
    logits_custom, time_custom = run_forward(model, inputs, args.repeat)

    diff = (logits_custom - logits_native).abs()
    native_next = torch.argmax(logits_native[0, -1]).item()
    custom_next = torch.argmax(logits_custom[0, -1]).item()

    print("\n========== Result ==========")
    print(f"native time       : {time_native:.3f} ms")
    print(f"custom time       : {time_custom:.3f} ms")
    print(f"speedup           : {time_native / time_custom:.3f}x")
    print(f"max_abs_diff      : {diff.max().item():.8e}")
    print(f"mean_abs_diff     : {diff.mean().item():.8e}")
    print(f"native next token : {tokenizer.decode([native_next])!r}")
    print(f"custom next token : {tokenizer.decode([custom_next])!r}")
    print(f"next token match  : {native_next == custom_next}")

    ok = torch.allclose(logits_native, logits_custom, atol=args.atol, rtol=args.rtol)
    print("PASS" if ok else "WARN: logits differ beyond tolerance")


if __name__ == "__main__":
    main()

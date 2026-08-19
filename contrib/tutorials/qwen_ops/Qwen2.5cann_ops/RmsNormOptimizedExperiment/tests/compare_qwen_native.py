#!/usr/bin/env python3
import argparse
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_extension import load_torch_ops


@torch.no_grad()
def run_forward(model, input_ids, repeat: int):
    _ = model(input_ids).logits

    times = []
    logits = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        logits = model(input_ids).logits
        times.append((time.perf_counter() - t0) * 1000.0)
    return logits, sum(times) / len(times)


def patch_qwen_rmsnorm(model):
    load_torch_ops()
    rms_norm = torch.ops.rmsnorm_custom.rms_norm
    patched = 0

    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if "rmsnorm" not in class_name:
            continue
        if not hasattr(module, "weight"):
            continue

        def custom_forward(self, hidden_states):
            eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
            out = rms_norm(hidden_states.float().contiguous(), self.weight.float().contiguous(), float(eps))
            return out.to(dtype=hidden_states.dtype)

        module.forward = types.MethodType(custom_forward, module)
        patched += 1

    print(f"[PATCH] patched {patched} Qwen RMSNorm modules with custom RMSNorm")


def load_local_or_remote(model_path: str):
    local_only = Path(model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
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
    args = parser.parse_args()

    model, tokenizer = load_local_or_remote(args.model)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print("[RUN] native Qwen")
    logits_native, time_native = run_forward(model, input_ids, args.repeat)

    print("[RUN] Qwen with custom RMSNorm")
    patch_qwen_rmsnorm(model)
    logits_custom, time_custom = run_forward(model, input_ids, args.repeat)

    diff = (logits_custom - logits_native).abs()
    native_next = torch.argmax(logits_native[0, -1]).item()
    custom_next = torch.argmax(logits_custom[0, -1]).item()

    print("\n========== Result ==========")
    print(f"native time       : {time_native:.3f} ms")
    print(f"custom time       : {time_custom:.3f} ms")
    print(f"max_abs_diff      : {diff.max().item():.8e}")
    print(f"mean_abs_diff     : {diff.mean().item():.8e}")
    print(f"native next token : {tokenizer.decode([native_next])!r}")
    print(f"custom next token : {tokenizer.decode([custom_next])!r}")
    print(f"next token match  : {native_next == custom_next}")

    ok = torch.allclose(logits_native, logits_custom, atol=5e-3, rtol=5e-3)
    print("PASS: logits close" if ok else "WARN: logits differ beyond tolerance")


if __name__ == "__main__":
    main()

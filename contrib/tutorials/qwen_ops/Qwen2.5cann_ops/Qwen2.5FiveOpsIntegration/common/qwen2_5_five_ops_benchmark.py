#!/usr/bin/env python3
"""Qwen2.5 five-operator end-to-end benchmark.

This program deliberately keeps the original wrappers unchanged.  GEMM,
SwiGLU and optimized RoPE are NPU-tensor operators; RMSNorm, GQA and baseline
RoPE use host-tensor ACL wrappers.  The bridge mode therefore copies only at
those wrapper boundaries and measures the complete cost, including the copies.
"""
import argparse
import ctypes
import json
import os
import time
import types
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
DEFAULT_MODEL = Path(
    os.environ.get("QWEN_OPS_MODEL_PATH", PROJECT_ROOT / "Models" / "Qwen2.5-0.5B")
)


def require_npu(torch):
    try:
        import torch_npu  # noqa: F401 - registers the npu device
    except Exception as exc:
        raise RuntimeError("torch_npu is required by GEMM/SwiGLU/optimized-RoPE") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("no usable NPU is visible to torch_npu")


def load_library(torch, path):
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run the project's scripts/build.sh")
    torch.ops.load_library(str(path))


def load_ops(torch, variant):
    # AscendC kernel DSOs contain ACL runtime references that must be
    # resolvable before the kernel itself is loaded.
    ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    suffix = "BaselineExperiment" if variant == "baseline" else "OptimizedExperiment"
    projects = {
        "rope": ROOT / f"Rope{suffix}",
        "gemm": ROOT / f"Gemm{suffix}",
        "rms": ROOT / f"RmsNorm{suffix}",
        "swiglu": ROOT / f"SwiGlu{suffix}",
        "gqa": ROOT / f"GqaAttention{suffix}",
    }
    # Keep the projects' original dynamic-library loading order.
    kernel_libraries = {
        # Fresh builds normally emit the operator-specific name.  Some
        # existing optimized artifacts use CANN's generic fallback name;
        # accept both so an integration run can report a real build/layout
        # error instead of assuming the baseline artifact layout.
        "gemm": ("libgemm_kernels_npu.so", "libascendc_kernels_npu.so"),
        "swiglu": ("libswiglu_kernels_npu.so", "libascendc_kernels_npu.so"),
        "rope": ("librope_kernels_npu.so", "libascendc_kernels_npu.so"),
        "rms": ("librmsnorm_kernels_npu.so", "libascendc_kernels_npu.so"),
        "gqa": ("libascendc_gqa_kernels.so",),
    }
    for name in ("gemm", "swiglu"):
        kernel = next((projects[name] / "out/lib" / candidate
                       for candidate in kernel_libraries[name]
                       if (projects[name] / "out/lib" / candidate).is_file()), None)
        if kernel is None:
            choices = ", ".join(kernel_libraries[name])
            raise FileNotFoundError(f"missing {name} kernel library ({choices}); run the project's scripts/build.sh")
        load_library(torch, kernel)
        register = "libgemm_torch_register.so" if name == "gemm" else "libswiglu_torch_register.so"
        load_library(torch, projects[name] / "out/lib" / register)
    for name, register in (("rope", "librope_torch_register.so"),
                           ("rms", "librmsnorm_torch_register.so")):
        kernel = next((projects[name] / "out/lib" / candidate
                       for candidate in kernel_libraries[name]
                       if (projects[name] / "out/lib" / candidate).is_file()), None)
        if kernel is None:
            choices = ", ".join(kernel_libraries[name])
            raise FileNotFoundError(f"missing {name} kernel library ({choices}); run the project's scripts/build.sh")
        load_library(torch, kernel)
        load_library(torch, projects[name] / "out/lib" / register)
    # The GQA register library has a DT_NEEDED dependency on this kernel
    # library.  Load it explicitly so the integration runner does not depend
    # on an externally prepared LD_LIBRARY_PATH.
    load_library(torch, projects["gqa"] / "out/lib" / kernel_libraries["gqa"][0])
    load_library(torch, projects["gqa"] / "out/lib/libgqa_attention_torch_register.so")

    rope = (torch.ops.qwen_rope_custom.rope_baseline if variant == "baseline"
            else torch.ops.qwen_rope_custom_opt.rope_qk_compact)
    gqa_ns = torch.ops.gqa_attention_custom if variant == "baseline" else torch.ops.gqa_attention_optimized_custom
    return {
        "rope": rope,
        "gemm": torch.ops.gemm_custom.gemm,
        "rms": torch.ops.rmsnorm_custom.rms_norm,
        "swiglu": torch.ops.swiglu_custom.swiglu,
        "gqa": gqa_ns.gqa_attention,
    }


def npu_to_cpu(tensor):
    return tensor.to("cpu").contiguous()


def cpu_to_npu(tensor):
    return tensor.to("npu").contiguous()


def patch_linear(torch, model, gemm):
    count = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        def custom_forward(self, x):
            shape = x.shape[:-1]
            x2d = x.reshape(-1, x.shape[-1]).float().contiguous()
            y = gemm(cpu_to_npu(x2d), cpu_to_npu(self.weight.t().float().contiguous()))
            y = npu_to_cpu(y)
            if self.bias is not None:
                y = y + self.bias.float()
            return y.reshape(*shape, self.weight.shape[0]).to(dtype=x.dtype)

        module.forward = types.MethodType(custom_forward, module)
        count += 1
    return count


def patch_rmsnorm(model, rms):
    count = 0
    for module in model.modules():
        if "rmsnorm" not in module.__class__.__name__.lower() or not hasattr(module, "weight"):
            continue

        def custom_forward(self, hidden_states):
            eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
            result = rms(hidden_states.float().contiguous(), self.weight.float().contiguous(), float(eps))
            return result.to(dtype=hidden_states.dtype)

        module.forward = types.MethodType(custom_forward, module)
        count += 1
    return count


def patch_swiglu(model, swiglu):
    count = 0
    for layer in model.model.layers:
        mlp = layer.mlp

        def custom_forward(self, x):
            gate = self.gate_proj(x).float().contiguous()
            up = self.up_proj(x).float().contiguous()
            result = swiglu(cpu_to_npu(gate), cpu_to_npu(up))
            return self.down_proj(npu_to_cpu(result).to(dtype=x.dtype))

        mlp.forward = types.MethodType(custom_forward, mlp)
        count += 1
    return count


def _unpack_attention_args(args, kwargs):
    position_embeddings = kwargs.get("position_embeddings")
    attention_mask = kwargs.get("attention_mask")
    position_ids = kwargs.get("position_ids")
    past_key_value = kwargs.get("past_key_value")
    cache_position = kwargs.get("cache_position")
    if args:
        if isinstance(args[0], tuple) and len(args[0]) == 2:
            position_embeddings = args[0]
            attention_mask = args[1] if len(args) > 1 else attention_mask
            past_key_value = args[2] if len(args) > 2 else past_key_value
            cache_position = args[3] if len(args) > 3 else cache_position
        else:
            attention_mask = args[0]
            position_ids = args[1] if len(args) > 1 else position_ids
            past_key_value = args[2] if len(args) > 2 else past_key_value
            cache_position = args[5] if len(args) > 5 else cache_position
    return position_embeddings, attention_mask, position_ids, past_key_value, cache_position


def patch_attention(model, ops, variant):
    count = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        q_heads = attn.q_proj.out_features // attn.head_dim
        kv_heads = attn.k_proj.out_features // attn.head_dim

        def custom_forward(self, hidden_states, *args, _q_heads=q_heads, _kv_heads=kv_heads, **kwargs):
            position_embeddings, attention_mask, position_ids, past_key_value, cache_position = _unpack_attention_args(args, kwargs)
            batch, sequence, _ = hidden_states.shape
            q = self.q_proj(hidden_states).view(batch, sequence, _q_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(batch, sequence, _kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(batch, sequence, _kv_heads, self.head_dim).transpose(1, 2)
            if position_embeddings is None:
                cos, sin = self.rotary_emb(v, seq_len=k.shape[-2])
                if position_ids is None:
                    raise ValueError("Qwen attention requires position_ids when position_embeddings is absent")
                cos, sin = cos[position_ids], sin[position_ids]
            else:
                cos, sin = position_embeddings
            head_dim = cos.shape[-1]
            if variant == "baseline":
                cos_q = cos.unsqueeze(1).expand(-1, _q_heads, -1, -1).reshape(-1, head_dim).contiguous()
                sin_q = sin.unsqueeze(1).expand(-1, _q_heads, -1, -1).reshape(-1, head_dim).contiguous()
                cos_k = cos.unsqueeze(1).expand(-1, _kv_heads, -1, -1).reshape(-1, head_dim).contiguous()
                sin_k = sin.unsqueeze(1).expand(-1, _kv_heads, -1, -1).reshape(-1, head_dim).contiguous()
                q = ops["rope"](q.reshape(-1, head_dim).float().contiguous(), cos_q.float(), sin_q.float()).view_as(q)
                k = ops["rope"](k.reshape(-1, head_dim).float().contiguous(), cos_k.float(), sin_k.float()).view_as(k)
            else:
                q_out, k_out = ops["rope"](
                    cpu_to_npu(q.reshape(-1, head_dim).float()), cpu_to_npu(k.reshape(-1, head_dim).float()),
                    cpu_to_npu(cos.reshape(-1, head_dim).float()), cpu_to_npu(sin.reshape(-1, head_dim).float()),
                    sequence, _q_heads, _kv_heads)
                q, k = npu_to_cpu(q_out).view_as(q), npu_to_cpu(k_out).view_as(k)
            if past_key_value is not None:
                k, v = past_key_value.update(k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
            if attention_mask is not None and attention_mask.dim() != 4:
                raise ValueError("five-operator GQA path supports an unpadded causal prompt only")
            result = ops["gqa"](q.float().contiguous(), k.float().contiguous(), v.float().contiguous(), 0.0, True)
            # Newer transformers Qwen2Attention no longer exposes
            # self.hidden_size. The output projection input width is exactly
            # query_heads * head_dim, which is stable across these versions.
            result = result.transpose(1, 2).contiguous().reshape(
                batch, sequence, _q_heads * self.head_dim
            )
            # Current Qwen2DecoderLayer expects (attn_output, attn_weights);
            # the cache object, when present, is updated in place above.
            return self.o_proj(result.to(dtype=hidden_states.dtype)), None

        attn.forward = types.MethodType(custom_forward, attn)
        count += 1
    return count


def timed_forward(torch, model, input_ids, repeat):
    model(input_ids=input_ids, use_cache=False).logits  # warmup
    values, logits = [], None
    for _ in range(repeat):
        start = time.perf_counter()
        logits = model(input_ids=input_ids, use_cache=False).logits
        values.append((time.perf_counter() - start) * 1000.0)
    return logits, sum(values) / len(values), values


def run(variant, model_path, prompt, repeat, result_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_npu(torch)
    model_dir = Path(model_path).expanduser()
    local_only = model_dir.exists()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, trust_remote_code=True, local_files_only=local_only).eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]  # CPU: RMSNorm/GQA wrapper contract
    native_logits, native_ms, native_runs = timed_forward(torch, model, input_ids, repeat)
    ops = load_ops(torch, variant)
    patched = {
        "gemm_linear": patch_linear(torch, model, ops["gemm"]),
        "rmsnorm": patch_rmsnorm(model, ops["rms"]),
        "swiglu_mlp": patch_swiglu(model, ops["swiglu"]),
        "attention_rope_gqa": patch_attention(model, ops, variant),
    }
    custom_logits, custom_ms, custom_runs = timed_forward(torch, model, input_ids, repeat)
    diff = (custom_logits - native_logits).abs()
    result = {
        "variant": variant, "model": str(model_dir), "prompt_tokens": int(input_ids.size(1)), "repeat": repeat,
        "native_forward_ms": native_ms, "custom_forward_ms": custom_ms,
        "native_over_custom": native_ms / custom_ms,
        "native_runs_ms": native_runs, "custom_runs_ms": custom_runs,
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "allclose_atol_1e-2_rtol_1e-2": bool(torch.allclose(native_logits, custom_logits, atol=1e-2, rtol=1e-2)),
        "patched": patched,
        "timing_scope": "CPU model forward plus all NPU/CPU wrapper bridge copies",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(default_variant):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--result", type=Path, default=None)
    args = parser.parse_args()
    result = args.result or ROOT / f"Qwen2.5{default_variant.title()}IntegrationExperiment/results/latest.json"
    run(default_variant, args.model, args.prompt, args.repeat, result)

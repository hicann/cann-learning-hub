#!/usr/bin/env python3
"""Qwen2.5 E2E — 原生 RoPE vs 自定义 NPU RoPE (per-instance patch)"""

import argparse, os, shutil, subprocess, time, types
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys_path = str(Path(__file__).resolve().parents[1])
import sys; sys.path.insert(0, sys_path)
from torch_extension import load_torch_ops


class NpuRuntimeUnavailable(RuntimeError):
    pass


def default_model_path():
    env_model = os.environ.get("QWEN_MODEL_PATH")
    if env_model:
        return env_model

    for local_model in (
        Path.home() / "Model" / "Qwen2.5-0.5B",
        Path.home() / "Models" / "Qwen2.5-0.5B",
    ):
        if local_model.exists():
            return str(local_model)

    return "Qwen/Qwen2.5-0.5B"


def is_local_model(model_name_or_path):
    return Path(os.path.expanduser(model_name_or_path)).exists()


@torch.no_grad()
def forward_timed(model, input_ids, warmup=True):
    if warmup:
        _ = model(input_ids).logits
    t0 = time.perf_counter()
    logits = model(input_ids).logits
    return logits, time.perf_counter() - t0


def require_npu_runtime():
    if os.environ.get("FORCE_CUSTOM_NPU") == "1":
        return
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        raise NpuRuntimeUnavailable("NPU runtime unavailable: npu-smi not found")
    try:
        ret = subprocess.run(
            [npu_smi, "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired as exc:
        raise NpuRuntimeUnavailable("NPU runtime unavailable: npu-smi info timed out") from exc
    if ret.returncode != 0:
        detail = (ret.stderr or ret.stdout).strip().splitlines()
        msg = detail[-1] if detail else f"returncode={ret.returncode}"
        raise NpuRuntimeUnavailable(
            "NPU runtime unavailable before launching ACL kernel: "
            f"npu-smi info failed ({msg}). Set FORCE_CUSTOM_NPU=1 to bypass this preflight."
        )


def patch_qwen_rope(model):
    """per-instance 替换每个 layer 的 RoPE (参考 SwiGluBaselineExperiment)"""
    require_npu_runtime()
    load_torch_ops()
    rope_npu = torch.ops.qwen_rope_custom.rope_baseline

    # warmup dispatcher
    d = torch.randn(2, 64)
    rope_npu(d, d, d)

    patched = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        num_heads    = attn.q_proj.out_features // attn.head_dim
        num_kv_heads = attn.k_proj.out_features // attn.head_dim
        num_kv_groups = num_heads // num_kv_heads

        def custom_forward(self, hidden_states, *args,
                           attention_mask=None, position_ids=None,
                           past_key_value=None, output_attentions=False,
                           use_cache=False, cache_position=None,
                           position_embeddings=None,
                           _num_heads=num_heads,
                           _num_kv_heads=num_kv_heads,
                           _num_kv_groups=num_kv_groups,
                           **kwargs):
            if args:
                first = args[0]
                if isinstance(first, tuple) and len(first) == 2:
                    position_embeddings = first
                    if len(args) > 1:
                        attention_mask = args[1]
                    if len(args) > 2:
                        past_key_value = args[2]
                    if len(args) > 3:
                        cache_position = args[3]
                else:
                    attention_mask = first
                    if len(args) > 1:
                        position_ids = args[1]
                    if len(args) > 2:
                        past_key_value = args[2]
                    if len(args) > 3:
                        output_attentions = args[3]
                    if len(args) > 4:
                        use_cache = args[4]
                    if len(args) > 5:
                        cache_position = args[5]

            bsz, seq_len, _ = hidden_states.shape

            query_states = self.q_proj(hidden_states)
            key_states   = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, seq_len, _num_heads, self.head_dim).transpose(1, 2)
            key_states   = key_states.view(bsz, seq_len, _num_kv_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, seq_len, _num_kv_heads, self.head_dim).transpose(1, 2)

            if position_embeddings is None:
                kv_seq_len = key_states.shape[-2]
                if past_key_value is not None:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                if position_ids is None:
                    raise ValueError("position_ids is required when position_embeddings is not provided")
                cos = cos[position_ids]
                sin = sin[position_ids]
            else:
                cos, sin = position_embeddings

            head_dim = cos.shape[-1]

            # Q RoPE
            cos_q = cos.unsqueeze(1).expand(-1, _num_heads, -1, -1)
            sin_q = sin.unsqueeze(1).expand(-1, _num_heads, -1, -1)
            q_f = query_states.reshape(-1, head_dim).contiguous()
            cos_q_f = cos_q.reshape(-1, head_dim).contiguous()
            sin_q_f = sin_q.reshape(-1, head_dim).contiguous()
            query_states = rope_npu(q_f, cos_q_f, sin_q_f).view_as(query_states)

            # K RoPE
            cos_k = cos_q[:, :_num_kv_heads, :, :].contiguous()
            sin_k = sin_q[:, :_num_kv_heads, :, :].contiguous()
            k_f = key_states.reshape(-1, head_dim).contiguous()
            cos_k_f = cos_k.reshape(-1, head_dim).contiguous()
            sin_k_f = sin_k.reshape(-1, head_dim).contiguous()
            key_states = rope_npu(k_f, cos_k_f, sin_k_f).view_as(key_states)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs)

            if _num_kv_groups > 1:
                key_states   = key_states.repeat_interleave(_num_kv_groups, dim=1)
                value_states = value_states.repeat_interleave(_num_kv_groups, dim=1)

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            is_causal = True if causal_mask is None and seq_len > 1 else False
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
            return self.o_proj(attn_output), None, past_key_value

        attn.forward = types.MethodType(custom_forward, attn)
        patched += 1

    print(f"[PATCH] {patched} attention layers → NPU RoPE kernel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=default_model_path())
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    model_path = os.path.expanduser(args.model)
    local_files_only = is_local_model(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).eval()

    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    print(f"  hidden_size={model.config.hidden_size}")
    print(f"  num_attention_heads={model.config.num_attention_heads}")
    print(f"  num_hidden_layers={model.config.num_hidden_layers}")
    print(f"[INPUT] '{args.prompt}' → {ids.shape}")

    print("\n─ 原生 RoPE ─")
    logits_n, dt_n = forward_timed(model, ids)
    tok_n = tokenizer.decode([torch.argmax(logits_n[0, -1]).item()])

    print("\n─ 自定义 NPU Kernel ─")
    try:
        patch_qwen_rope(model)
        logits_c, dt_c = forward_timed(model, ids)
        tok_c = tokenizer.decode([torch.argmax(logits_c[0, -1]).item()])
    except NpuRuntimeUnavailable as e:
        print(f"[SKIP] {e}")
        return
    except Exception as e:
        print(f"[SKIP] {e}")
        import traceback; traceback.print_exc()
        return

    diff = (logits_c - logits_n).abs()
    print(f"\n{'='*50}")
    print(f"  原生 RoPE  time: {dt_n*1000:.1f} ms  |  next token: '{tok_n}'")
    print(f"  自定义 NPU time: {dt_c*1000:.1f} ms  |  next token: '{tok_c}'")
    print(f"  logits max_abs_diff: {diff.max().item():.6e}")
    print(f"  logits mean_abs_diff: {diff.mean().item():.6e}")
    match = tok_n == tok_c
    close = torch.allclose(logits_n, logits_c, atol=1e-2)
    print(f"  {'✅' if match else '❌'} Next token {'MATCH' if match else 'MISMATCH'}")
    print(f"  {'✅' if close else '❌'} Logits within atol=1e-2")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

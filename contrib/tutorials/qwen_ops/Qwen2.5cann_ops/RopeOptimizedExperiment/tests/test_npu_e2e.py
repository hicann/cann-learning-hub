#!/usr/bin/env python3
"""
E2E 正确性测试：NPU-resident 自定义 RoPE 算子

将 Qwen2.5-0.5B 模型加载到 NPU，用自定义 NPU-resident RoPE
替换原生实现，验证前向 pass 结果正确（token match）。

原理：
  - model.to("npu") 后所有权重、激活均在 NPU 上
  - monkey-patch apply_rotary_pos_emb → 自定义 NPU-resident op
  - 自定义 op 通过 AutogradPrivateUse1 dispatch 走零拷贝路径
    （x.data_ptr() 直接传 NPU 设备指针，无 H2D/D2H）

用法:
  cd ~/Projects/QwenRoPeCustomOpt
  bash scripts/run_test.sh tests/test_npu_e2e.py

依赖:
  - CANN 8.5 + torch_npu 2.10 (GE 初始化正常)
  - 本地模型 ~/Models/Qwen2.5-0.5B
"""

import os
import sys
from pathlib import Path

# ── 环境 ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _setup_env import auto_setup; auto_setup()

import torch
import torch_npu

from torch_extension import load_torch_ops


def test_e2e(model_path="~/Models/Qwen2.5-0.5B", prompt="The capital of France is"):
    """
    E2E 正确性测试：
    1. 加载模型到 NPU
    2. 原生 NPU RoPE forward，记录输出 token
    3. Monkey-patch 为自定义 NPU-resident RoPE
    4. 再次加载模型到 NPU（全新实例），forward，比较 token

    返回: (passed: bool, native_token: str, custom_token: str, native_ms: float, custom_ms: float)
    """

    # ── 延迟 import，避免 CANN 未 source 时报错 ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.models.qwen2.modeling_qwen2 as mq

    load_torch_ops()

    # ── 加载 tokenizer ──
    mp = os.path.expanduser(model_path)
    if not Path(mp).exists():
        raise FileNotFoundError(f"模型不存在: {mp}")
    print(f"[模型] {mp}")

    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, local_files_only=True)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("npu")
    seq_len = input_ids.shape[1]
    print(f"[输入] seq={seq_len}  prompt='{prompt}'")

    # ── 原生 NPU RoPE ──
    print("[原生] 加载模型到 NPU ...")
    t0 = __import__("time").perf_counter()
    m_native = AutoModelForCausalLM.from_pretrained(
        mp, torch_dtype=torch.float32,
        trust_remote_code=True, local_files_only=True,
    ).eval().to("npu")

    with torch.no_grad():
        logits_native = m_native(input_ids).logits
    torch_npu.npu.synchronize()
    dt_native = (__import__("time").perf_counter() - t0) * 1000

    token_native = tokenizer.decode([torch.argmax(logits_native[0, -1]).item()])
    print(f"  token='{token_native}'  耗时={dt_native:.0f}ms")
    del m_native

    # ── 自定义 NPU-resident RoPE ──
    print("[自定义] 注册 NPU-resident RoPE operator ...")

    # 获取自定义算子 + 预热（吸收 ~850ms 首次 dispatcher 开销）
    rope_qk = torch.ops.qwen_rope_custom_opt.rope_qk_compact
    rope_qk(
        torch.randn(56, 64, device="npu"),
        torch.randn( 8, 64, device="npu"),
        torch.randn( 4, 64, device="npu"),
        torch.randn( 4, 64, device="npu"),
        4, 14, 2,
    )

    # 自定义 RoPE wrapper — 适配 Qwen2Attention 旧 API
    #   apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)
    #   q: [batch, num_heads, seq_len, head_dim]
    #   k: [batch, num_kv_heads, seq_len, head_dim]
    def custom_rope(q, k, cos, sin, unsqueeze_dim=1):
        head_dim = cos.shape[-1]
        q_heads = q.shape[1]
        k_heads = k.shape[1]
        seq = q.shape[2]

        # Flatten → [tokens, head_dim]
        cos_f = cos.reshape(-1, head_dim).contiguous()
        sin_f = sin.reshape(-1, head_dim).contiguous()
        q_f = q.reshape(-1, head_dim).contiguous()
        k_f = k.reshape(-1, head_dim).contiguous()

        # 调用自定义 NPU-resident fused QK RoPE
        # ★ tensor 全在 NPU 上 → AutogradPrivateUse1 dispatch → 零 H2D/D2H
        q_rope, k_rope = rope_qk(q_f, k_f, cos_f, sin_f, seq, q_heads, k_heads)

        return q_rope.view_as(q), k_rope.view_as(k)

    # Monkey-patch 全局 apply_rotary_pos_emb
    mq.apply_rotary_pos_emb = custom_rope

    print("[自定义] 加载模型到 NPU ...")
    t0 = __import__("time").perf_counter()
    m_custom = AutoModelForCausalLM.from_pretrained(
        mp, torch_dtype=torch.float32,
        trust_remote_code=True, local_files_only=True,
    ).eval().to("npu")

    with torch.no_grad():
        logits_custom = m_custom(input_ids).logits
    torch_npu.npu.synchronize()
    dt_custom = (__import__("time").perf_counter() - t0) * 1000

    token_custom = tokenizer.decode([torch.argmax(logits_custom[0, -1]).item()])
    print(f"  token='{token_custom}'  耗时={dt_custom:.0f}ms")
    del m_custom

    # ── 结果 ──
    passed = token_native == token_custom
    print(f"\n{'─' * 50}")
    print(f"  原生 token:   {token_native}")
    print(f"  自定义 token: {token_custom}")
    print(f"  Token 匹配:   {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"{'─' * 50}")

    return passed, token_native, token_custom, dt_native, dt_custom


def main():
    print("=" * 60)
    print("  NPU-Resident RoPE E2E 正确性测试")
    print("=" * 60)

    try:
        passed, _, _, _, _ = test_e2e()
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

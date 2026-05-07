"""Minimal Sana-Video NPU compatibility helpers for the tutorial."""

import importlib
import sys
import types

import torch
import torch.nn.functional as F


def install_triton_rms_norm_stub() -> None:
    """Avoid importing Triton-only RMSNorm kernels on NPU tutorial environments."""
    module_name = "diffusion.model.dc_ae.efficientvit.models.nn.triton_rms_norm"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class TritonRMSNorm2dFunc:
        @staticmethod
        def apply(*args, **kwargs):
            raise RuntimeError("triton rms norm is not available in this NPU tutorial workspace")

    module.__dict__["TritonRMSNorm2dFunc"] = TritonRMSNorm2dFunc
    sys.modules[module_name] = module


def install_wan_rotary_complex64_fix() -> None:
    """Keep Wan rotary embedding generation on a dtype that works on NPU."""
    sana_blocks = importlib.import_module("diffusion.model.nets.sana_blocks")

    def forward(self, fhw: torch.Tensor, device: torch.device) -> torch.Tensor:
        ppf, pph, ppw = fhw

        freqs = self.freqs.to(dtype=torch.complex64).to(device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        return torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)

    sana_blocks.WanRotaryPosEmbed.forward = forward


def install_cross_attention_contiguous_fix() -> None:
    """Keep cross-attention outputs contiguous before view on NPU."""
    sana_blocks = importlib.import_module("diffusion.model.nets.sana_blocks")

    def cross_attention_forward(self, x, cond, mask=None):
        batch_size, num_tokens, channels = x.shape
        first_dim = 1 if sana_blocks._xformers_available else batch_size

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(first_dim, -1, 2, channels)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(first_dim, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(first_dim, -1, self.num_heads, self.head_dim)
        v = v.view(first_dim, -1, self.num_heads, self.head_dim)

        if sana_blocks._xformers_available:
            attn_bias = None
            if mask is not None:
                attn_bias = sana_blocks.xformers.ops.fmha.BlockDiagonalMask.from_seqlens([num_tokens] * batch_size, mask)
            x = sana_blocks.xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if mask is not None and mask.ndim == 2:
                mask = (1 - mask.to(q.dtype)) * -10000.0
                mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def cross_attention_image_forward(self, x, cond, mask=None, image_embeds=None):
        batch_size, _, channels = x.shape

        q = self.q_linear(x)
        text_kv = self.kv_linear(cond).view(batch_size, -1, 2, channels)
        text_k, text_v = text_kv.unbind(2)

        image_kv = self.image_kv_linear(image_embeds).view(batch_size, -1, 2, channels)
        image_k, image_v = image_kv.unbind(2)

        q = self.q_norm(q).view(batch_size, -1, self.num_heads, self.head_dim)
        text_k = self.k_norm(text_k).view(batch_size, -1, self.num_heads, self.head_dim)
        text_v = text_v.view(batch_size, -1, self.num_heads, self.head_dim)
        image_k = self.image_k_norm(image_k).view(batch_size, -1, self.num_heads, self.head_dim)
        image_v = image_v.view(batch_size, -1, self.num_heads, self.head_dim)

        q, text_k, text_v = q.transpose(1, 2), text_k.transpose(1, 2), text_v.transpose(1, 2)
        image_k, image_v = image_k.transpose(1, 2), image_v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(q.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, text_k, text_v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x + F.scaled_dot_product_attention(q, image_k, image_v, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    sana_blocks.MultiHeadCrossAttention.forward = cross_attention_forward
    sana_blocks.MultiHeadCrossAttentionImageEmbed.forward = cross_attention_image_forward


def install_compatibility_shims() -> None:
    install_triton_rms_norm_stub()
    install_wan_rotary_complex64_fix()
    install_cross_attention_contiguous_fix()

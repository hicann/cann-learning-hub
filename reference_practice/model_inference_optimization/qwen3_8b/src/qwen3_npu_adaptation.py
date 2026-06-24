"""Qwen3-8B recipes tutorial validation utilities."""

from __future__ import annotations

import unicodedata
from typing import Iterable


def qwen3_rmsnorm_reference(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Reference implementation of Qwen3 RMSNorm math."""
    import torch

    input_dtype = hidden_states.dtype
    hidden_states_fp32 = hidden_states.to(torch.float32)
    variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
    normalized = hidden_states_fp32 * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(input_dtype)


def qwen3_add_rmsnorm_reference(
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for residual add followed by Qwen3 RMSNorm."""
    summed = residual + hidden_states
    return qwen3_rmsnorm_reference(summed, weight, epsilon=epsilon), summed


def run_rmsnorm_precision_check(
    shapes: Iterable[Iterable[int]] | None = None,
    dtype: str = "bfloat16",
    device: str = "npu:0",
    seed: int = 0,
    epsilon: float = 1e-6,
    atol: float = 1e-2,
) -> list[dict]:
    """Compare Qwen3 RMSNorm reference math with torch_npu.npu_rms_norm."""
    import torch
    import torch_npu

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    check_shapes = list(shapes or ([1, 1, 4096], [1, 16, 4096], [1, 16, 32, 128]))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(device)
        torch.npu.manual_seed_all(seed)
    torch.manual_seed(seed)

    rows = []
    for shape_values in check_shapes:
        shape = tuple(int(v) for v in shape_values)
        hidden_size = shape[-1]
        try:
            hidden_states = torch.randn(shape, device=device, dtype=torch_dtype)
            weight = torch.ones(hidden_size, device=device, dtype=torch_dtype)
            ref = qwen3_rmsnorm_reference(hidden_states, weight, epsilon=epsilon)
            opt = torch_npu.npu_rms_norm(hidden_states, weight, epsilon=epsilon)[0]
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize()
            diff = (ref.float() - opt.float()).abs()
            if not torch.isfinite(diff).all().item():
                rows.append(
                    {
                        "shape": list(shape),
                        "dtype": dtype,
                        "status": "failed",
                        "error": "non_finite_diff",
                    }
                )
                continue
            denom = ref.float().abs().clamp_min(1e-6)
            max_abs_diff = float(diff.max().item())
            mean_abs_diff = float(diff.mean().item())
            max_rel_diff = float((diff / denom).max().item())
            warnings = []
            if max_abs_diff > atol:
                warnings.append("max_abs_diff_exceeds_atol")
            rows.append(
                {
                    "shape": list(shape),
                    "dtype": dtype,
                    "status": "ok" if not warnings else "warning",
                    "warnings": warnings,
                    "atol": atol,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                    "max_rel_diff": max_rel_diff,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "shape": list(shape),
                    "dtype": dtype,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def run_add_rmsnorm_precision_check(
    shapes: Iterable[Iterable[int]] | None = None,
    dtype: str = "bfloat16",
    device: str = "npu:0",
    seed: int = 0,
    epsilon: float = 1e-6,
    atol: float = 1e-2,
) -> list[dict]:
    """Compare residual add + RMSNorm reference math with torch_npu.npu_add_rms_norm."""
    import torch
    import torch_npu

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    check_shapes = list(shapes or ([1, 1, 4096], [1, 16, 4096], [1, 16, 32, 128]))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(device)
        torch.npu.manual_seed_all(seed)
    torch.manual_seed(seed)

    rows = []
    for shape_values in check_shapes:
        shape = tuple(int(v) for v in shape_values)
        hidden_size = shape[-1]
        try:
            residual = torch.randn(shape, device=device, dtype=torch_dtype)
            hidden_states = torch.randn(shape, device=device, dtype=torch_dtype)
            weight = torch.ones(hidden_size, device=device, dtype=torch_dtype)
            ref_norm, ref_sum = qwen3_add_rmsnorm_reference(residual, hidden_states, weight, epsilon=epsilon)
            opt_norm, _, opt_sum = torch_npu.npu_add_rms_norm(residual, hidden_states, weight, epsilon)
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize()
            norm_diff = (ref_norm.float() - opt_norm.float()).abs()
            sum_diff = (ref_sum.float() - opt_sum.float()).abs()
            if not torch.isfinite(norm_diff).all().item() or not torch.isfinite(sum_diff).all().item():
                rows.append(
                    {
                        "shape": list(shape),
                        "dtype": dtype,
                        "status": "failed",
                        "error": "non_finite_diff",
                    }
                )
                continue
            denom = ref_norm.float().abs().clamp_min(1e-6)
            max_norm_abs_diff = float(norm_diff.max().item())
            mean_norm_abs_diff = float(norm_diff.mean().item())
            max_norm_rel_diff = float((norm_diff / denom).max().item())
            max_sum_abs_diff = float(sum_diff.max().item())
            warnings = []
            if max_norm_abs_diff > atol:
                warnings.append("max_norm_abs_diff_exceeds_atol")
            if max_sum_abs_diff > atol:
                warnings.append("max_sum_abs_diff_exceeds_atol")
            rows.append(
                {
                    "shape": list(shape),
                    "dtype": dtype,
                    "status": "ok" if not warnings else "warning",
                    "warnings": warnings,
                    "atol": atol,
                    "max_norm_abs_diff": max_norm_abs_diff,
                    "mean_norm_abs_diff": mean_norm_abs_diff,
                    "max_norm_rel_diff": max_norm_rel_diff,
                    "max_sum_abs_diff": max_sum_abs_diff,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "shape": list(shape),
                    "dtype": dtype,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def analyze_generated_text(
    text: str,
    generated_tokens: int | None = None,
    max_new_tokens: int | None = None,
    expect_chinese: bool = False,
) -> dict:
    """Check decoded text for basic readability and tokenization artifacts."""
    issues = []
    warnings = []
    stripped = text.strip()

    if not stripped:
        issues.append("empty_text")
    if "\ufffd" in text:
        issues.append("unicode_replacement_character")
    if "<|" in text or "|>" in text:
        issues.append("raw_special_token_marker")
    if any(unicodedata.category(ch).startswith("C") and ch not in "\n\r\t" for ch in text):
        issues.append("control_character")

    if stripped:
        chinese_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in stripped)
        non_space_chars = sum(not ch.isspace() for ch in stripped)
        printable_chars = sum(ch.isprintable() for ch in stripped)
        printable_ratio = printable_chars / max(len(stripped), 1)
        chinese_ratio = chinese_chars / max(non_space_chars, 1)
    else:
        chinese_ratio = 0.0
        printable_ratio = 0.0

    max_char_run = 0
    current_run = 0
    previous = None
    for ch in stripped:
        if ch == previous:
            current_run += 1
        else:
            current_run = 1
            previous = ch
        max_char_run = max(max_char_run, current_run)
    if max_char_run >= 6:
        issues.append("excessive_repeated_character")

    for width in (2, 3, 4):
        repeat_count = 1
        previous_piece = None
        for i in range(0, max(len(stripped) - width + 1, 0), width):
            piece = stripped[i : i + width]
            if piece == previous_piece:
                repeat_count += 1
            else:
                repeat_count = 1
                previous_piece = piece
            if repeat_count >= 4:
                issues.append(f"excessive_repeated_{width}gram")
                break

    if printable_ratio < 0.95:
        issues.append("low_printable_ratio")
    if expect_chinese and stripped:
        if chinese_ratio < 0.2:
            warnings.append("low_chinese_ratio_for_chinese_prompt")
    if (
        max_new_tokens is not None
        and generated_tokens is not None
        and generated_tokens >= max_new_tokens
        and stripped
        and stripped[-1] not in "。！？!?；;：:，,、.）)]}\"'”’"
    ):
        warnings.append("possibly_truncated_by_max_new_tokens")

    return {
        "status": "ok" if not issues else "failed",
        "issues": sorted(set(issues)),
        "warnings": sorted(set(warnings)),
        "char_count": len(stripped),
        "chinese_ratio": chinese_ratio,
        "printable_ratio": printable_ratio,
        "max_repeated_char_run": max_char_run,
    }


def run_generation_text_sanity_check(
    rows: Iterable[dict],
    max_new_tokens: int | None = None,
) -> dict:
    """Check generated text rows for normal decoded output."""
    details = []
    for row in rows:
        prompt_text = str(row.get("prompt", ""))
        expect_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in prompt_text)
        detail = analyze_generated_text(
            str(row.get("generated_text", "")),
            generated_tokens=row.get("generated_tokens"),
            max_new_tokens=max_new_tokens,
            expect_chinese=expect_chinese,
        )
        detail["prompt_id"] = row.get("prompt_id")
        details.append(detail)

    failed = [item for item in details if item["status"] != "ok"]
    warnings = [item for item in details if item["warnings"]]
    return {
        "status": "ok" if not failed else "failed",
        "num_checked": len(details),
        "num_failed": len(failed),
        "num_with_warnings": len(warnings),
        "details": details,
    }

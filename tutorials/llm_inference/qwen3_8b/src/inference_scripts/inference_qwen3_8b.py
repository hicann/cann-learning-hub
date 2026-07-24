#!/usr/bin/env python3
"""Shared parsing and metrics helpers for the Qwen3-8B recipes notebooks."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("QWEN3_8B_DOWNLOAD_BACKEND", "modelscope")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

SRC_ROOT = Path(__file__).resolve().parents[1]
RECIPE_ROOT = Path(__file__).resolve().parent / "recipe_qwen3_8b"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@contextmanager
def maybe_silence_output(enabled: bool = False):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def read_recipe_default_prompts() -> list[dict[str, Any]]:
    prompt_path = RECIPE_ROOT / "dataset" / "default_prompt.json"
    try:
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        text = data["text"]
    except Exception as exc:
        raise RuntimeError(f"Failed to read recipes default prompt from {prompt_path}") from exc

    prompt_texts = text if isinstance(text, list) else [text]
    return [
        {
            "id": f"default_{index}",
            "prompt": str(prompt),
            "messages": [{"role": "user", "content": str(prompt)}],
            "recipe_prompt": str(prompt),
        }
        for index, prompt in enumerate(prompt_texts)
    ]


def messages_to_recipe_prompt(messages: list[dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0].get("content", ""))
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def resolve_model_path(model_path: str, quiet_model_io: bool = False) -> tuple[str, str]:
    local_path = Path(model_path).expanduser()
    if local_path.exists():
        return str(local_path), "local_path"

    backend = os.environ.get("QWEN3_8B_DOWNLOAD_BACKEND", "modelscope").strip().lower()
    if backend in {"modelscope", "ms"}:
        print(f"[模型下载] 通道 = ModelScope，模型 = {model_path}", flush=True)
        print("[模型下载] 正在检查缓存/下载权重。", flush=True)
        try:
            from modelscope import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "ModelScope 下载通道不可用。请安装 modelscope，或设置 "
                "QWEN3_8B_MODEL_PATH 指向本地模型目录，或设置 "
                "QWEN3_8B_DOWNLOAD_BACKEND=huggingface。"
            ) from exc
        with maybe_silence_output(quiet_model_io):
            resolved = snapshot_download(model_path, max_workers=1)
        print(f"[模型下载] 完成，路径 = {resolved}", flush=True)
        return str(resolved), "modelscope"

    if backend in {"huggingface", "hf"}:
        print(f"[模型下载] 通道 = Hugging Face，模型 = {model_path}", flush=True)
        return model_path, "huggingface"

    raise ValueError("QWEN3_8B_DOWNLOAD_BACKEND 只能是 'modelscope' 或 'huggingface'。")


def print_tail(text: str, max_lines: int = 80) -> None:
    lines = text.splitlines()
    for line in lines[-max_lines:]:
        print(line, flush=True)


def find_recipe_case_log(yaml_path: Path) -> Path | None:
    date = time.strftime("%Y%m%d")
    expected = RECIPE_ROOT / "models" / "qwen" / "res" / date / f"qwen3_8b_{yaml_path.stem}" / "log_0.log"
    if expected.is_file():
        return expected
    logs = sorted((RECIPE_ROOT / "models" / "qwen" / "res").glob("**/log_0.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse_recipe_log(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.is_file():
        return {
            "generated_texts": [],
            "prefill_time_s": None,
            "decode_step_times_s": [],
            "decode_average_time_ms": None,
            "recipe_log_path": None,
        }

    text = log_path.read_text(encoding="utf-8", errors="replace")
    generated = []
    for match in re.finditer(r"Request\s+(\d+):\s+outputs:\s+(.*)", text):
        generated.append((int(match.group(1)), match.group(2).strip()))
    generated_texts = [item[1] for item in sorted(generated, key=lambda item: item[0])]

    prefill_times = [
        float(match.group(1)) / 1000.0
        for match in re.finditer(r"Inference time \(prefill\):\s+([0-9.]+)\s+ms", text)
    ]
    decode_times = [
        float(match.group(1)) / 1000.0
        for match in re.finditer(r"Inference time \(decode\):\s+([0-9.]+)\s+ms", text)
    ]
    decode_average_ms = None
    avg_match = re.search(r"decode average inference time cost is\s+([0-9.]+)\s+ms", text)
    if avg_match:
        decode_average_ms = float(avg_match.group(1))

    return {
        "generated_texts": generated_texts,
        "prefill_time_s": sum(prefill_times) if prefill_times else None,
        "decode_step_times_s": decode_times,
        "decode_average_time_ms": decode_average_ms,
        "recipe_log_path": str(log_path),
    }


def ensure_recipe_log_ok(log_path: Path | None) -> None:
    if log_path is None or not log_path.is_file():
        raise RuntimeError("Recipe inference log_0.log was not found.")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failure_markers = ("Traceback (most recent call last)", "RuntimeError:", "ImportError:", "[ERROR]")
    if any(marker in text for marker in failure_markers):
        print("[Recipe 推理] log_0.log 检测到错误，最近日志如下：", flush=True)
        print_tail(text)
        raise RuntimeError(f"Recipe inference failed; see {log_path}")


def count_prompt_tokens(tokenizer, prompts: list[dict[str, Any]], enable_thinking: bool, max_input_tokens: int) -> list[int]:
    counts = []
    for item in prompts:
        if "recipe_prompt" in item:
            encoded = tokenizer(
                item["recipe_prompt"],
                truncation=True,
                max_length=max_input_tokens,
                return_attention_mask=False,
            )
            counts.append(len(encoded.input_ids))
            continue
        messages = item["messages"]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt_text, truncation=True, max_length=max_input_tokens, return_attention_mask=False)
        counts.append(len(encoded.input_ids))
    return counts


def build_output_rows(
    prompts: list[dict[str, Any]],
    generated_texts: list[str],
    tokenizer,
) -> list[dict[str, Any]]:
    rows = []
    for index, prompt in enumerate(prompts):
        text = generated_texts[index] if index < len(generated_texts) else ""
        token_count = len(tokenizer(text, return_attention_mask=False).input_ids) if text else 0
        rows.append(
            {
                "prompt_id": prompt["id"],
                "prompt": prompt.get("recipe_prompt") or messages_to_recipe_prompt(prompt["messages"]),
                "generated_tokens": int(token_count),
                "generated_text": text,
            }
        )
    return rows

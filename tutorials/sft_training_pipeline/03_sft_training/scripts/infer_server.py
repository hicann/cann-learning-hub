# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal OpenAI-compatible server for Qwen3-1.7B on Ascend NPU.
Usage: python3 scripts/infer_server.py --model ./outputs/checkpoint_wordle_sft/step-20
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .to("npu:0")
        .eval()
    )
    return tok, model


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req.get("messages", [])
        text = self.server.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.server.tokenizer(text, return_tensors="pt").to("npu:0")
        t0 = time.time()
        out = self.server.model.generate(
            **inputs,
            max_new_tokens=req.get("max_tokens", 1024),
            temperature=req.get("temperature", 0.6),
            do_sample=True,
            pad_token_id=self.server.tokenizer.eos_token_id,
        )
        reply = self.server.tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        self._json(
            {
                "id": f"chatcmpl-{int(t0)}",
                "object": "chat.completion",
                "created": int(t0),
                "model": req.get("model", "qwen3"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": reply},
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
        print(f"[{time.time() - t0:.1f}s] {reply[:80]}...")

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="./assets/hf/Qwen3-1.7B")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server.tokenizer, server.model = load(args.model)
    server.serve_forever()

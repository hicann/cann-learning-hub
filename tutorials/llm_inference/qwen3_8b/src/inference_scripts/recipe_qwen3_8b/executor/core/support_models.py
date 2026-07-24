# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Lazy registry mapping model name to (ForCausalLM, [ModelMTP,] Config) classes."""

import importlib

_specs: dict[str, list[tuple[str, str]]] = {
    "qwen3_8b": [
        ("models.qwen.models.modeling_qwen", "QwenForCausalLM"),
        ("models.qwen.models.configuration_qwen", "Qwen3Config"),
    ],
}


def load_model_classes(name: str) -> tuple:
    if name not in _specs:
        raise ValueError(f"Unsupported model: {name}")
    try:
        return tuple(getattr(importlib.import_module(m), a) for m, a in _specs[name])
    except Exception as e:
        raise ImportError(f"failed to load model '{name}': {e}") from e

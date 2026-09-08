# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.0/vllm/model_executor/layers/quantization/base_config.py
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import nn

QuantizationMethods = str
QUANTIZATION_METHODS: list[str] = []


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, *weight_args,
                       **extra_weight_attrs):
        """
        Create weights for a layer.
        The weights will be set as attributes of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """
        Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer.
        """
        raise NotImplementedError

    # Not required functions
    def embedding(self, layer: torch.nn.Module, *args,
                  **kwargs) -> torch.Tensor:
        """
        Gather embeddings in the layer based on indices in the input tensor.
        Expects create_weights to have been called before on the layer.
        """
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module, **kwargs) -> None:
        """
        Process the weight after loading.
        This can be used for example, to transpose weights for computation.
        """
        return


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    def __init__(self):
        super().__init__()
        # mapping is updated by models as they initialize
        self.packed_modules_mapping: dict[str, list[str]] = dict()
        # quant_mode default w16a16, will be initalized in function from_config()
        self.mm_quant_mode = "w16a16"
        self.gmm_quant_mode = "w16a16"
        self.kv_cache_quant_mode = "unquant"

    @abstractmethod
    def get_name(self) -> QuantizationMethods:
        """Name of the quantization method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        """
           Detects if this quantization method can support a given checkpoint
           format by overriding the user specified quantization method --
           this method should only be overwritten by subclasses in exceptional
           circumstances
        """
        return None

    @staticmethod
    def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's "
                         "quantization config.")

    @staticmethod
    def get_from_keys_or(config: dict[str, Any], keys: list[str],
                         default: Any) -> Any:
        """Get a optional value from the model's quantization config."""
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        raise NotImplementedError

    def get_cache_scale(self, name: str) -> Optional[str]:
        return None

    def set_quant_mode(self, name: str, value: str) -> None:
        setattr(self, name, value)


# Adapted from vllm.model_executor.layers.quantization.get_quantization_config
def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    raise ValueError(
        f"Quantization method {quantization!r} is not included in the "
        "Qwen3-1.7B chapters 01-04 tutorial subset."
    )


# Adapted from vllm.model_executor.model_loader.weight_utils.get_quant_config
def get_quant_config(hf_config, quantization, model_path) -> QuantizationConfig:
    return get_quantization_config(quantization)()

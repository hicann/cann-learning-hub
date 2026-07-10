# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

#!/usr/bin/env python3
import argparse
import difflib
import logging
import sys
from pathlib import Path


VALID_MODES = ("offline",)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def error(message):
    logger.error(message)
    return 1


def suggest(value, candidates):
    matches = difflib.get_close_matches(value, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def validate_model(models_root, model):
    if not model:
        return error("--model is required.")

    model_path = Path(model)
    if model_path.name != model or model_path.is_absolute():
        return error(f"--model must be a directory name under models/, got: {model}")

    target = models_root / model
    if target.is_dir():
        return 0

    available = sorted(path.name for path in models_root.iterdir() if path.is_dir()) if models_root.is_dir() else []
    hint = suggest(model, available)
    if hint:
        return error(f"model directory not found: {target}. Did you mean '{hint}'?")
    return error(f"model directory not found: {target}")


def validate_mode(mode):
    if mode in VALID_MODES:
        return 0

    hint = suggest(mode, VALID_MODES)
    if hint:
        return error(f"--mode must be offline, got: {mode}. Did you mean '{hint}'?")
    return error(f"--mode must be offline, got: {mode}")


def validate_file(path, label):
    if not path:
        return error(f"{label} is required.")

    target = Path(path)
    if target.is_file():
        return 0
    return error(f"{label} file not found: {target}")


def validate_args(args):
    models_root = Path(args.models_root).resolve()

    for rc in (validate_model(models_root, args.model), validate_mode(args.mode)):
        if rc:
            return rc

    return validate_file(args.yaml, "YAML")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate executor/scripts/infer.sh arguments.")
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--yaml", default="")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(validate_args(parse_args()))

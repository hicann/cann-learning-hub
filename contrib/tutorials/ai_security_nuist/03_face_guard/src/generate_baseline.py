# -*- coding: utf-8 -*-
"""
第一阶段：用预训练 ProGAN 直接生成基线结果。

加载 celebaCropped 128×128 预训练权重，固定 16 个潜变量，
生成 outputs/baseline/baseline_grid.png，并保存 fixed_latent/fixed_z.pt
供后续微调对比使用。

设备自适应：NPU > CUDA > CPU。本地无 NPU 时可在 CPU 上烟测。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保能 import 同目录下的 _common
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    PROJECT_DIR,
    OUTPUT_DIR,
    setup_device,
    load_pgans,
    get_or_create_fixed_z,
    generate_grid,
    DEFAULT_NUM_FIXED,
    DEFAULT_NROW,
    log,
)


def main() -> int:
    dev = setup_device()
    model = load_pgans(base_lr=1e-5, useGPU=dev["useGPU"])
    fixed_z = get_or_create_fixed_z(model, n=DEFAULT_NUM_FIXED)

    out_dir = OUTPUT_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_grid(model, fixed_z, out_dir / "baseline_grid.png", nrow=DEFAULT_NROW)

    # 同时保存一份微调前的完整 checkpoint（step_0000）
    step0 = OUTPUT_DIR / "step_0000"
    step0.mkdir(parents=True, exist_ok=True)
    torch_save_step0(model, step0 / "progan_finetuned.pth")
    log("baseline done.")
    return 0


def torch_save_step0(model, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.getStateDict(), path)
    log(f"saved step_0000 checkpoint -> {path}")


if __name__ == "__main__":
    import torch  # noqa: F401
    sys.exit(main())

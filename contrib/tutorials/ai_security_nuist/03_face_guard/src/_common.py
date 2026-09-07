# -*- coding: utf-8 -*-
"""共享工具：设备选择、NPU 兼容、模型加载、固定潜变量、样本生成。

设备优先级：NPU (torch_npu) > CUDA > CPU。
NPU 模式下导入 torch_npu.contrib.transfer_to_npu，把旧 GAN Zoo 中的
.cuda() / torch.cuda.* 映射到 NPU，必须在加载模型代码之前完成。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

# src/ 目录（本文件所在目录）；项目根 = src/ 的上一级
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# 代码与资产均在 src/ 下；实验产出 answer/ 与 images/ 在项目根
REPO_DIR = SRC_DIR / "code" / "pytorch_GAN_zoo"
DATA_DIR = SRC_DIR / "data" / "aigibench_128"
CKPT_PATH = SRC_DIR / "pretrained" / "celebaCropped_s5_i83000-2b0acc76.pth"
OUTPUT_DIR = PROJECT_DIR / "answer"
FIXED_Z_PATH = SRC_DIR / "fixed_latent" / "fixed_z.pt"

# 默认实验参数（执行文档）
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-5
DEFAULT_TOTAL_STEPS = 2000
DEFAULT_SAVE_EVERY = 250
DEFAULT_SAMPLE_EVERY = 250
DEFAULT_NUM_FIXED = 16          # 固定潜变量数量
DEFAULT_NROW = 4                # 网格列数


def log(msg: str) -> None:
    print(f"[progan] {msg}", flush=True)


def setup_device() -> dict:
    """选择设备并完成必要的环境初始化。返回 {mode, useGPU, device}。

    mode: 'npu' | 'cuda' | 'cpu'
    """
    # ---- NPU ----
    try:
        import torch_npu  # noqa: F401
        import torch_npu.contrib.transfer_to_npu  # noqa: F401  # 必须在加载模型前
        if torch.npu.is_available():
            torch.npu.set_device(0)
            dev = torch.device("npu:0")
            log(f"device: NPU  torch_npu={torch_npu.__version__}  count={torch.npu.device_count()}")
            return {"mode": "npu", "useGPU": True, "device": dev}
        log("torch_npu imported but NPU not available, fallback to CUDA/CPU")
    except Exception as exc:
        log(f"torch_npu unavailable ({exc!r}), try CUDA/CPU")

    # ---- CUDA ----
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        log(f"device: CUDA  count={torch.cuda.device_count()}")
        return {"mode": "cuda", "useGPU": True, "device": dev}

    # ---- CPU ----
    dev = torch.device("cpu")
    log("device: CPU")
    return {"mode": "cpu", "useGPU": False, "device": dev}


def load_pgans(base_lr: float = DEFAULT_LR, useGPU: bool = False) -> "ProgressiveGAN":
    """从本地 GAN Zoo 仓库构建 PGAN 并加载预训练 checkpoint。"""
    if not REPO_DIR.exists():
        raise FileNotFoundError(f"GAN Zoo repo not found: {REPO_DIR}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {CKPT_PATH}")

    log(f"torch.hub.load local: {REPO_DIR}")
    model = torch.hub.load(
        str(REPO_DIR),
        "PGAN",
        source="local",
        pretrained=False,
        useGPU=useGPU,
        config={"baseLearningRate": base_lr},
    )

    log(f"loading checkpoint: {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    log("checkpoint loaded")
    return model


def get_or_create_fixed_z(model, n: int = DEFAULT_NUM_FIXED) -> torch.Tensor:
    """复用 fixed_latent/fixed_z.pt，不存在则用模型 buildNoiseData 生成并保存。"""
    FIXED_Z_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FIXED_Z_PATH.exists():
        z = torch.load(FIXED_Z_PATH, map_location="cpu", weights_only=False)
        log(f"loaded fixed_z from {FIXED_Z_PATH}  shape={tuple(z.shape)}")
        return z
    z, _ = model.buildNoiseData(n)
    z = z.detach().cpu()
    torch.save(z, FIXED_Z_PATH)
    log(f"created fixed_z -> {FIXED_Z_PATH}  shape={tuple(z.shape)}")
    return z


@torch.no_grad()
def generate_grid(model, fixed_z: torch.Tensor, out_path: Path, nrow: int = DEFAULT_NROW) -> None:
    """用平均生成器 (getAvG) 生成并保存网格 PNG。PGAN 输出 [-1,1]。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.netG.eval()
    images = model.test(fixed_z, getAvG=True, toCPU=True)
    images = images.clamp(-1, 1)
    save_image((images + 1) / 2, out_path, nrow=nrow)
    log(f"saved grid -> {out_path}")


def save_checkpoint(model, out_path: Path) -> None:
    """用 GAN Zoo 的 getStateDict() 保存（含 netG/netD/avgG/config）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.getStateDict(), out_path)
    log(f"saved checkpoint -> {out_path}")

# -*- coding: utf-8 -*-
"""
第二阶段：用 AIGIBench 人脸微调 ProGAN。

流程：
  1. 设备初始化 (NPU>CUDA>CPU)，NPU 下自动启用 transfer_to_npu 兼容。
  2. 加载预训练 PGAN + celeba checkpoint。
  3. 加载 data/aigibench_128 数据，训练时随机水平翻转，归一化到 [-1,1]。
  4. 用固定潜变量在 step=0,250,500,1000,2000 生成对比网格并保存 checkpoint。
  5. loss 写入 answer/loss_log.csv。

注意：保存 checkpoint 使用 model.getStateDict()（GAN Zoo 格式，含 netG/netD/avgG/config），
      而非 nn.Module.state_dict()。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    PROJECT_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    setup_device,
    load_pgans,
    get_or_create_fixed_z,
    generate_grid,
    save_checkpoint,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_TOTAL_STEPS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_NUM_FIXED,
    DEFAULT_NROW,
    log,
)


class FaceDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image)


def save_samples(model, fixed_z, step: int) -> None:
    out_dir = OUTPUT_DIR / f"step_{step:04d}"
    generate_grid(model, fixed_z, out_dir / "grid.png", nrow=DEFAULT_NROW)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--total_steps", type=int, default=DEFAULT_TOTAL_STEPS)
    ap.add_argument("--save_every", type=int, default=DEFAULT_SAVE_EVERY)
    ap.add_argument("--sample_every", type=int, default=DEFAULT_SAMPLE_EVERY)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    dev = setup_device()
    model = load_pgans(base_lr=args.lr, useGPU=dev["useGPU"])

    if not DATA_DIR.exists():
        log(f"ERROR: data dir not found: {DATA_DIR}. 先运行 prepare_dataset.py")
        return 2
    dataset = FaceDataset(DATA_DIR)
    log(f"dataset: {len(dataset)} images from {DATA_DIR}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    fixed_z = get_or_create_fixed_z(model, n=DEFAULT_NUM_FIXED)

    # loss 日志
    loss_csv = OUTPUT_DIR / "loss_log.csv"
    loss_csv.parent.mkdir(parents=True, exist_ok=True)
    loss_fields = ["step", "lossD_real", "lossD_fake", "lossG"]
    loss_writer = csv.DictWriter(open(loss_csv, "w", newline="", encoding="utf-8"), fieldnames=loss_fields)
    loss_writer.writeheader()

    # 微调前基线 (step 0)
    save_samples(model, fixed_z, step=0)
    save_checkpoint(model, OUTPUT_DIR / "step_0000" / "progan_finetuned.pth")

    # 恢复训练模式
    model.netG.train()
    model.netD.train()

    step = 0
    log(f"start finetune: total_steps={args.total_steps} batch_size={args.batch_size} lr={args.lr}")
    while step < args.total_steps:
        for real_images in loader:
            losses = model.optimizeParameters(real_images)

            step += 1

            loss_writer.writerow({
                "step": step,
                "lossD_real": float(losses.get("lossD_real", float("nan"))),
                "lossD_fake": float(losses.get("lossD_fake", float("nan"))),
                "lossG": float(losses.get("lossG", float("nan"))),
            })

            if step % 20 == 0:
                log(f"step={step}/{args.total_steps}  {losses}")

            if step % args.sample_every == 0 or step % args.save_every == 0:
                save_samples(model, fixed_z, step)
                model.netG.train()
                model.netD.train()

            if step % args.save_every == 0:
                save_checkpoint(model, OUTPUT_DIR / f"step_{step:04d}" / "progan_finetuned.pth")

            if step >= args.total_steps:
                break

    log("finetune done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

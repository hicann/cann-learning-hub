#!/usr/bin/env python3
"""Teaching-oriented YOLO single-NPU + optional DDP AMP training script for Ascend NPU.

This script is intentionally compact. It demonstrates the distributed training
control path used by a YOLO detector while defaulting to world_size=1:
PASCAL VOC split lists, AMP, warmup learning rate, checkpointing, and basic
throughput logging. HCCL/DDP hooks are kept as an extension path for future multi-card
resources. The model and loss are simplified so learners can focus on the
system optimization workflow.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# VOC mirrors may contain recoverable truncated JPEGs; keep the DataLoader alive.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/yolo_ascend.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--profile", action="store_true", help="shorten the run for external MSPROF capture")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return not is_dist() or dist.get_rank() == 0


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(local_rank: int) -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def init_distributed(cfg: dict, device: torch.device) -> tuple[int, int, int]:
    local_rank = env_int("LOCAL_RANK", 0)
    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return rank, local_rank, world_size

    if device.type == "npu":
        backend = cfg["distributed"].get("backend", "hccl")
    elif device.type == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=cfg["distributed"].get("init_method", "env://"),
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


class YoloTxtDataset(Dataset):
    """Read images and YOLO txt labels.

    Labels are expected as: class_id x_center y_center width height
    with normalized coordinates in [0, 1].
    """

    def __init__(
        self,
        root: Path,
        image_subdir: str,
        label_subdir: str,
        image_size: int,
        num_classes: int,
        train: bool,
        image_list: str = "",
    ) -> None:
        self.root = root
        self.image_dir = root / image_subdir
        self.label_dir = root / label_subdir
        self.image_size = image_size
        self.num_classes = num_classes
        self.train = train
        self.images = self._load_images(image_list)
        self._bad_image_warnings: set[Path] = set()
        if not self.images:
            raise FileNotFoundError(f"no images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _load_images(self, image_list: str) -> list[Path]:
        if image_list:
            list_path = Path(image_list)
            if not list_path.is_absolute():
                list_path = self.root / image_list
            if not list_path.exists():
                raise FileNotFoundError(f"image list not found: {list_path}")
            images = []
            for line in list_path.read_text(encoding="utf-8").splitlines():
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = self.root / path
                images.append(path)
            return sorted(path for path in images if path.suffix.lower() in IMAGE_SUFFIXES)
        return sorted(path for path in self.image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)

    def _label_path_for_image(self, image_path: Path) -> Path:
        flat_path = self.label_dir / f"{image_path.stem}.txt"
        if flat_path.exists():
            return flat_path

        try:
            image_rel = image_path.relative_to(self.root)
        except ValueError:
            return flat_path

        parts = image_rel.parts
        if parts and parts[0] == "images":
            mirrored = self.root / "labels" / Path(*parts[1:]).with_suffix(".txt")
            if mirrored.exists():
                return mirrored
        return flat_path

    def _read_labels(self, image_path: Path) -> torch.Tensor:
        label_path = self._label_path_for_image(image_path)
        rows = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = map(float, parts)
                if 0 <= cls < self.num_classes and w > 0 and h > 0:
                    rows.append([cls, x, y, w, h])
        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32)
        target = torch.tensor(rows, dtype=torch.float32)
        target[:, 1:] = target[:, 1:].clamp(0.0, 1.0)
        return target

    def _read_image(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def _warn_bad_image(self, image_path: Path, exc: Exception) -> None:
        if image_path in self._bad_image_warnings:
            return
        self._bad_image_warnings.add(image_path)
        if rank0():
            print(f"[WARN] skip unreadable image: {image_path} ({exc})", flush=True)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        last_error: Exception | None = None
        for offset in range(len(self.images)):
            image_path = self.images[(index + offset) % len(self.images)]
            try:
                image = self._read_image(image_path)
                target = self._read_labels(image_path)
                break
            except (OSError, UnidentifiedImageError) as exc:
                last_error = exc
                self._warn_bad_image(image_path, exc)
        else:
            raise RuntimeError("all images failed to load") from last_error

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if target.numel() > 0:
                target[:, 1] = 1.0 - target[:, 1]

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, target


class SyntheticYoloDataset(Dataset):
    """Small random dataset used when no real images are available."""

    def __init__(self, length: int, image_size: int, num_classes: int) -> None:
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        image = torch.rand((3, self.image_size, self.image_size), generator=generator)
        boxes = []
        for _ in range(int(torch.randint(1, 4, (1,), generator=generator).item())):
            cls = int(torch.randint(0, self.num_classes, (1,), generator=generator).item())
            x = float(torch.rand((), generator=generator) * 0.8 + 0.1)
            y = float(torch.rand((), generator=generator) * 0.8 + 0.1)
            w = float(torch.rand((), generator=generator) * 0.25 + 0.05)
            h = float(torch.rand((), generator=generator) * 0.25 + 0.05)
            boxes.append([cls, x, y, w, h])
        return image, torch.tensor(boxes, dtype=torch.float32)


def collate_yolo(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


def make_dataset(cfg: dict, split: str) -> Dataset:
    data_cfg = cfg["data"]
    train = split == "train"
    root = Path(data_cfg["root"])
    image_key = "train_images" if train else "val_images"
    label_key = "train_labels" if train else "val_labels"
    list_key = "train_list" if train else "val_list"
    try:
        return YoloTxtDataset(
            root=root,
            image_subdir=data_cfg[image_key],
            label_subdir=data_cfg[label_key],
            image_size=int(data_cfg["image_size"]),
            num_classes=int(cfg["model"]["num_classes"]),
            train=train,
            image_list=data_cfg.get(list_key, ""),
        )
    except FileNotFoundError as exc:
        if train and cfg["train"].get("synthetic_when_missing", False):
            if rank0():
                print(f"[warning] {exc}; using synthetic data for a runnable demo")
            return SyntheticYoloDataset(256, int(data_cfg["image_size"]), int(cfg["model"]["num_classes"]))
        if not train and cfg["train"].get("synthetic_when_missing", False):
            return SyntheticYoloDataset(64, int(data_cfg["image_size"]), int(cfg["model"]["num_classes"]))
        raise


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyYolo(nn.Module):
    def __init__(self, num_classes: int, anchors: list[list[int]]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.backbone = nn.Sequential(
            ConvBNAct(3, 32, 2),
            ConvBNAct(32, 64, 2),
            ConvBNAct(64, 128, 2),
            ConvBNAct(128, 256, 2),
            ConvBNAct(256, 384, 2),
        )
        self.head = nn.Conv2d(384, self.num_anchors * (5 + num_classes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        b, _, h, w = x.shape
        x = x.view(b, self.num_anchors, 5 + self.num_classes, h, w)
        return x.permute(0, 1, 3, 4, 2).contiguous()


def build_targets(
    targets: list[torch.Tensor],
    anchors_px: torch.Tensor,
    image_size: int,
    grid_size: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(targets)
    num_anchors = anchors_px.shape[0]
    obj = torch.zeros((batch_size, num_anchors, grid_size, grid_size), device=device)
    box = torch.zeros((batch_size, num_anchors, grid_size, grid_size, 4), device=device)
    cls = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.long, device=device)
    anchors = anchors_px.to(device=device, dtype=torch.float32) / float(image_size)

    for batch_idx, target in enumerate(targets):
        if target.numel() == 0:
            continue
        target = target.to(device)
        for item in target:
            class_id = int(item[0].clamp(0, num_classes - 1).item())
            x, y, w, h = item[1:].clamp(1e-4, 1.0)
            gx = x * grid_size
            gy = y * grid_size
            gi = int(torch.clamp(gx.floor(), 0, grid_size - 1).item())
            gj = int(torch.clamp(gy.floor(), 0, grid_size - 1).item())

            wh = torch.stack([w, h])
            ratio = torch.min(wh[None, :] / anchors, anchors / wh[None, :]).prod(dim=1)
            anchor_idx = int(ratio.argmax().item())

            obj[batch_idx, anchor_idx, gj, gi] = 1.0
            box[batch_idx, anchor_idx, gj, gi, 0] = gx - gi
            box[batch_idx, anchor_idx, gj, gi, 1] = gy - gj
            box[batch_idx, anchor_idx, gj, gi, 2] = torch.log(w / anchors[anchor_idx, 0]).clamp(-4.0, 4.0)
            box[batch_idx, anchor_idx, gj, gi, 3] = torch.log(h / anchors[anchor_idx, 1]).clamp(-4.0, 4.0)
            cls[batch_idx, anchor_idx, gj, gi] = class_id

    return obj, box, cls


def yolo_loss(
    pred: torch.Tensor,
    targets: list[torch.Tensor],
    anchors_px: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_size = int(cfg["data"]["image_size"])
    grid_size = pred.shape[2]
    num_classes = int(cfg["model"]["num_classes"])
    target_obj, target_box, target_cls = build_targets(
        targets, anchors_px, image_size, grid_size, num_classes, device
    )

    pred_xy = pred[..., 0:2].sigmoid()
    pred_wh = pred[..., 2:4]
    pred_obj = pred[..., 4]
    pred_cls = pred[..., 5:]
    pos = target_obj.bool()

    loss_obj = F.binary_cross_entropy_with_logits(pred_obj, target_obj)
    if pos.any():
        loss_box = F.mse_loss(torch.cat([pred_xy[pos], pred_wh[pos]], dim=-1), target_box[pos])
        loss_cls = F.cross_entropy(pred_cls[pos], target_cls[pos])
    else:
        loss_box = pred.sum() * 0.0
        loss_cls = pred.sum() * 0.0

    loss = 5.0 * loss_box + loss_obj + loss_cls
    metrics = {
        "loss_box": float(loss_box.detach().item()),
        "loss_obj": float(loss_obj.detach().item()),
        "loss_cls": float(loss_cls.detach().item()),
        "positives": float(pos.sum().detach().item()),
    }
    return loss, metrics


def make_amp(device: torch.device, enabled: bool) -> tuple[Callable[[], object], object | None]:
    if not enabled or device.type == "cpu":
        return nullcontext, None
    if device.type == "npu":
        import torch_npu

        return torch_npu.npu.amp.autocast, torch_npu.npu.amp.GradScaler()
    return torch.cuda.amp.autocast, torch.cuda.amp.GradScaler()


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def scheduled_lr(cfg: dict, epoch: int, step: int, steps_per_epoch: int) -> float:
    train_cfg = cfg["train"]
    base_lr = float(train_cfg["learning_rate"])
    min_lr = float(train_cfg["min_learning_rate"])
    warmup_epochs = int(train_cfg["warmup_epochs"])
    total_epochs = int(train_cfg["epochs"])
    global_step = epoch * steps_per_epoch + step + 1
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    if global_step <= warmup_steps:
        return base_lr * global_step / warmup_steps
    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def reduce_value(value: float, device: torch.device) -> float:
    if not is_dist():
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def sync_device(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pt")
    torch.save(ckpt, output_dir / "last.pt")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str, device: torch.device) -> int:
    if not path:
        return 0
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    anchors: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        pred = model(images)
        loss, _ = yolo_loss(pred, targets, anchors, cfg, device)
        total += float(loss.item())
        count += 1
    model.train()
    return reduce_value(total / max(1, count), device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["train"]["workers"] = args.workers
    if args.profile:
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 2)
        cfg["train"]["log_interval"] = 1

    local_rank = env_int("LOCAL_RANK", 0)
    device = select_device(local_rank)
    rank, local_rank, world_size = init_distributed(cfg, device)
    seed_everything(2026, rank)

    train_dataset = make_dataset(cfg, "train")
    val_dataset = make_dataset(cfg, "val")
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["train"]["workers"]),
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_yolo,
        persistent_workers=int(cfg["train"]["workers"]) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(0, int(cfg["train"]["workers"]) // 2),
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_yolo,
    )

    model = TinyYolo(
        num_classes=int(cfg["model"]["num_classes"]),
        anchors=cfg["model"]["anchors"],
    ).to(device)
    if world_size > 1 and device.type != "cpu" and cfg["distributed"].get("sync_batchnorm", True):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        momentum=float(cfg["train"]["momentum"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        nesterov=True,
    )
    start_epoch = load_checkpoint(model, optimizer, cfg["train"].get("resume", ""), device)

    if world_size > 1:
        ddp_kwargs = {
            "find_unused_parameters": bool(cfg["distributed"].get("find_unused_parameters", False)),
        }
        if device.type != "cpu":
            ddp_kwargs["device_ids"] = [local_rank]
        model = DDP(model, **ddp_kwargs)

    autocast, scaler = make_amp(device, bool(cfg["train"]["amp"]))
    anchors = torch.tensor(cfg["model"]["anchors"], dtype=torch.float32, device=device)
    output_dir = Path(cfg["project"]["output_dir"])
    grad_accum = max(1, int(cfg["train"]["gradient_accumulation"]))
    log_interval = max(1, int(cfg["train"]["log_interval"]))

    if rank0():
        print("=" * 80)
        title = "YOLO single-NPU AMP training on Ascend" if world_size == 1 else "YOLO DDP AMP training on Ascend"
        print(title)
        print(f"device={device}, world_size={world_size}, batch_per_rank={cfg['train']['batch_size']}")
        print(f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}")
        print(f"amp={cfg['train']['amp']}, warmup_epochs={cfg['train']['warmup_epochs']}")
        print("=" * 80)

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (images, targets) in enumerate(train_loader):
            lr = scheduled_lr(cfg, epoch, step, len(train_loader))
            set_lr(optimizer, lr)
            images = images.to(device, non_blocking=True)

            with autocast():
                pred = model(images)
                loss, metrics = yolo_loss(pred, targets, anchors, cfg, device)
                loss_to_backward = loss / grad_accum

            if scaler is not None:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().item())

            if (step + 1) % log_interval == 0:
                sync_device(device)
                elapsed = time.time() - step_start
                images_per_sec = (
                    log_interval * int(cfg["train"]["batch_size"]) * world_size / max(elapsed, 1e-6)
                )
                mean_loss = reduce_value(running_loss / log_interval, device)
                if rank0():
                    print(
                        f"epoch={epoch + 1:03d} step={step + 1:04d}/{len(train_loader)} "
                        f"loss={mean_loss:.4f} lr={lr:.6f} "
                        f"box={metrics['loss_box']:.3f} obj={metrics['loss_obj']:.3f} "
                        f"cls={metrics['loss_cls']:.3f} pos={metrics['positives']:.0f} "
                        f"throughput={images_per_sec:.1f} img/s"
                    )
                running_loss = 0.0
                step_start = time.time()

            if args.profile and step >= int(cfg["profile"].get("warmup_steps", 10)) + int(
                cfg["profile"].get("active_steps", 20)
            ):
                break

        val_loss = validate(model, val_loader, anchors, cfg, device)
        if rank0():
            print(f"epoch={epoch + 1:03d} validation_loss={val_loss:.4f}")
            if (epoch + 1) % int(cfg["train"]["save_interval"]) == 0 or epoch + 1 == int(cfg["train"]["epochs"]):
                save_checkpoint(model, optimizer, epoch + 1, cfg, output_dir)
                print(f"checkpoint saved to {output_dir.resolve()}")

    cleanup_distributed()


if __name__ == "__main__":
    main()

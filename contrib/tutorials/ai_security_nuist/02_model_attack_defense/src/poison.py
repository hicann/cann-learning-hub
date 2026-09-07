"""BadNets 投毒相关函数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.utils import ensure_dir


SUPPORTED_TRIGGER_TYPES = ("square", "checkerboard")


def _to_hwc(image: np.ndarray) -> tuple[np.ndarray, str]:
    """把图像统一转成 HWC，便于局部区域修改。"""

    if image.ndim != 3:
        raise ValueError("图像数组必须是 3 维。")

    if image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0)).astype(np.float32), "CHW"

    return image.astype(np.float32).copy(), "HWC"


def _restore_layout(image_hwc: np.ndarray, layout: str) -> np.ndarray:
    """把处理后的 HWC 图像还原回原始布局。"""

    if layout == "CHW":
        return np.transpose(image_hwc, (2, 0, 1)).astype(np.float32)
    return image_hwc.astype(np.float32)


def _resolve_trigger_region(height: int, width: int, trigger_size: int, position: str) -> tuple[int, int, int]:
    """计算触发器左上角坐标和实际边长。"""

    size = int(max(1, min(trigger_size, height, width)))

    if position == "bottom_right":
        top = height - size
        left = width - size
    elif position == "top_left":
        top = 0
        left = 0
    else:
        raise ValueError("当前仅支持 bottom_right 或 top_left。")

    return top, left, size


def add_square_trigger(
    image: np.ndarray,
    trigger_size: int = 4,
    alpha: float = 1.0,
    position: str = "bottom_right",
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """给单张图像添加纯色方形触发器，作为经典 BadNets baseline。"""

    image_hwc, layout = _to_hwc(image)
    poisoned = image_hwc.copy()

    height, width, _ = poisoned.shape
    top, left, size = _resolve_trigger_region(height, width, trigger_size, position)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    color_patch = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)

    region = poisoned[top : top + size, left : left + size, :]
    poisoned[top : top + size, left : left + size, :] = (1.0 - alpha) * region + alpha * color_patch

    poisoned = np.clip(poisoned, 0.0, 1.0)
    return _restore_layout(poisoned, layout)


def add_checkerboard_trigger(
    image: np.ndarray,
    trigger_size: int = 4,
    alpha: float = 1.0,
    position: str = "bottom_right",
    color_a: tuple[float, float, float] = (1.0, 1.0, 1.0),
    color_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """给单张图像添加棋盘格 patch 触发器。"""

    image_hwc, layout = _to_hwc(image)
    poisoned = image_hwc.copy()

    height, width, _ = poisoned.shape
    top, left, size = _resolve_trigger_region(height, width, trigger_size, position)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    rows, cols = np.indices((size, size))
    mask = ((rows + cols) % 2 == 0)[..., None]
    patch_a = np.asarray(color_a, dtype=np.float32).reshape(1, 1, 3)
    patch_b = np.asarray(color_b, dtype=np.float32).reshape(1, 1, 3)
    checkerboard = np.where(mask, patch_a, patch_b).astype(np.float32)

    region = poisoned[top : top + size, left : left + size, :]
    poisoned[top : top + size, left : left + size, :] = (1.0 - alpha) * region + alpha * checkerboard

    poisoned = np.clip(poisoned, 0.0, 1.0)
    return _restore_layout(poisoned, layout)


def add_trigger(
    image: np.ndarray,
    trigger_size: int = 4,
    alpha: float = 1.0,
    position: str = "bottom_right",
    trigger_type: str = "square",
) -> np.ndarray:
    """根据 trigger_type 添加对应触发器。"""

    normalized_type = trigger_type.lower().strip()
    if normalized_type == "square":
        return add_square_trigger(image=image, trigger_size=trigger_size, alpha=alpha, position=position)
    if normalized_type == "checkerboard":
        return add_checkerboard_trigger(image=image, trigger_size=trigger_size, alpha=alpha, position=position)

    supported = ", ".join(SUPPORTED_TRIGGER_TYPES)
    raise ValueError(f"未知 trigger_type: {trigger_type}，支持的类型：{supported}")


def poison_dataset_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    poison_ratio: float,
    target_label: int,
    trigger_size: int,
    alpha: float,
    trigger_type: str = "square",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对一批样本执行 BadNets 投毒。"""

    if len(images) != len(labels):
        raise ValueError("images 和 labels 的长度必须一致。")

    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError("poison_ratio 必须位于 [0, 1] 区间。")

    rng = np.random.default_rng(seed)
    poisoned_images = images.astype(np.float32).copy()
    poisoned_labels = labels.astype(np.int32).copy()

    candidate_indices = np.where(labels != target_label)[0]
    poison_count = int(len(candidate_indices) * poison_ratio)

    if poison_ratio > 0 and poison_count == 0 and len(candidate_indices) > 0:
        poison_count = 1

    if poison_count > 0:
        selected_indices = rng.choice(candidate_indices, size=poison_count, replace=False)
    else:
        selected_indices = np.asarray([], dtype=np.int64)

    poison_mask = np.zeros(len(labels), dtype=bool)
    poison_mask[selected_indices] = True

    for index in selected_indices:
        poisoned_images[index] = add_trigger(
            image=poisoned_images[index],
            trigger_size=trigger_size,
            alpha=alpha,
            trigger_type=trigger_type,
        )
        poisoned_labels[index] = np.int32(target_label)

    return poisoned_images, poisoned_labels, poison_mask


def save_poison_preview(clean_image: np.ndarray, poisoned_image: np.ndarray, save_path: Path) -> Path:
    """保存投毒前后的对比图，便于报告插图。"""

    clean_hwc, _ = _to_hwc(clean_image)
    poisoned_hwc, _ = _to_hwc(poisoned_image)
    canvas = np.concatenate([clean_hwc, poisoned_hwc], axis=1)
    canvas = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)

    ensure_dir(save_path.parent)
    Image.fromarray(canvas).save(save_path)
    return save_path

from __future__ import annotations

import random
import shutil
import os
import stat
from pathlib import Path
from typing import Any

from .paths import ensure_dir, save_json


IMAGE_SUFFIXES = {".ppm", ".png", ".jpg", ".jpeg", ".bmp"}


def _list_class_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{split_dir}")
    class_dirs = [path for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not class_dirs:
        raise ValueError(f"没有找到形如 00000/*.ppm 的类别目录：{split_dir}")
    return sorted(class_dirs, key=lambda path: int(path.name))


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return

    for current in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            os.chmod(current, stat.S_IWRITE)
        except Exception:
            continue
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass

    def handle_remove_error(function, current_path, exc_info):  # noqa: ANN001
        try:
            os.chmod(current_path, stat.S_IWRITE)
            function(current_path)
        except Exception:
            raise exc_info[1]

    shutil.rmtree(path, onerror=handle_remove_error)


def _copy_split(source_dir: Path, target_dir: Path, per_class: int, rng: random.Random) -> dict[str, Any]:
    if target_dir.exists():
        _remove_tree(target_dir)
    ensure_dir(target_dir)

    class_counts: dict[str, int] = {}
    selected_files: dict[str, list[str]] = {}
    total = 0

    for class_dir in _list_class_dirs(source_dir):
        images = _list_images(class_dir)
        shuffled = list(images)
        rng.shuffle(shuffled)
        selected = shuffled[: min(int(per_class), len(shuffled))]
        class_target_dir = ensure_dir(target_dir / class_dir.name)
        for image_path in selected:
            shutil.copy2(image_path, class_target_dir / image_path.name)
        class_counts[class_dir.name] = len(selected)
        selected_files[class_dir.name] = [str(path.relative_to(source_dir)) for path in selected]
        total += len(selected)

    return {
        "class_counts": class_counts,
        "selected_files": selected_files,
        "total": total,
    }


def create_demo_subset(
    train_dir: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> dict[str, Any]:
    """Create a deterministic ImageFolder-style demo subset under onsite outputs."""

    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    if train_per_class <= 0 or test_per_class <= 0:
        raise ValueError("train_per_class 和 test_per_class 必须大于 0")

    ensure_dir(output_dir)
    rng = random.Random(int(seed))
    train_info = _copy_split(train_dir, output_dir / "train", int(train_per_class), rng)
    test_info = _copy_split(test_dir, output_dir / "test", int(test_per_class), rng)

    manifest = {
        "seed": int(seed),
        "train_per_class": int(train_per_class),
        "test_per_class": int(test_per_class),
        "source_paths": {
            "train": str(train_dir),
            "test": str(test_dir),
        },
        "target_paths": {
            "train": str(output_dir / "train"),
            "test": str(output_dir / "test"),
        },
        "train_class_counts": train_info["class_counts"],
        "test_class_counts": test_info["class_counts"],
        "train_total": int(train_info["total"]),
        "test_total": int(test_info["total"]),
        "selected_train_files": train_info["selected_files"],
        "selected_test_files": test_info["selected_files"],
    }
    save_json(manifest, output_dir / "subset_manifest.json")
    return manifest


def list_image_records(split_dir: str | Path) -> list[tuple[Path, int]]:
    split_dir = Path(split_dir)
    records: list[tuple[Path, int]] = []
    for class_dir in _list_class_dirs(split_dir):
        label = int(class_dir.name)
        for image_path in _list_images(class_dir):
            records.append((image_path, label))
    return records


def count_records(split_dir: str | Path) -> int:
    return len(list_image_records(split_dir))


def pick_first_non_target_image(split_dir: str | Path, target_label: int = 0) -> Path:
    for image_path, label in list_image_records(split_dir):
        if int(label) != int(target_label):
            return image_path
    raise ValueError(f"测试子集中没有非 target label={target_label} 的样本，无法展示触发器攻击图。")

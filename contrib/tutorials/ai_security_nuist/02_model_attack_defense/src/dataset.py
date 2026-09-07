"""使用 MindSpore ImageFolderDataset 读取整理后的 GTSRB 数据集。

本文件依赖 `scripts/prepare_gtsrb.py` 先把数据整理成下面的目录结构：

src/data/train/00000/*.ppm
src/data/train/00001/*.ppm
...
src/data/test/00000/*.ppm
src/data/test/00001/*.ppm
...

只要目录已经整理成“每个类别一个子目录”的形式，
MindSpore 的 ImageFolderDataset 就可以自动把子目录名当作类别读取。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    import mindspore as ms
    import mindspore.dataset as ds
    import mindspore.dataset.transforms as transforms
    import mindspore.dataset.vision as vision

    MINDSPORE_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - 是否安装 MindSpore 取决于运行环境
    ms = None
    ds = None
    transforms = None
    vision = None
    MINDSPORE_IMPORT_ERROR = exc


IMAGE_SUFFIXES = {".ppm", ".png", ".jpg", ".jpeg"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_IMAGE_SIZE = (48, 48)

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover
    RESAMPLE_BILINEAR = Image.BILINEAR


def ensure_mindspore_available() -> None:
    """在真正创建数据集之前，先给出更友好的依赖缺失提示。"""

    if MINDSPORE_IMPORT_ERROR is not None:
        raise ImportError(
            "当前 Python 环境未安装 MindSpore，无法创建 ImageFolderDataset。"
            "请先安装与本机或云端环境匹配的 mindspore。"
        ) from MINDSPORE_IMPORT_ERROR


def normalize_image_size(image_size: Union[int, tuple[int, int], list[int]]) -> tuple[int, int]:
    """把图像尺寸统一整理成 (height, width) 二元组。"""

    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size 必须是正整数")
        return (image_size, image_size)

    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        height, width = int(image_size[0]), int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size 中的高和宽都必须是正整数")
        return (height, width)

    raise ValueError("image_size 必须是 int，或长度为 2 的 tuple/list")


def resolve_num_parallel_workers(
    num_parallel_workers: Optional[int] = None,
    server_fast: bool = False,
) -> int:
    """根据运行场景给出较稳妥的数据并行 worker 数。"""

    if num_parallel_workers is not None and int(num_parallel_workers) > 0:
        return int(num_parallel_workers)

    cpu_count = os.cpu_count() or 4
    if server_fast:
        return max(4, min(16, cpu_count // 2 if cpu_count >= 8 else cpu_count))
    return max(1, min(4, cpu_count))


def configure_dataset_runtime(prefetch_size: Optional[int] = None) -> None:
    """配置 MindSpore 数据集运行时参数。"""

    ensure_mindspore_available()

    if prefetch_size is not None and int(prefetch_size) > 0:
        ds.config.set_prefetch_size(int(prefetch_size))


def list_prepared_class_dirs(split_dir: Path) -> list[Path]:
    """列出整理后数据目录中的类别子目录。"""

    if not split_dir.exists() or not split_dir.is_dir():
        return []

    class_dirs = [path for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(class_dirs, key=lambda path: int(path.name))


def count_images_in_class_dir(class_dir: Path) -> int:
    """统计单个类别目录中的图片数量。"""

    count = 0
    for image_path in class_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            count += 1
    return count


def validate_prepared_split(split_dir: Path) -> tuple[list[Path], int]:
    """检查整理后的数据目录是否适合 ImageFolderDataset 读取。"""

    if not split_dir.exists():
        raise FileNotFoundError(
            f"数据目录不存在：{split_dir}\n"
            "请先运行 python scripts\\prepare_gtsrb.py 整理 GTSRB 官方数据集。"
        )

    if not split_dir.is_dir():
        raise NotADirectoryError(f"给定路径不是目录：{split_dir}")

    class_dirs = list_prepared_class_dirs(split_dir)
    if not class_dirs:
        raise ValueError(
            f"目录中没有找到类别子目录：{split_dir}\n"
            "期望结构应类似 src/data/train/00000/*.ppm 或 src/data/test/00000/*.ppm。"
        )

    invalid_class_dirs: list[str] = []
    for child in split_dir.iterdir():
        if not child.is_dir() or child.name.isdigit():
            continue
        if count_images_in_class_dir(child) > 0:
            invalid_class_dirs.append(child.name)

    if invalid_class_dirs:
        invalid_names = ", ".join(sorted(invalid_class_dirs))
        raise ValueError(
            f"检测到非数字类别目录：{invalid_names}\n"
            "请确认 src/data/train 和 src/data/test 中只保留 00000 这种类别文件夹。"
        )

    total_image_count = sum(count_images_in_class_dir(class_dir) for class_dir in class_dirs)
    if total_image_count == 0:
        raise ValueError(
            f"目录存在类别子文件夹，但里面没有图片：{split_dir}\n"
            "请先检查 prepare_gtsrb.py 是否已成功整理数据。"
        )

    return class_dirs, total_image_count


def resolve_split_dir(data_dir: Union[str, Path], split_name: str) -> Path:
    """兼容传入 data 根目录或直接传入 data/train、data/test 两种写法。"""

    data_dir = Path(data_dir)
    if data_dir.name.lower() == split_name.lower():
        return data_dir
    return data_dir / split_name


def build_train_image_transforms(
    image_size: tuple[int, int],
    light_augment: bool = False,
) -> list[object]:
    """构建训练集图像预处理与轻量增强流水线。

    这里的增强尽量保持保守：
    1. 小角度旋转；
    2. 轻微平移和缩放；
    3. 轻微亮度/对比度扰动。
    """

    ensure_mindspore_available()

    if light_augment:
        return [
            vision.Decode(),
            vision.Resize(image_size, interpolation=vision.Inter.BILINEAR),
            vision.RandomRotation(degrees=8, resample=vision.Inter.BILINEAR, expand=False, fill_value=0),
            vision.RandomColorAdjust(brightness=(0.95, 1.05), contrast=(0.95, 1.05)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=True),
            vision.HWC2CHW(),
        ]

    return [
        vision.Decode(),
        vision.Resize(image_size, interpolation=vision.Inter.BILINEAR),
        vision.RandomRotation(degrees=10, resample=vision.Inter.BILINEAR, expand=False, fill_value=0),
        vision.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.95, 1.05),
            resample=vision.Inter.BILINEAR,
            fill_value=0,
        ),
        vision.RandomColorAdjust(brightness=(0.9, 1.1), contrast=(0.9, 1.1)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=True),
        vision.HWC2CHW(),
    ]


def build_eval_image_transforms(image_size: tuple[int, int]) -> list[object]:
    """构建验证/测试图像预处理流水线。"""

    ensure_mindspore_available()

    return [
        vision.Decode(),
        vision.Resize(image_size, interpolation=vision.Inter.BILINEAR),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=True),
        vision.HWC2CHW(),
    ]


def create_imagefolder_dataset(
    split_dir: Union[str, Path],
    image_size: Union[int, tuple[int, int]] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    shuffle: bool = True,
    num_samples: Optional[int] = None,
    drop_remainder: bool = False,
    is_training: bool = False,
    num_parallel_workers: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    server_fast: bool = False,
):
    """创建通用的 ImageFolderDataset。"""

    ensure_mindspore_available()

    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    split_dir = Path(split_dir)
    normalize_size = normalize_image_size(image_size)
    validate_prepared_split(split_dir)
    configure_dataset_runtime(prefetch_size=prefetch_size)
    resolved_workers = resolve_num_parallel_workers(
        num_parallel_workers=num_parallel_workers,
        server_fast=server_fast,
    )

    dataset = ds.ImageFolderDataset(
        dataset_dir=str(split_dir),
        shuffle=shuffle,
        num_samples=num_samples,
        num_parallel_workers=resolved_workers,
    )
    dataset = dataset.map(
        operations=build_train_image_transforms(
            normalize_size,
            light_augment=server_fast and is_training,
        )
        if is_training
        else build_eval_image_transforms(normalize_size),
        input_columns=["image"],
        num_parallel_workers=resolved_workers,
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(ms.int32),
        input_columns=["label"],
        num_parallel_workers=resolved_workers,
    )
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


def create_train_dataset(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    image_size: Union[int, tuple[int, int]] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    shuffle: bool = True,
    num_samples: Optional[int] = None,
    augment: bool = True,
    drop_remainder: bool = False,
    num_parallel_workers: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    server_fast: bool = False,
):
    """创建训练集数据管道。"""

    train_dir = resolve_split_dir(data_dir, "train")
    return create_imagefolder_dataset(
        split_dir=train_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        num_samples=num_samples,
        drop_remainder=drop_remainder,
        is_training=augment,
        num_parallel_workers=num_parallel_workers,
        prefetch_size=prefetch_size,
        server_fast=server_fast,
    )


def create_test_dataset(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    image_size: Union[int, tuple[int, int]] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    num_samples: Optional[int] = None,
    drop_remainder: bool = False,
    num_parallel_workers: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    server_fast: bool = False,
):
    """创建测试集数据管道。"""

    test_dir = resolve_split_dir(data_dir, "test")
    return create_imagefolder_dataset(
        split_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        num_samples=num_samples,
        drop_remainder=drop_remainder,
        is_training=False,
        num_parallel_workers=num_parallel_workers,
        prefetch_size=prefetch_size,
        server_fast=server_fast,
    )


def scan_class_folders(data_dir: Union[str, Path]) -> list[tuple[Path, int]]:
    """扫描类别目录，返回 (图片路径, 标签) 列表。"""

    split_dir = Path(data_dir)
    class_dirs = list_prepared_class_dirs(split_dir)

    records: list[tuple[Path, int]] = []
    for class_dir in class_dirs:
        label = int(class_dir.name)
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append((image_path, label))

    return records


def load_single_image(image_path: Path, image_size: Union[int, tuple[int, int]] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """读取单张图像，并转换成模型可直接使用的 CHW 浮点数组。"""

    height, width = normalize_image_size(image_size)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), RESAMPLE_BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def load_folder_samples(
    data_dir: Union[str, Path],
    image_size: Union[int, tuple[int, int]] = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """把整理好的类别目录读成 NumPy 数组。"""

    split_dir = Path(data_dir)
    records = scan_class_folders(split_dir)
    height, width = normalize_image_size(image_size)

    if not records:
        empty_images = np.empty((0, 3, height, width), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.int32)
        return empty_images, empty_labels

    images = [load_single_image(path, image_size=(height, width)) for path, _ in records]
    labels = [label for _, label in records]

    return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int32)


def create_numpy_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_remainder: bool = False,
):
    """把 NumPy 数组封装成 MindSpore 数据集。"""

    ensure_mindspore_available()

    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("输入数组为空，无法创建 MindSpore 数据集")

    dataset = ds.NumpySlicesDataset(
        {"image": images.astype(np.float32), "label": labels.astype(np.int32)},
        shuffle=shuffle,
    )
    return dataset.batch(batch_size, drop_remainder=drop_remainder)


def create_image_dataset(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    image_size: Union[int, tuple[int, int]] = DEFAULT_IMAGE_SIZE,
    drop_remainder: bool = False,
    is_training: bool = False,
    num_parallel_workers: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    server_fast: bool = False,
):
    """兼容旧代码的通用入口。"""

    return create_imagefolder_dataset(
        split_dir=Path(data_dir),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        num_samples=None,
        drop_remainder=drop_remainder,
        is_training=is_training,
        num_parallel_workers=num_parallel_workers,
        prefetch_size=prefetch_size,
        server_fast=server_fast,
    )


def debug_main() -> None:
    """调试入口：直接检查训练集/测试集是否能被正确读取。"""

    print("=" * 72)
    print("GTSRB ImageFolderDataset 调试入口")
    print("=" * 72)
    print("本调试脚本依赖数据已整理到 src/data/train 和 src/data/test")

    train_dataset = create_train_dataset(
        data_dir=DEFAULT_DATA_DIR,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=8,
        shuffle=False,
    )
    test_dataset = create_test_dataset(
        data_dir=DEFAULT_DATA_DIR,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=8,
        shuffle=False,
    )

    print(f"训练集 dataset size（按 batch 计）: {train_dataset.get_dataset_size()}")
    print(f"测试集 dataset size（按 batch 计）: {test_dataset.get_dataset_size()}")

    first_batch = next(train_dataset.create_dict_iterator(output_numpy=True))
    print(f"一个 batch 的 image shape: {first_batch['image'].shape}")
    print(f"一个 batch 的 label shape: {first_batch['label'].shape}")


if __name__ == "__main__":
    try:
        debug_main()
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] dataset.py 调试失败：{exc}")
        raise

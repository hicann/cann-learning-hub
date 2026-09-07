from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Union

import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_image_dataset, create_test_dataset, create_train_dataset, scan_class_folders
from src.poison import SUPPORTED_TRIGGER_TYPES, add_trigger, save_poison_preview
from src.utils import ensure_dir, save_json, set_seed, timestamp
import src.model as model_module


def str2bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数只接受 true/false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTSRB clean / badnet 训练脚本")
    parser.add_argument("--task", default="clean", choices=["clean", "badnet"])
    parser.add_argument("--ms-mode", default="PYNATIVE", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "src" / "data" / "train")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "src" / "data" / "test")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "src" / "outputs" / "checkpoints" / "train_run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=43)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--device-target", default="CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-fast", type=str2bool, default=False)
    parser.add_argument("--num-parallel-workers", type=int, default=0)
    parser.add_argument("--prefetch-size", type=int, default=0)
    parser.add_argument("--drop-remainder", type=str2bool, default=False)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-train-acc", type=str2bool, default=False)
    parser.add_argument("--cache-poisoned-train", type=str2bool, default=True)
    parser.add_argument("--poison-cache-dir", type=Path, default=PROJECT_ROOT / "src" / "outputs" / "poison_cache")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--poison-rate", type=float, default=0.1)
    parser.add_argument("--trigger-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--trigger-type", default="square", choices=SUPPORTED_TRIGGER_TYPES)
    parser.add_argument("--position", default="bottom_right", choices=["bottom_right", "top_left"])
    parser.add_argument("--norm-type", default="group", choices=getattr(model_module, "SUPPORTED_NORM_TYPES", ("group", "batch", "none")))
    return parser.parse_args()


def configure_context(device_target: str, ms_mode: str) -> None:
    context.set_context(
        mode=context.GRAPH_MODE if ms_mode.upper() == "GRAPH" else context.PYNATIVE_MODE,
        device_target=device_target,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0 or args.num_classes <= 0:
        raise ValueError("epochs、batch-size、image-size、num-classes 必须为正整数")
    if args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("lr 和 grad-clip 必须大于 0")
    if args.num_parallel_workers < 0 or args.prefetch_size < 0:
        raise ValueError("num-parallel-workers 和 prefetch-size 必须大于等于 0")
    if args.log_interval <= 0:
        raise ValueError("log-interval 必须大于 0")
    if args.eval_interval <= 0:
        raise ValueError("eval-interval 必须大于 0")
    if not args.train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在：{args.train_dir}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在：{args.test_dir}")
    if args.task == "badnet":
        if not 0.0 <= args.poison_rate <= 1.0:
            raise ValueError("poison-rate 必须位于 [0, 1]")
        if not 0.0 <= args.alpha <= 1.0:
            raise ValueError("alpha 必须位于 [0, 1]")
        if args.trigger_size <= 0:
            raise ValueError("trigger-size 必须为正整数")
        if not 0 <= args.target_label < args.num_classes:
            raise ValueError(f"target-label 超出类别范围：{args.target_label}")


def build_network(args: argparse.Namespace, num_classes: int) -> nn.Cell:
    create_model_fn = getattr(model_module, "create_model", None)
    if callable(create_model_fn):
        for kwargs in (
            {"num_classes": num_classes, "norm_type": args.norm_type},
            {"class_num": num_classes, "norm_type": args.norm_type},
            {"num_classes": num_classes},
            {"class_num": num_classes},
            {},
        ):
            try:
                return create_model_fn(**kwargs)
            except TypeError:
                continue
    simple_cnn_cls = getattr(model_module, "SimpleCNN", None)
    if simple_cnn_cls is None:
        raise AttributeError("src/model.py 中既没有 create_model(...)，也没有 SimpleCNN")
    for kwargs in ({"num_classes": num_classes, "norm_type": args.norm_type}, {"num_classes": num_classes}, {}):
        try:
            return simple_cnn_cls(**kwargs)
        except TypeError:
            continue
    raise TypeError("无法使用当前参数创建模型")


def format_class_name(label: int) -> str:
    return f"{int(label):05d}"


def load_image_as_float_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_float_image(array: np.ndarray, save_path: Path) -> None:
    ensure_dir(save_path.parent)
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] in (1, 3):
        data = np.transpose(data, (1, 2, 0))
    Image.fromarray((np.clip(data, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(save_path)


def make_unique_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path
    index = 1
    while True:
        candidate = target_path.with_name(f"{target_path.stem}_{index}{target_path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def read_json_if_exists(json_path: Path) -> Optional[dict[str, object]]:
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_poison_cache_name(args: argparse.Namespace) -> str:
    poison_rate_text = f"{float(args.poison_rate):.4f}".replace(".", "p")
    alpha_text = f"{float(args.alpha):.3f}".replace(".", "p")
    trigger_type = str(args.trigger_type).lower().strip()
    return f"target_{int(args.target_label):02d}_pr_{poison_rate_text}_ts_{int(args.trigger_size):02d}_alpha_{alpha_text}_{trigger_type}_{args.position}_seed_{int(args.seed)}"


def try_reuse_poisoned_train(cache_root: Path, source_train_dir: Path) -> Optional[dict[str, object]]:
    metadata = read_json_if_exists(cache_root / "metadata.json")
    if metadata is None or not cache_root.exists() or not scan_class_folders(cache_root):
        return None
    if Path(str(metadata.get("source_train_dir", ""))).resolve() != source_train_dir.resolve():
        return None
    metadata["poisoned_train_dir"] = str(cache_root)
    metadata["cache_hit"] = True
    return metadata


def build_poisoned_train_copy(args: argparse.Namespace, runtime_dir: Path) -> tuple[Path, dict[str, object]]:
    records = scan_class_folders(args.train_dir)
    if not records:
        raise ValueError(f"训练集为空，无法构建投毒副本：{args.train_dir}")
    cache_base_dir = ensure_dir(args.poison_cache_dir)
    cache_root = cache_base_dir / build_poison_cache_name(args)
    if args.cache_poisoned_train:
        reused_metadata = try_reuse_poisoned_train(cache_root, args.train_dir)
        if reused_metadata is not None:
            save_json(reused_metadata, runtime_dir / "poisoned_train_metadata.json")
            print(f"[信息] 复用已有投毒训练集副本：{cache_root}")
            return cache_root, reused_metadata
    poisoned_root = cache_root if args.cache_poisoned_train and not cache_root.exists() else runtime_dir / f"poisoned_train_{timestamp()}"
    poisoned_root = ensure_dir(poisoned_root)
    preview_path = poisoned_root / "poison_preview.png"
    labels = np.asarray([label for _, label in records], dtype=np.int32)
    candidates = np.where(labels != int(args.target_label))[0]
    poison_count = int(len(candidates) * float(args.poison_rate))
    if args.poison_rate > 0 and poison_count == 0 and len(candidates) > 0:
        poison_count = 1
    rng = np.random.default_rng(args.seed)
    poison_ids = set(rng.choice(candidates, size=poison_count, replace=False).tolist()) if poison_count > 0 else set()
    class_counter: Counter[str] = Counter()
    preview_saved = False
    copied_count = poisoned_count = failed_count = 0
    for label in sorted(set(labels.tolist()) | {int(args.target_label)}):
        ensure_dir(poisoned_root / format_class_name(label))
    for index, (source_path, source_label) in enumerate(records):
        try:
            if index in poison_ids:
                clean_image = load_image_as_float_array(source_path)
                poisoned_image = add_trigger(
                    image=clean_image,
                    trigger_size=args.trigger_size,
                    alpha=args.alpha,
                    position=args.position,
                    trigger_type=args.trigger_type,
                )
                target_dir = poisoned_root / format_class_name(args.target_label)
                target_path = make_unique_path(target_dir / f"poison_from_{format_class_name(source_label)}_{source_path.stem}.png")
                save_float_image(poisoned_image, target_path)
                if not preview_saved:
                    save_poison_preview(clean_image, poisoned_image, preview_path)
                    preview_saved = True
                poisoned_count += 1
                class_counter[target_dir.name] += 1
            else:
                target_dir = poisoned_root / format_class_name(source_label)
                shutil.copy2(source_path, make_unique_path(target_dir / source_path.name))
                copied_count += 1
                class_counter[target_dir.name] += 1
        except Exception as exc:
            failed_count += 1
            print(f"[警告] 处理训练样本失败，已跳过：{source_path}，错误信息：{exc}")
    metadata = {
        "task": "badnet",
        "source_train_dir": str(args.train_dir),
        "poisoned_train_dir": str(poisoned_root),
        "target_label": int(args.target_label),
        "poison_rate": float(args.poison_rate),
        "trigger_size": int(args.trigger_size),
        "alpha": float(args.alpha),
        "trigger_type": args.trigger_type,
        "position": args.position,
        "poison_cache_dir": str(cache_base_dir),
        "total_samples": len(records),
        "copied_clean_samples": copied_count,
        "poisoned_samples": poisoned_count,
        "failed_samples": failed_count,
        "preview_path": str(preview_path) if preview_saved else None,
        "class_distribution": dict(sorted(class_counter.items())),
        "cache_hit": False,
    }
    save_json(metadata, runtime_dir / "poisoned_train_metadata.json")
    save_json(metadata, poisoned_root / "metadata.json")
    print(f"[信息] 投毒训练集副本已生成：{poisoned_root}")
    return poisoned_root, metadata


def build_dataset_factory(args: argparse.Namespace, task: str, train_source_dir: Path, test_dir: Path) -> tuple[Callable[[bool], ds.Dataset], Callable[[], ds.Dataset]]:
    dataset_kwargs = {
        "image_size": (args.image_size, args.image_size),
        "batch_size": args.batch_size,
        "drop_remainder": args.drop_remainder,
        "num_parallel_workers": args.num_parallel_workers if args.num_parallel_workers > 0 else None,
        "prefetch_size": args.prefetch_size if args.prefetch_size > 0 else None,
        "server_fast": args.server_fast,
    }
    if task == "clean":
        def train_builder(shuffle: bool) -> ds.Dataset:
            return create_train_dataset(data_dir=train_source_dir, shuffle=shuffle, augment=True, **dataset_kwargs)
    else:
        def train_builder(shuffle: bool) -> ds.Dataset:
            return create_image_dataset(data_dir=train_source_dir, shuffle=shuffle, is_training=False, **dataset_kwargs)
    def test_builder() -> ds.Dataset:
        return create_test_dataset(data_dir=test_dir, image_size=(args.image_size, args.image_size), batch_size=args.batch_size, shuffle=False, drop_remainder=False, num_parallel_workers=args.num_parallel_workers if args.num_parallel_workers > 0 else None, prefetch_size=args.prefetch_size if args.prefetch_size > 0 else None, server_fast=args.server_fast)
    return train_builder, test_builder

def to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def normalize_label_array(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim == 0:
        array = array.reshape(1)
    return np.squeeze(array).astype(np.int32).reshape(-1)


def has_nan(array: np.ndarray) -> bool:
    return bool(np.isnan(array).any())


def has_inf(array: np.ndarray) -> bool:
    return bool(np.isinf(array).any())


def tensor_stats(array: np.ndarray) -> dict[str, object]:
    data = np.asarray(array)
    return {"shape": tuple(int(x) for x in data.shape), "min": float(np.min(data)), "max": float(np.max(data)), "mean": float(np.mean(data)), "has_nan": has_nan(data), "has_inf": has_inf(data)}


def validate_labels(labels: np.ndarray, num_classes: int, prefix: str) -> None:
    if labels.size == 0:
        raise ValueError(f"{prefix} 标签为空")
    if int(labels.min()) < 0 or int(labels.max()) >= int(num_classes):
        raise ValueError(f"{prefix} 标签越界，范围为 [{int(labels.min())}, {int(labels.max())}]，num_classes={num_classes}")


def batch_to_ms_tensors(batch: dict[str, object]) -> tuple[ms.Tensor, ms.Tensor]:
    image_value = batch["image"]
    label_value = batch["label"]
    if isinstance(image_value, ms.Tensor):
        image_tensor = image_value
    else:
        image_tensor = ms.Tensor(np.asarray(image_value, dtype=np.float32), ms.float32)
    if image_tensor.dtype != ms.float32:
        image_tensor = ops.cast(image_tensor, ms.float32)
    if isinstance(label_value, ms.Tensor):
        label_tensor = ops.cast(label_value, ms.int32)
    else:
        label_tensor = ms.Tensor(normalize_label_array(np.asarray(label_value)), ms.int32)
    label_tensor = ops.reshape(label_tensor, (-1,))
    return image_tensor, label_tensor


def batch_to_tensors(batch: dict[str, object]) -> tuple[np.ndarray, np.ndarray, ms.Tensor, ms.Tensor]:
    image_tensor, label_tensor = batch_to_ms_tensors(batch)
    images = np.asarray(to_numpy(image_tensor), dtype=np.float32)
    labels = normalize_label_array(to_numpy(label_tensor))
    return images, labels, image_tensor, label_tensor


def first_batch(dataset: ds.Dataset, output_numpy: bool = False) -> dict[str, object]:
    try:
        return next(dataset.create_dict_iterator(output_numpy=output_numpy))
    except StopIteration as exc:
        raise ValueError("数据集为空，无法读取第一个 batch") from exc


def inspect_input_batch(name: str, batch: dict[str, object], num_classes: int) -> dict[str, object]:
    images, labels, _, _ = batch_to_tensors(batch)
    validate_labels(labels, num_classes, f"{name} 第一个 batch")
    stats = {"image_shape": tuple(int(x) for x in images.shape), "image_dtype": str(images.dtype), "label_shape": tuple(int(x) for x in labels.shape), "label_dtype": str(labels.dtype), "label_min": int(labels.min()), "label_max": int(labels.max()), "label_unique_preview": np.unique(labels)[:10].tolist(), "image_min": float(images.min()), "image_max": float(images.max()), "image_mean": float(images.mean()), "image_has_nan": has_nan(images), "image_has_inf": has_inf(images)}
    print(f"[诊断] {name} 第一个 batch 检查：")
    print(f"  image shape: {stats['image_shape']}")
    print(f"  image dtype: {stats['image_dtype']}")
    print(f"  label shape: {stats['label_shape']}")
    print(f"  label dtype: {stats['label_dtype']}")
    print(f"  label min: {stats['label_min']}")
    print(f"  label max: {stats['label_max']}")
    print(f"  label unique 前若干个值: {stats['label_unique_preview']}")
    print(f"  image min: {stats['image_min']:.6f}")
    print(f"  image max: {stats['image_max']:.6f}")
    print(f"  image mean: {stats['image_mean']:.6f}")
    print(f"  是否存在 nan: {stats['image_has_nan']}")
    print(f"  是否存在 inf: {stats['image_has_inf']}")
    return stats


class NumericalIssueError(RuntimeError):
    def __init__(self, stage: str, message: str, details: Optional[dict[str, object]] = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


def scalar(value) -> float:
    current = value
    while isinstance(current, (tuple, list)) and current:
        current = current[0]
    if hasattr(current, "asnumpy"):
        current = current.asnumpy()
    return float(np.asarray(current, dtype=np.float32).mean())


def stable_sparse_ce_numpy(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    logits64 = np.asarray(logits, dtype=np.float64)
    labels32 = normalize_label_array(labels)
    batch_indices = np.arange(labels32.shape[0], dtype=np.int64)
    max_logits = np.max(logits64, axis=1, keepdims=True)
    shifted = logits64 - max_logits
    sum_exp = np.sum(np.exp(shifted), axis=1)
    logsumexp = np.log(sum_exp) + max_logits[:, 0]
    true_label_logits = logits64[batch_indices, labels32]
    per_sample_loss = logsumexp - true_label_logits
    max_abs_position = np.unravel_index(int(np.argmax(np.abs(logits64))), logits64.shape)
    max_abs_sample_index = int(max_abs_position[0])
    max_abs_class_index = int(max_abs_position[1])
    return {"per_sample_loss_preview": [float(x) for x in per_sample_loss[:5]], "true_label_logits_preview": [float(x) for x in true_label_logits[:5]], "per_sample_loss_min": float(np.min(per_sample_loss)), "per_sample_loss_max": float(np.max(per_sample_loss)), "per_sample_loss_mean": float(np.mean(per_sample_loss)), "per_sample_loss_has_nan": has_nan(per_sample_loss), "per_sample_loss_has_inf": has_inf(per_sample_loss), "reduced_loss_numpy": float(np.mean(per_sample_loss)), "max_abs_logit_sample_index": max_abs_sample_index, "max_abs_logit_class_index": max_abs_class_index, "max_abs_logit_value": float(logits64[max_abs_sample_index, max_abs_class_index]), "max_abs_logit_abs_value": float(np.abs(logits64[max_abs_sample_index, max_abs_class_index]))}


def merge_loss_details(details: dict[str, object], loss_details: dict[str, object], forward_loss: float) -> dict[str, object]:
    merged = dict(details)
    merged.update(loss_details)
    merged["forward_loss"] = float(forward_loss)
    merged["per_sample_loss_already_inf"] = bool(loss_details["per_sample_loss_has_inf"] or loss_details["per_sample_loss_has_nan"])
    merged["reduction_after_per_sample_became_inf"] = bool((np.isinf(forward_loss) or np.isnan(forward_loss)) and not merged["per_sample_loss_already_inf"])
    return merged


def grad_stats(grads, names: list[str]) -> dict[str, object]:
    has_nan_flag = has_inf_flag = False
    max_abs = 0.0
    sample: list[dict[str, object]] = []
    first_issue = None
    for index, grad in enumerate(grads):
        grad_array = np.asarray(to_numpy(grad), dtype=np.float32)
        grad_has_nan = has_nan(grad_array)
        grad_has_inf = has_inf(grad_array)
        grad_max_abs = float(np.max(np.abs(grad_array))) if grad_array.size > 0 else 0.0
        name = names[index] if index < len(names) else f"grad_{index}"
        has_nan_flag = has_nan_flag or grad_has_nan
        has_inf_flag = has_inf_flag or grad_has_inf
        max_abs = max(max_abs, grad_max_abs)
        if len(sample) < 3:
            sample.append({"name": name, "max_abs": grad_max_abs, "has_nan": grad_has_nan, "has_inf": grad_has_inf})
        if first_issue is None and (grad_has_nan or grad_has_inf):
            first_issue = {"name": name, "max_abs": grad_max_abs, "has_nan": grad_has_nan, "has_inf": grad_has_inf}
    return {"has_nan": has_nan_flag, "has_inf": has_inf_flag, "max_abs": max_abs, "sample": sample, "first_issue": first_issue}


def param_health(network: nn.Cell) -> dict[str, object]:
    issues = []
    first_issue = None
    for parameter in network.trainable_params():
        array = to_numpy(parameter).astype(np.float32)
        if has_nan(array) or has_inf(array):
            issue = {"name": parameter.name, "has_nan": has_nan(array), "has_inf": has_inf(array)}
            issues.append(issue)
            if first_issue is None:
                first_issue = issue
    return {"has_nan": any(item["has_nan"] for item in issues), "has_inf": any(item["has_inf"] for item in issues), "issues": issues[:3], "first_issue": first_issue}


class StableSparseCrossEntropy(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.reduce_max = ops.ReduceMax(keep_dims=True)
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.gather_d = ops.GatherD()
        self.reshape = ops.Reshape()

    def construct(self, logits: ms.Tensor, labels: ms.Tensor) -> ms.Tensor:
        logits = self.cast(logits, ms.float32)
        labels = self.cast(labels, ms.int32)
        label_index = self.expand_dims(labels, 1)
        max_logits = self.reduce_max(logits, 1)
        shifted_logits = logits - max_logits
        sum_exp = self.reduce_sum(ops.exp(shifted_logits), 1)
        logsumexp = ops.log(sum_exp) + max_logits
        true_label_logits = self.gather_d(logits, 1, label_index)
        per_sample_loss = self.reshape(logsumexp - true_label_logits, (-1,))
        return self.reduce_mean(per_sample_loss)


def build_optimizer(network: nn.Cell, lr: float) -> tuple[nn.Optimizer, dict[str, object]]:
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=1e-4, use_nesterov=True)
    return optimizer, {"name": optimizer.__class__.__name__, "learning_rate": float(lr), "grad_clip": None, "momentum": 0.9, "weight_decay": 1e-4, "use_nesterov": True}


def print_optimizer_info(info: dict[str, object]) -> None:
    print(f"[信息] optimizer: {info['name']}")
    print(f"[信息] learning rate: {info['learning_rate']}")
    print(f"[信息] grad clip: global norm {info['grad_clip']}")
    print(f"[信息] momentum: {info['momentum']}")
    print(f"[信息] weight decay: {info['weight_decay']}")
    print(f"[信息] use nesterov: {info['use_nesterov']}")


def print_runtime_config(args: argparse.Namespace) -> None:
    worker_text = args.num_parallel_workers if args.num_parallel_workers > 0 else "auto"
    prefetch_text = args.prefetch_size if args.prefetch_size > 0 else "default"
    print(f"[信息] server-fast: {args.server_fast}")
    print(f"[信息] num-parallel-workers: {worker_text}")
    print(f"[信息] prefetch-size: {prefetch_text}")
    print(f"[信息] drop-remainder: {args.drop_remainder}")
    print(f"[信息] log-interval: {args.log_interval}")
    print(f"[信息] eval-interval: {args.eval_interval}")
    print(f"[信息] eval-train-acc: {args.eval_train_acc}")
    print(f"[信息] cache-poisoned-train: {args.cache_poisoned_train}")
    print(f"[信息] poison-cache-dir: {args.poison_cache_dir}")
    print(f"[信息] norm-type: {args.norm_type}")
    if args.task == "badnet":
        print(f"[信息] trigger-type: {args.trigger_type}")
    if args.server_fast and args.ms_mode.upper() != "GRAPH":
        print("[说明] server-fast 一般和 GRAPH 模式一起用，Python 侧调度会更少。")


def print_issue(stage: str, message: str, details: dict[str, object], optimizer_info: dict[str, object]) -> None:
    print("[错误] 训练过程中检测到数值异常，已停止当前训练。")
    print(f"[错误] 异常阶段: {stage}")
    print(f"[错误] 异常信息: {message}")
    for key in ["epoch", "step", "forward_loss", "loss", "label_min", "label_max", "logits_min", "logits_max", "image_has_nan", "image_has_inf", "logits_has_nan", "logits_has_inf", "loss_has_nan", "loss_has_inf", "grads_has_nan", "grads_has_inf", "grad_max_abs", "params_has_nan", "params_has_inf", "per_sample_loss_min", "per_sample_loss_max", "per_sample_loss_mean", "per_sample_loss_has_nan", "per_sample_loss_has_inf", "reduced_loss_numpy", "per_sample_loss_already_inf", "reduction_after_per_sample_became_inf", "max_abs_logit_sample_index", "max_abs_logit_class_index", "max_abs_logit_value", "max_abs_logit_abs_value"]:
        if key in details:
            print(f"[错误] {key}: {details[key]}")
    if "per_sample_loss_preview" in details:
        print(f"[错误] 前几个样本的逐样本 loss: {details['per_sample_loss_preview']}")
    if "true_label_logits_preview" in details:
        print(f"[错误] 前几个样本真实标签对应的 logit: {details['true_label_logits_preview']}")
    if details.get("grad_issue") is not None:
        print(f"[错误] 首个异常梯度: {details['grad_issue']}")
    if details.get("param_issue") is not None:
        print(f"[错误] 首个异常参数: {details['param_issue']}")
    if "param_issue_sample" in details:
        print(f"[错误] 参数异常样例: {details['param_issue_sample']}")
    print(f"[错误] 当前 optimizer: {optimizer_info['name']}")
    print(f"[错误] 当前 learning rate: {optimizer_info['learning_rate']}")
    print(f"[错误] 当前 grad clip: {optimizer_info['grad_clip']}")


class StableTrainer:
    def __init__(self, network: nn.Cell, loss_fn: nn.Cell, optimizer: nn.Optimizer, grad_clip: float, num_classes: int, debug_enabled: bool = True) -> None:
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_clip = float(grad_clip)
        self.num_classes = int(num_classes)
        self.debug_enabled = bool(debug_enabled)
        self.weights = network.trainable_params()
        self.weight_names = [param.name for param in self.weights]
        value_and_grad = getattr(ms, "value_and_grad", ops.value_and_grad)
        self.grad_fn = value_and_grad(self.forward_fn, None, self.weights, has_aux=True)

    def forward_fn(self, data: ms.Tensor, label: ms.Tensor):
        logits = self.network(data)
        loss = self.loss_fn(logits, label)
        return loss, logits

    def _attach_image_health(self, details: dict[str, object], image_tensor: ms.Tensor) -> dict[str, object]:
        if details.get("image_has_nan") is None or details.get("image_has_inf") is None:
            images = np.asarray(to_numpy(image_tensor), dtype=np.float32)
            details["image_has_nan"] = has_nan(images)
            details["image_has_inf"] = has_inf(images)
        return details

    def inspect_forward(self, batch: dict[str, object]) -> dict[str, object]:
        images, labels, image_tensor, label_tensor = batch_to_tensors(batch)
        validate_labels(labels, self.num_classes, "训练前前向检查")
        loss, logits = self.forward_fn(image_tensor, label_tensor)
        logits_array = np.asarray(to_numpy(logits), dtype=np.float32)
        logits_info = tensor_stats(logits_array)
        loss_value = scalar(loss)
        loss_details = stable_sparse_ce_numpy(logits_array, labels)
        loss_has_nan = bool(np.isnan(loss_value))
        loss_has_inf = bool(np.isinf(loss_value))
        print("[诊断] 训练前模型前向检查：")
        print(f"  logits shape: {logits_info['shape']}")
        print(f"  logits min: {logits_info['min']:.6f}")
        print(f"  logits max: {logits_info['max']:.6f}")
        print(f"  logits mean: {logits_info['mean']:.6f}")
        print(f"  logits 是否存在 nan: {logits_info['has_nan']}")
        print(f"  logits 是否存在 inf: {logits_info['has_inf']}")
        print(f"  初始 loss 数值: {loss_value:.6f}")
        print(f"  loss 是否为 nan: {loss_has_nan}")
        print(f"  loss 是否为 inf: {loss_has_inf}")
        if has_nan(images) or has_inf(images):
            raise NumericalIssueError("输入异常", "训练前检查发现输入图像包含 nan 或 inf", {"image_has_nan": has_nan(images), "image_has_inf": has_inf(images)})
        if logits_info["has_nan"] or logits_info["has_inf"]:
            raise NumericalIssueError("前向输出异常", "训练前检查发现 logits 包含 nan 或 inf", {"logits_min": logits_info["min"], "logits_max": logits_info["max"], "logits_has_nan": logits_info["has_nan"], "logits_has_inf": logits_info["has_inf"]})
        if loss_has_nan or loss_has_inf:
            raise NumericalIssueError("loss 异常", "训练前检查发现初始 loss 为 nan 或 inf", merge_loss_details({"loss": loss_value, "loss_has_nan": loss_has_nan, "loss_has_inf": loss_has_inf}, loss_details, loss_value))
        return {"logits": logits_info, "loss": {"value": loss_value, "has_nan": loss_has_nan, "has_inf": loss_has_inf, "loss_numpy": loss_details["reduced_loss_numpy"]}}

    def train_step(self, batch: dict[str, object], stats_level: str = "none") -> dict[str, object]:
        self.network.set_train(True)
        image_tensor, label_tensor = batch_to_ms_tensors(batch)
        labels = normalize_label_array(to_numpy(label_tensor))
        validate_labels(labels, self.num_classes, "训练过程 step 检查")
        image_has_nan = image_has_inf = None
        if stats_level == "full":
            images = np.asarray(to_numpy(image_tensor), dtype=np.float32)
            image_has_nan = has_nan(images)
            image_has_inf = has_inf(images)
            if image_has_nan or image_has_inf:
                raise NumericalIssueError("输入异常", "训练过程中发现输入图像存在 nan 或 inf", {"image_has_nan": image_has_nan, "image_has_inf": image_has_inf, "label_min": int(labels.min()), "label_max": int(labels.max())})
        (loss, logits), grads = self.grad_fn(image_tensor, label_tensor)
        loss_value = scalar(loss)
        loss_has_nan = bool(np.isnan(loss_value))
        loss_has_inf = bool(np.isinf(loss_value))
        logits_array = np.asarray(to_numpy(logits), dtype=np.float32)
        logits_info = tensor_stats(logits_array)
        if logits_array.ndim != 2 or logits_array.shape[1] != self.num_classes:
            raise NumericalIssueError("前向输出异常", f"logits 形状不符合预期：{tuple(logits_array.shape)}", {"logits_shape": tuple(int(x) for x in logits_array.shape)})
        loss_details = stable_sparse_ce_numpy(logits_array, labels)
        common_details = {"loss": loss_value, "label_min": int(labels.min()), "label_max": int(labels.max()), "logits_min": logits_info["min"], "logits_max": logits_info["max"], "image_has_nan": image_has_nan, "image_has_inf": image_has_inf, "logits_has_nan": logits_info["has_nan"], "logits_has_inf": logits_info["has_inf"], "loss_has_nan": loss_has_nan, "loss_has_inf": loss_has_inf}
        if logits_info["has_nan"] or logits_info["has_inf"]:
            raise NumericalIssueError("前向输出异常", "训练过程中发现 logits 存在 nan 或 inf", merge_loss_details(self._attach_image_health(common_details, image_tensor), loss_details, loss_value))
        if loss_has_nan or loss_has_inf:
            raise NumericalIssueError("loss 异常", "训练过程中发现 loss 为 nan 或 inf", merge_loss_details(self._attach_image_health(common_details, image_tensor), loss_details, loss_value))
        clipped_grads = ops.clip_by_global_norm(grads, self.grad_clip)
        clipped_info = None
        if self.debug_enabled or stats_level == "full":
            clipped_info = grad_stats(clipped_grads, self.weight_names)
            if clipped_info["has_nan"] or clipped_info["has_inf"]:
                raise NumericalIssueError("grads 异常", "梯度裁剪后 grads 仍然存在 nan 或 inf", merge_loss_details({**self._attach_image_health(common_details, image_tensor), "grads_has_nan": clipped_info["has_nan"], "grads_has_inf": clipped_info["has_inf"], "grad_max_abs": clipped_info["max_abs"], "grad_issue": clipped_info["first_issue"]}, loss_details, loss_value))
        self.optimizer(clipped_grads)
        if self.debug_enabled and stats_level == "full":
            param_info = param_health(self.network)
            if param_info["has_nan"] or param_info["has_inf"]:
                raise NumericalIssueError("参数更新后异常", "优化器更新后检测到参数包含 nan 或 inf", merge_loss_details({**self._attach_image_health(common_details, image_tensor), "params_has_nan": param_info["has_nan"], "params_has_inf": param_info["has_inf"], "param_issue": param_info["first_issue"], "param_issue_sample": param_info["issues"]}, loss_details, loss_value))
        result = {**common_details, "loss_numpy": loss_details["reduced_loss_numpy"]}
        if clipped_info is not None:
            result["grad_max_abs"] = clipped_info["max_abs"]
        return result


def run_epoch_train(trainer: StableTrainer, train_dataset: ds.Dataset, epoch_index: int, optimizer_info: dict[str, object], debug_steps: int = 3, log_interval: int = 100, server_fast: bool = False) -> float:
    losses = []
    for step_index, batch in enumerate(train_dataset.create_dict_iterator(output_numpy=False), start=1):
        stats_level = "light" if server_fast and (step_index == 1 or step_index % log_interval == 0) else ("full" if not server_fast and step_index <= debug_steps else "none")
        try:
            info = trainer.train_step(batch, stats_level=stats_level)
        except NumericalIssueError as exc:
            details = dict(exc.details)
            details["epoch"] = epoch_index
            details["step"] = step_index
            print_issue(exc.stage, str(exc), details, optimizer_info)
            raise NumericalIssueError(exc.stage, str(exc), details) from exc
        losses.append(float(info["loss"]))
        if server_fast:
            if step_index == 1 or step_index % log_interval == 0:
                print(f"[进度][Epoch {epoch_index} Step {step_index}] loss: {info['loss']:.6f} | logits min/max: {info['logits_min']:.6f}/{info['logits_max']:.6f}")
        elif step_index <= debug_steps:
            print(f"[诊断][Epoch {epoch_index} Step {step_index}] step loss: {info['loss']:.6f} | label min/max: {info['label_min']}/{info['label_max']} | logits min/max: {info['logits_min']:.6f}/{info['logits_max']:.6f} | image nan/inf: {info['image_has_nan']}/{info['image_has_inf']} | logits nan/inf: {info['logits_has_nan']}/{info['logits_has_inf']} | loss nan/inf: {info['loss_has_nan']}/{info['loss_has_inf']}")
    if not losses:
        raise ValueError("训练数据集为空，当前 epoch 没有任何 step")
    return float(sum(losses) / len(losses))


def evaluate_accuracy(network: nn.Cell, dataset: ds.Dataset, num_classes: int, split_name: str) -> float:
    network.set_train(False)
    total = 0
    correct = 0
    for batch in dataset.create_dict_iterator(output_numpy=False):
        image_tensor, label_tensor = batch_to_ms_tensors(batch)
        labels = normalize_label_array(to_numpy(label_tensor))
        validate_labels(labels, num_classes, f"{split_name} 评估")
        logits = np.asarray(to_numpy(network(image_tensor)), dtype=np.float32)
        info = tensor_stats(logits)
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise ValueError(f"{split_name} 评估阶段 logits 形状不符合预期：{tuple(logits.shape)}")
        if info["has_nan"] or info["has_inf"]:
            raise NumericalIssueError("评估前向异常", f"{split_name} 评估阶段发现 logits 存在 nan 或 inf", {"logits_min": info["min"], "logits_max": info["max"], "logits_has_nan": info["has_nan"], "logits_has_inf": info["has_inf"]})
        correct += int((logits.argmax(axis=1).astype(np.int32) == labels).sum())
        total += int(labels.shape[0])
    network.set_train(True)
    if total == 0:
        raise ValueError(f"{split_name} 评估数据集为空")
    return float(correct / total)


def build_log_payload(args: argparse.Namespace, train_source_dir: Path, save_dir: Path, optimizer_info: dict[str, object], loss_name: str, history: list[dict[str, object]], poison_metadata: Optional[dict[str, object]], diagnostics: dict[str, object], best_epoch: int, best_test_acc: float, best_ckpt_path: Optional[Path], best_ckpt_save_method: Optional[str], status: str = "running", error_stage: Optional[str] = None, error_message: Optional[str] = None) -> dict[str, object]:
    return {
        "task": args.task,
        "ms_mode": args.ms_mode,
        "device_target": args.device_target,
        "train_dir": str(args.train_dir),
        "effective_train_dir": str(train_source_dir),
        "test_dir": str(args.test_dir),
        "save_dir": str(save_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": float(args.lr),
        "grad_clip": float(args.grad_clip),
        "num_classes": args.num_classes,
        "image_size": args.image_size,
        "seed": args.seed,
        "server_fast": args.server_fast,
        "num_parallel_workers": args.num_parallel_workers,
        "prefetch_size": args.prefetch_size,
        "drop_remainder": args.drop_remainder,
        "log_interval": args.log_interval,
        "eval_interval": args.eval_interval,
        "eval_train_acc": args.eval_train_acc,
        "cache_poisoned_train": args.cache_poisoned_train,
        "poison_cache_dir": str(args.poison_cache_dir),
        "norm_type": args.norm_type,
        "trigger_type": getattr(args, "trigger_type", None),
        "optimizer": optimizer_info,
        "loss_name": loss_name,
        "status": status,
        "error_stage": error_stage,
        "error_message": error_message,
        "best_epoch": best_epoch,
        "best_test_acc": float(best_test_acc),
        "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
        "best_ckpt_save_method": best_ckpt_save_method,
        "history": history,
        "poison": poison_metadata,
        "diagnostics": diagnostics,
    }


def save_log(log_path: Path, content: dict[str, object]) -> None:
    save_json(content, log_path)


def save_checkpoint_with_fallback(network: nn.Cell, ckpt_path: Path) -> str:
    try:
        ms.save_checkpoint(network, str(ckpt_path))
        return "native"
    except Exception as exc:
        tmp_path = ckpt_path.with_suffix(".tmp")
        if not tmp_path.exists():
            raise
        shutil.copy2(tmp_path, ckpt_path)
        print(f"[警告] checkpoint 原生重命名失败，已改为复制临时文件生成：{ckpt_path}")
        print(f"[警告] 原始异常信息：{exc}")
        cleanup_exc: Optional[OSError] = None
        for _ in range(5):
            try:
                tmp_path.unlink()
                return "copied_from_tmp"
            except OSError as current_exc:
                cleanup_exc = current_exc
                time.sleep(0.2)
        print(f"[警告] 临时 checkpoint 删除失败，可手动清理：{tmp_path}")
        print(f"[警告] 删除异常信息：{cleanup_exc}")
        return "copied_from_tmp_with_tmp_left"


def make_best_epoch_ckpt_path(save_dir: Path, epoch: int) -> Path:
    return save_dir / f"best_epoch_{int(epoch):02d}.ckpt"


def main() -> dict[str, object]:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    configure_context(args.device_target, args.ms_mode)
    save_dir = ensure_dir(args.save_dir)
    runtime_dir = ensure_dir(save_dir / "runtime")
    log_path = save_dir / "train_log.json"
    print("=" * 72)
    print("GTSRB 训练脚本")
    print("=" * 72)
    print(f"[信息] task: {args.task}")
    print(f"[信息] ms-mode: {args.ms_mode}")
    print(f"[信息] device-target: {args.device_target}")
    print(f"[信息] train-dir: {args.train_dir}")
    print(f"[信息] test-dir: {args.test_dir}")
    print(f"[信息] save-dir: {save_dir}")
    print_runtime_config(args)
    train_source_dir = args.train_dir
    poison_metadata = None
    if args.task == "badnet":
        train_source_dir, poison_metadata = build_poisoned_train_copy(args, runtime_dir)
    train_builder, test_builder = build_dataset_factory(args, args.task, train_source_dir, args.test_dir)
    initial_train_dataset = train_builder(False)
    initial_test_dataset = test_builder()
    if initial_train_dataset.get_dataset_size() <= 0 or initial_test_dataset.get_dataset_size() <= 0:
        raise ValueError("训练集或测试集为空，无法继续")
    print(f"[信息] 训练集 batch 数：{initial_train_dataset.get_dataset_size()}")
    print(f"[信息] 测试集 batch 数：{initial_test_dataset.get_dataset_size()}")
    train_batch = first_batch(initial_train_dataset, output_numpy=False)
    test_batch = first_batch(initial_test_dataset, output_numpy=False)
    train_batch_stats = inspect_input_batch("train", train_batch, args.num_classes)
    test_batch_stats = inspect_input_batch("test", test_batch, args.num_classes)
    network = build_network(args, args.num_classes)
    loss_fn = StableSparseCrossEntropy()
    optimizer, optimizer_info = build_optimizer(network, args.lr)
    optimizer_info["grad_clip"] = float(args.grad_clip)
    print_optimizer_info(optimizer_info)
    trainer = StableTrainer(network, loss_fn, optimizer, args.grad_clip, args.num_classes, debug_enabled=not args.server_fast)
    try:
        initial_forward = trainer.inspect_forward(train_batch)
    except NumericalIssueError as exc:
        print_issue(exc.stage, str(exc), exc.details, optimizer_info)
        save_log(log_path, build_log_payload(args, train_source_dir, save_dir, optimizer_info, loss_fn.__class__.__name__, [], poison_metadata, {"train_first_batch": train_batch_stats, "test_first_batch": test_batch_stats}, 0, -1.0, None, None, status="failed_before_training_due_to_numerical_issue", error_stage=exc.stage, error_message=str(exc)))
        raise
    history: list[dict[str, object]] = []
    best_test_acc = -1.0
    best_epoch = 0
    best_ckpt_path: Optional[Path] = None
    best_ckpt_save_method = None
    for epoch in range(1, args.epochs + 1):
        train_dataset = train_builder(True)
        should_eval = epoch == args.epochs or epoch % args.eval_interval == 0
        train_acc: Optional[float] = None
        test_acc: Optional[float] = None
        try:
            train_loss = run_epoch_train(trainer, train_dataset, epoch, optimizer_info, debug_steps=3, log_interval=args.log_interval, server_fast=args.server_fast)
            if should_eval:
                if args.eval_train_acc:
                    train_eval_dataset = train_builder(False)
                    train_acc = evaluate_accuracy(network, train_eval_dataset, args.num_classes, "train")
                test_dataset = test_builder()
                test_acc = evaluate_accuracy(network, test_dataset, args.num_classes, "test")
        except NumericalIssueError as exc:
            save_log(log_path, build_log_payload(args, train_source_dir, save_dir, optimizer_info, loss_fn.__class__.__name__, history, poison_metadata, {"train_first_batch": train_batch_stats, "test_first_batch": test_batch_stats, "initial_forward": initial_forward, "failure_details": exc.details}, best_epoch, best_test_acc, best_ckpt_path, best_ckpt_save_method, status="failed_due_to_numerical_issue", error_stage=exc.stage, error_message=str(exc)))
            raise
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc) if train_acc is not None else None,
            "test_acc": float(test_acc) if test_acc is not None else None,
        })
        train_acc_text = f"{train_acc:.4f}" if train_acc is not None else "skipped"
        test_acc_text = f"{test_acc:.4f}" if test_acc is not None else "skipped"
        print(f"[Epoch {epoch}/{args.epochs}] train loss: {train_loss:.6f} | train acc: {train_acc_text} | test acc: {test_acc_text}")
        if test_acc is not None and test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_ckpt_path = make_best_epoch_ckpt_path(save_dir, epoch)
            best_ckpt_save_method = save_checkpoint_with_fallback(network, best_ckpt_path)
            print(f"[信息] best checkpoint 已更新：{best_ckpt_path}")
        save_log(log_path, build_log_payload(args, train_source_dir, save_dir, optimizer_info, loss_fn.__class__.__name__, history, poison_metadata, {"train_first_batch": train_batch_stats, "test_first_batch": test_batch_stats, "initial_forward": initial_forward}, best_epoch, best_test_acc, best_ckpt_path, best_ckpt_save_method, status="completed" if epoch == args.epochs else "running"))
    print("=" * 72)
    print("训练完成")
    print("=" * 72)
    print(f"[结果] train loss: {history[-1]['train_loss']:.6f}")
    final_train_acc = history[-1]["train_acc"]
    final_test_acc = history[-1]["test_acc"]
    print(f"[结果] train acc: {final_train_acc:.4f}" if final_train_acc is not None else "[结果] train acc: skipped")
    print(f"[结果] test acc: {final_test_acc:.4f}" if final_test_acc is not None else "[结果] test acc: skipped")
    print(f"[结果] best checkpoint: {best_ckpt_path}")
    print(f"[结果] 训练日志: {log_path}")
    return {
        "task": args.task,
        "ms_mode": args.ms_mode,
        "optimizer": optimizer_info,
        "loss_name": loss_fn.__class__.__name__,
        "lr": float(args.lr),
        "grad_clip": float(args.grad_clip),
        "server_fast": args.server_fast,
        "eval_interval": args.eval_interval,
        "eval_train_acc": args.eval_train_acc,
        "trigger_type": args.trigger_type,
        "best_epoch": best_epoch,
        "best_test_acc": float(best_test_acc),
        "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
        "best_ckpt_save_method": best_ckpt_save_method,
        "log_path": str(log_path),
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mindspore as ms
import mindspore.context as context
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_folder_samples
from src.poison import SUPPORTED_TRIGGER_TYPES, add_trigger
from src.utils import ensure_dir, save_json, set_seed, timestamp
import src.model as model_module


def parse_args() -> argparse.Namespace:
    """解析评估参数。"""

    parser = argparse.ArgumentParser(description="评估 GTSRB clean accuracy 与 ASR")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "src" / "data" / "test", help="测试集目录")
    parser.add_argument("--ckpt-path", type=Path, required=True, help="待评估 checkpoint 路径")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "src" / "outputs", help="评估结果输出目录")
    parser.add_argument("--batch-size", type=int, default=32, help="评估 batch 大小")
    parser.add_argument("--num-classes", type=int, default=43, help="类别数")
    parser.add_argument("--device-target", default="CPU", help="设备类型")
    parser.add_argument("--ms-mode", default="PYNATIVE", choices=["GRAPH", "PYNATIVE"], help="MindSpore 执行模式")
    parser.add_argument("--image-size", type=int, default=48, help="图像尺寸")
    parser.add_argument("--target-label", type=int, default=0, help="后门目标标签")
    parser.add_argument("--poison-rate", type=float, default=0.1, help="训练时投毒比例，仅用于结果记录")
    parser.add_argument("--trigger-size", type=int, default=4, help="触发器大小")
    parser.add_argument("--alpha", type=float, default=1.0, help="触发器透明度")
    parser.add_argument("--trigger-type", default="square", choices=SUPPORTED_TRIGGER_TYPES, help="触发器类型")
    parser.add_argument("--position", default="bottom_right", choices=["bottom_right", "top_left"], help="触发器位置")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--norm-type", default="group", choices=getattr(model_module, "SUPPORTED_NORM_TYPES", ("group", "batch", "none")), help="模型归一化类型")
    return parser.parse_args()


def configure_context(device_target: str, ms_mode: str) -> None:
    """配置 MindSpore 上下文。"""

    runtime_mode = context.GRAPH_MODE if ms_mode.upper() == "GRAPH" else context.PYNATIVE_MODE
    context.set_context(mode=runtime_mode, device_target=device_target)


def build_network(num_classes: int, norm_type: str):
    """优先使用 create_model，若不存在则回退到 SimpleCNN。"""

    create_model_fn = getattr(model_module, "create_model", None)
    if callable(create_model_fn):
        for kwargs in (
            {"num_classes": num_classes, "norm_type": norm_type},
            {"class_num": num_classes, "norm_type": norm_type},
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
    for kwargs in ({"num_classes": num_classes, "norm_type": norm_type}, {"num_classes": num_classes}, {}):
        try:
            return simple_cnn_cls(**kwargs)
        except TypeError:
            continue
    raise TypeError("无法使用当前参数创建模型")


def predict_in_batches(network, images: np.ndarray, batch_size: int) -> np.ndarray:
    """分批推理，避免一次性送入过多样本。"""

    predictions: list[np.ndarray] = []
    network.set_train(False)

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        logits = network(ms.Tensor(batch, ms.float32))
        predictions.append(logits.asnumpy().argmax(axis=1).astype(np.int32))

    if not predictions:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(predictions).astype(np.int32)


def normalize_for_model(images: np.ndarray) -> np.ndarray:
    """把 [0, 1] 图像范围转换为训练时使用的 [-1, 1]。"""

    return images.astype(np.float32) * 2.0 - 1.0


def compute_accuracy(network, images: np.ndarray, labels: np.ndarray, batch_size: int) -> tuple[float, int]:
    """计算 clean accuracy。"""

    predictions = predict_in_batches(network, normalize_for_model(images), batch_size=batch_size)
    if len(predictions) == 0:
        return 0.0, 0
    accuracy = float((predictions == labels.astype(np.int32)).mean())
    return accuracy, int(len(predictions))


def build_triggered_test_images(
    images: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    trigger_size: int,
    alpha: float,
    trigger_type: str,
    position: str,
) -> tuple[np.ndarray, np.ndarray]:
    """把所有非目标类测试样本都加上触发器，用于 ASR 评估。"""

    candidate_mask = labels.astype(np.int32) != int(target_label)
    candidate_images = images[candidate_mask]
    if len(candidate_images) == 0:
        return np.empty((0, 3, images.shape[2], images.shape[3]), dtype=np.float32), np.empty((0,), dtype=np.int32)

    poisoned_images = []
    for image in candidate_images:
        poisoned_images.append(
            add_trigger(
                image=image,
                trigger_size=trigger_size,
                alpha=alpha,
                trigger_type=trigger_type,
                position=position,
            )
        )

    poisoned_labels = np.full((len(poisoned_images),), int(target_label), dtype=np.int32)
    return np.asarray(poisoned_images, dtype=np.float32), poisoned_labels


def compute_asr(
    network,
    images: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    trigger_size: int,
    alpha: float,
    trigger_type: str,
    position: str,
    batch_size: int,
) -> tuple[float, int]:
    """计算 Attack Success Rate。"""

    poisoned_images, poisoned_labels = build_triggered_test_images(
        images=images,
        labels=labels,
        target_label=target_label,
        trigger_size=trigger_size,
        alpha=alpha,
        trigger_type=trigger_type,
        position=position,
    )

    if len(poisoned_labels) == 0:
        return 0.0, 0

    predictions = predict_in_batches(network, normalize_for_model(poisoned_images), batch_size=batch_size)
    asr = float((predictions == poisoned_labels).mean()) if len(predictions) > 0 else 0.0
    return asr, int(len(predictions))


def validate_args(args: argparse.Namespace) -> None:
    """检查评估参数是否合法。"""

    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{args.ckpt_path}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在：{args.test_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    if args.num_classes <= 0:
        raise ValueError("num-classes 必须大于 0")
    if args.image_size <= 0:
        raise ValueError("image-size 必须大于 0")
    if args.trigger_size <= 0:
        raise ValueError("trigger-size 必须大于 0")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")
    if not 0.0 <= args.poison_rate <= 1.0:
        raise ValueError("poison-rate 必须位于 [0, 1]")
    if not 0 <= args.target_label < args.num_classes:
        raise ValueError(f"target-label 超出范围：{args.target_label}")


def main() -> dict[str, object]:
    """执行评估流程并保存结果。"""

    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    configure_context(device_target=args.device_target, ms_mode=args.ms_mode)

    network = build_network(args.num_classes, args.norm_type)
    param_dict = ms.load_checkpoint(str(args.ckpt_path))
    ms.load_param_into_net(network, param_dict)
    network.set_train(False)

    images, labels = load_folder_samples(
        data_dir=args.test_dir,
        image_size=(args.image_size, args.image_size),
    )
    if len(labels) == 0:
        raise ValueError(f"测试集为空，无法评估：{args.test_dir}")

    clean_accuracy, clean_sample_count = compute_accuracy(
        network=network,
        images=images,
        labels=labels,
        batch_size=args.batch_size,
    )
    asr, attack_sample_count = compute_asr(
        network=network,
        images=images,
        labels=labels,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        alpha=args.alpha,
        trigger_type=args.trigger_type,
        position=args.position,
        batch_size=args.batch_size,
    )

    metrics_dir = ensure_dir(args.output_dir / "metrics")
    summary = {
        "checkpoint_path": str(args.ckpt_path),
        "test_dir": str(args.test_dir),
        "clean_accuracy": clean_accuracy,
        "clean_sample_count": clean_sample_count,
        "asr": asr,
        "attack_sample_count": attack_sample_count,
        "target_label": int(args.target_label),
        "poison_rate": float(args.poison_rate),
        "trigger_size": int(args.trigger_size),
        "alpha": float(args.alpha),
        "trigger_type": args.trigger_type,
        "position": args.position,
        "norm_type": args.norm_type,
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "device_target": args.device_target,
        "ms_mode": args.ms_mode,
    }

    summary_path = metrics_dir / f"eval_summary_{timestamp()}.json"
    save_json(summary, summary_path)

    print(f"[结果] Clean Accuracy: {clean_accuracy:.4f}")
    print(f"[结果] ASR: {asr:.4f}")
    print(f"[结果] 攻击测试样本数: {attack_sample_count}")
    print(f"[信息] 评估结果已保存到: {summary_path}")
    return summary


if __name__ == "__main__":
    main()

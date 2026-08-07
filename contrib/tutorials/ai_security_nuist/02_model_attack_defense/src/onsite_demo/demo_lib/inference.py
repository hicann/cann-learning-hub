from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .paths import find_checkpoint, resolve_project_root
from .runtime import configure_mindspore_device


DEFAULT_IMAGE_SIZE = (48, 48)


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover
    RESAMPLE_BILINEAR = Image.BILINEAR


def _ensure_project_on_path() -> Path:
    paths = resolve_project_root()
    project_root = paths["PROJECT_ROOT"]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def _build_network(num_classes: int, norm_type: str, pool_type: str | None = None):
    _ensure_project_on_path()
    import src.model as model_module

    normalized_pool_type = None if pool_type is None else str(pool_type).strip().lower()
    create_model_fn = getattr(model_module, "create_model", None)
    if callable(create_model_fn):
        kwargs_options = []
        if normalized_pool_type:
            kwargs_options.extend(
                [
                    {"num_classes": num_classes, "norm_type": norm_type, "pool_type": normalized_pool_type},
                    {"class_num": num_classes, "norm_type": norm_type, "pool_type": normalized_pool_type},
                ]
            )
        kwargs_options.extend(
            [
                {"num_classes": num_classes, "norm_type": norm_type},
                {"class_num": num_classes, "norm_type": norm_type},
                {"num_classes": num_classes},
                {"class_num": num_classes},
                {},
            ]
        )
        for kwargs in kwargs_options:
            try:
                return create_model_fn(**kwargs)
            except TypeError:
                continue

    simple_cnn_cls = getattr(model_module, "SimpleCNN", None)
    if simple_cnn_cls is None:
        raise AttributeError("src/model.py 中没有 create_model(...) 或 SimpleCNN，无法创建模型。")
    kwargs_options = []
    if normalized_pool_type:
        kwargs_options.append({"num_classes": num_classes, "norm_type": norm_type, "pool_type": normalized_pool_type})
    kwargs_options.extend(({"num_classes": num_classes, "norm_type": norm_type}, {"num_classes": num_classes}, {}))
    for kwargs in kwargs_options:
        try:
            return simple_cnn_cls(**kwargs)
        except TypeError:
            continue
    raise TypeError("无法使用当前参数创建模型，请检查 src/model.py 的模型构造函数。")


def load_model_from_checkpoint(
    experiment_name: str,
    checkpoint_path: str | Path | None,
    num_classes: int,
    norm_type: str,
    device_target: str,
    ms_mode: str,
    allow_cpu_fallback: bool = True,
    pool_type: str | None = None,
):
    """Load the official model checkpoint for onsite inference/evaluation."""

    import mindspore as ms

    actual_device = configure_mindspore_device(
        preferred=device_target,
        ms_mode=ms_mode,
        allow_cpu_fallback=bool(allow_cpu_fallback),
    )
    ckpt_path = find_checkpoint(experiment_name=experiment_name, checkpoint_path=checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"正式 checkpoint 不存在：{ckpt_path}\n"
            "修复建议：请上传服务器完整训练得到的 ckpt，或修正 demo_config/final_summary 中的路径。"
        )

    network = _build_network(
        num_classes=int(num_classes),
        norm_type=str(norm_type),
        pool_type=pool_type,
    )
    param_dict = ms.load_checkpoint(str(ckpt_path))
    ms.load_param_into_net(network, param_dict)
    network.set_train(False)
    setattr(network, "demo_checkpoint_path", str(ckpt_path))
    setattr(network, "demo_checkpoint_loaded", True)
    setattr(network, "demo_experiment_name", experiment_name)
    setattr(network, "demo_device_target", actual_device)
    setattr(network, "demo_ms_mode", str(ms_mode).upper().strip())
    return network


def load_image_chw_01(image_path: str | Path, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), RESAMPLE_BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] in (1, 3):
        return np.transpose(data, (1, 2, 0))
    return data


def normalize_for_model(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) * 2.0 - 1.0


def apply_optional_trigger(
    image_chw_01: np.ndarray,
    trigger_type: str | None = None,
    trigger_size: int = 4,
    alpha: float = 0.8,
    position: str = "bottom_right",
) -> np.ndarray:
    if trigger_type is None:
        return image_chw_01.astype(np.float32).copy()
    normalized = str(trigger_type).lower().strip()
    if normalized in {"", "none", "clean"}:
        return image_chw_01.astype(np.float32).copy()

    _ensure_project_on_path()
    from src.poison import add_trigger

    return add_trigger(
        image=image_chw_01,
        trigger_size=int(trigger_size),
        alpha=float(alpha),
        position=position,
        trigger_type=normalized,
    ).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    return (exp_values / np.sum(exp_values, axis=1, keepdims=True)).astype(np.float32)


def predict_probabilities(model, images_chw_01: np.ndarray, batch_size: int = 32) -> np.ndarray:
    import mindspore as ms

    data = np.asarray(images_chw_01, dtype=np.float32)
    if data.ndim == 3:
        data = data[None, ...]

    model.set_train(False)
    probabilities: list[np.ndarray] = []
    for start in range(0, len(data), int(batch_size)):
        batch = normalize_for_model(data[start : start + int(batch_size)])
        logits = model(ms.Tensor(batch, ms.float32))
        probabilities.append(_softmax(logits.asnumpy()))
    if not probabilities:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(probabilities, axis=0)


def predict_one_image(
    model,
    image_path: str | Path,
    trigger_type: str | None = None,
    target_label: int = 0,
    trigger_size: int = 4,
    alpha: float = 0.8,
    position: str = "bottom_right",
) -> dict[str, Any]:
    """Run clean or triggered single-image inference for classroom display."""

    clean_image = load_image_chw_01(image_path)
    processed = apply_optional_trigger(
        clean_image,
        trigger_type=trigger_type,
        trigger_size=trigger_size,
        alpha=alpha,
        position=position,
    )
    probs = predict_probabilities(model, processed, batch_size=1)[0]
    top_indices = np.argsort(probs)[::-1][:5]
    pred_label = int(top_indices[0])
    return {
        "image_path": str(image_path),
        "trigger_type": trigger_type,
        "target_label": int(target_label),
        "pred_label": pred_label,
        "probability": float(probs[pred_label]),
        "target_probability": float(probs[int(target_label)]) if int(target_label) < len(probs) else None,
        "top5": [
            {"label": int(index), "probability": float(probs[index])}
            for index in top_indices
        ],
        "processed_image": chw_to_hwc(processed),
    }

#!/usr/bin/env python3
"""Benchmark YOLO postprocess on CPU and optional custom NPU backend."""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from config_utils import load_config
except ImportError:
    from .config_utils import load_config


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-7)


def cpu_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_det: int) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        idx = order[0]
        keep.append(idx)
        if order.size == 1:
            break
        ious = box_iou_one_to_many(boxes[idx], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess_cpu(pred: np.ndarray, cfg: dict) -> np.ndarray:
    pp = cfg["postprocess"]
    raw = np.asarray(pred, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError(f"expected prediction shape [N, 5 + classes], got {pred.shape}")

    boxes = xywh_to_xyxy(raw[:, :4])
    obj = raw[:, 4]
    cls_scores = raw[:, 5:]
    class_ids = cls_scores.argmax(axis=1)
    scores = obj * cls_scores[np.arange(raw.shape[0]), class_ids]
    mask = scores >= float(pp["score_threshold"])
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
    if boxes.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    max_det = int(pp["max_detections"])
    if bool(pp.get("class_agnostic", False)):
        keep = cpu_nms(boxes, scores, float(pp["nms_iou_threshold"]), max_det)
    else:
        keep_parts = []
        for cls_id in np.unique(class_ids):
            cls_indices = np.flatnonzero(class_ids == cls_id)
            cls_keep = cpu_nms(
                boxes[cls_indices],
                scores[cls_indices],
                float(pp["nms_iou_threshold"]),
                max_det,
            )
            keep_parts.extend(cls_indices[cls_keep].tolist())
        keep = np.asarray(keep_parts, dtype=np.int64)
        if keep.size > 0:
            keep = keep[np.argsort(scores[keep])[::-1]][:max_det]

    result = np.concatenate([boxes[keep], scores[keep, None], class_ids[keep, None]], axis=1)
    return result.astype(np.float32)


def postprocess_npu_optional(pred: np.ndarray, cfg: dict) -> np.ndarray:
    if not bool(cfg.get("custom_op", {}).get("enabled", False)):
        return postprocess_cpu(pred, cfg)
    try:
        import yolo_nms_custom

        return yolo_nms_custom.nms(pred, cfg["postprocess"])
    except Exception:
        return postprocess_cpu(pred, cfg)


def random_prediction(cfg: dict) -> np.ndarray:
    seed = int(cfg["benchmark"]["random_seed"])
    shape = tuple(int(x) for x in cfg["model"]["output_shape"])
    rng = np.random.default_rng(seed)
    pred = rng.random(shape, dtype=np.float32)
    pred[..., :4] *= float(cfg["preprocess"]["image_size"])
    return pred


def timeit(fn, repeat: int, warmup: int) -> tuple[float, np.ndarray]:
    result = None
    for _ in range(warmup):
        result = fn()
    start = time.perf_counter()
    for _ in range(repeat):
        result = fn()
    elapsed = (time.perf_counter() - start) * 1000.0 / repeat
    return elapsed, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/yolo_edge.yaml")
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    repeat = args.repeat or int(cfg["benchmark"]["repeat"])
    warmup = args.warmup or int(cfg["benchmark"]["warmup"])
    pred = random_prediction(cfg)

    cpu_ms, cpu_result = timeit(lambda: postprocess_cpu(pred, cfg), repeat, warmup)
    custom_enabled = bool(cfg.get("custom_op", {}).get("enabled", False))
    npu_ms, npu_result = (None, None)
    if custom_enabled:
        npu_ms, npu_result = timeit(lambda: postprocess_npu_optional(pred, cfg), repeat, warmup)

    print("=" * 72)
    print("YOLO postprocess benchmark")
    print("=" * 72)
    print(f"prediction shape: {pred.shape}")
    print(f"CPU postprocess: {cpu_ms:.3f} ms, detections={len(cpu_result)}")
    if custom_enabled and npu_ms is not None and npu_result is not None:
        print(f"NPU/custom path:  {npu_ms:.3f} ms, detections={len(npu_result)}")
        print(f"speedup:          {cpu_ms / npu_ms:.2f}x")
        print("Note: if custom op is not installed, this path falls back to CPU for functional parity.")
    else:
        print("NPU/custom path:  disabled in config; set custom_op.enabled=true after deploying the custom op.")
        print("speedup:          not reported for CPU fallback.")


if __name__ == "__main__":
    main()

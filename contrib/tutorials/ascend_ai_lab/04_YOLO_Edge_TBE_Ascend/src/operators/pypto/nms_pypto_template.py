"""PyPTO-style Python template for YOLO NMS operator prototyping.

The exact PyPTO API may differ by course image and CANN version. Keep this file
as a design notebook companion: start from this Python-level algorithm, verify
correctness, then lower the hotspot to a target-specific custom operator.
"""

from __future__ import annotations

import numpy as np


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-7)


def reference_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_output: int) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_output:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        ious = iou_one_to_many(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int32)


def pypto_lowering_plan() -> list[str]:
    return [
        "Keep boxes/scores contiguous and aligned.",
        "Move IoU block computation to vector-friendly device code.",
        "Use a suppression mask instead of Python while-loop on the target path.",
        "Return keep indices and valid count to PyACL postprocess pipeline.",
    ]


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    boxes = rng.random((16, 4), dtype=np.float32)
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] * 0.2
    scores = rng.random(16, dtype=np.float32)
    print(reference_nms(boxes, scores, 0.45, 10))
    print("\n".join(pypto_lowering_plan()))

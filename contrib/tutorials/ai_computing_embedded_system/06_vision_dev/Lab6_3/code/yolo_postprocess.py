# -*- coding: utf-8 -*-
"""
yolo_postprocess.py - YOLOv8 后处理工具模块
============================================================
功能:
  1. letterbox 预处理 (保持宽高比缩放 + 灰色填充)
  2. YOLOv8 输出解码 (anchor-free 解码)
  3. NMS 非极大值抑制
  4. 检测框坐标还原到原图
  5. 检测结果可视化绘制

在昇腾香橙派上，该模块被以下脚本调用:
  - yolo_image_detect.py       (DVPP+AIPP 图片检测)
  - yolo_image_detect_opencv.py (OpenCV 图片检测)
  - yolo_video_detect.py       (DVPP+AIPP 视频检测)
  - yolo_video_detect_opencv.py (OpenCV 视频检测)
  - yolo_usb_camera_detect.py  (USB 摄像头实时检测)
"""

import numpy as np
import cv2

# COCO 80 类别名称
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# 不同类别使用不同颜色 (BGR 格式)
_COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    letterbox 预处理: 保持宽高比缩放，多余部分用灰色填充
    这是 YOLO 系列标准预处理方式

    参数:
        img: 输入图像 (BGR, HWC)
        new_shape: 目标尺寸 (w, h)
        color: 填充颜色

    返回:
        img640: letterbox 后的 640x640 图像
        ratio: 缩放比例
        (dw, dh): 宽高填充量
    """
    h, w = img.shape[:2]
    target_w, target_h = new_shape
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 创建目标画布并居中放置
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, ratio, (dw, dh)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def yolov8_decode_raw(outputs, conf_thres=0.25, iou_thres=0.45,
                       num_classes=80, reg_max=16, top_k=1000):
    """
    YOLOv8 原始 9 输出后处理 (含 DFL 解码)

    模型输出 3 个尺度, 每个尺度 3 个输出:
      outputs[3*i+0]: [1, 4*reg_max, H, W]  框回归 (DFL 原始)
      outputs[3*i+1]: [1, num_classes, H, W] 类别分数 (已 sigmoid)
      outputs[3*i+2]: [1, 1, H, W]          忽略

    strides = [8, 16, 32] (对应 80x80, 40x40, 20x20)

    返回: detections list of [x1, y1, x2, y2, score, class_id]
    """
    strides = [8, 16, 32]
    all_boxes = []
    all_scores = []
    all_class_ids = []

    for i, stride in enumerate(strides):
        if 3 * i + 1 >= len(outputs):
            break
        box_raw = outputs[3 * i]       # [1, 64, H, W]
        cls = outputs[3 * i + 1]      # [1, 80, H, W]

        _, C, H, W = box_raw.shape

        # DFL 解码: [1, 64, H, W] -> [1, 4, 16, H, W] -> softmax(dim=2) -> 加权求和 -> [1, 4, H, W]
        box = box_raw.reshape(1, 4, reg_max, H, W)
        box = box - box.max(axis=2, keepdims=True)
        box = np.exp(box)
        box = box / box.sum(axis=2, keepdims=True)
        proj = np.arange(reg_max, dtype=np.float32)
        box = (box * proj[None, None, :, None, None]).sum(axis=2)  # [1, 4, H, W]

        # 生成网格坐标
        grid_y, grid_x = np.meshgrid(np.arange(H, dtype=np.float32),
                                     np.arange(W, dtype=np.float32),
                                     indexing='ij')

        # 解码为 xyxy
        x1 = (grid_x - box[0, 0]) * stride
        y1 = (grid_y - box[0, 1]) * stride
        x2 = (grid_x + box[0, 2]) * stride
        y2 = (grid_y + box[0, 3]) * stride

        # 类别分数: [1, 80, H, W] -> [H*W, 80]
        cls_flat = cls[0].transpose(1, 2, 0).reshape(H * W, num_classes)
        class_ids = np.argmax(cls_flat, axis=1)
        max_scores = np.max(cls_flat, axis=1)

        # 置信度过滤
        mask = max_scores > conf_thres
        if not mask.any():
            continue

        xyxy = np.stack([x1.ravel()[mask], y1.ravel()[mask],
                         x2.ravel()[mask], y2.ravel()[mask]], axis=1)
        all_boxes.append(xyxy)
        all_scores.append(max_scores[mask])
        all_class_ids.append(class_ids[mask])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_class_ids)

    print(f"  [debug] 过阈值({conf_thres}): {len(boxes)}", flush=True)

    # top-k 截断
    if len(boxes) > top_k:
        top_idx = np.argsort(scores)[::-1][:top_k]
        boxes = boxes[top_idx]
        scores = scores[top_idx]
        class_ids = class_ids[top_idx]
        print(f"  [debug] top-{top_k} 截断后: {len(boxes)}", flush=True)

    # NMS
    detections = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_thres)
        for idx in keep:
            detections.append([
                cls_boxes[idx, 0], cls_boxes[idx, 1],
                cls_boxes[idx, 2], cls_boxes[idx, 3],
                cls_scores[idx], int(cls_id)
            ])
    return detections


def yolov8_decode(output, conf_thres=0.25, iou_thres=0.45, num_classes=80, apply_sigmoid=False, top_k=1000):
    """
    YOLOv8 输出后处理: 解码检测框 + NMS

    支持的输出格式:
      - [1, 84, 8400] (cx, cy, w, h, 80类置信度) 标准 YOLOv8
      - [1, C, H, W]  (cx, cy, w, h, 类别置信度) 4D 空间输出
      - 经转置后: [1, 8400, 84]

    参数:
        output: 模型输出 numpy 数组
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值
        num_classes: 类别数
        apply_sigmoid: 对类别置信度施加 sigmoid (模型未内置 sigmoid 时使用)
        top_k: NMS 前最大候选框数量 (防止 O(N²) NMS 卡死)

    返回:
        detections: list of [x1, y1, x2, y2, score, class_id]
    """
    # 处理 4D 输出 [1, C, H, W] -> [1, C, H*W]
    if output.ndim == 4:
        output = output.reshape(output.shape[0], output.shape[1], -1)
    # 统一输出形状到 (N, 4+num_classes)
    if output.ndim == 3:
        output = output[0]
    expected_c = 4 + num_classes
    if output.shape[0] == expected_c and output.shape[1] != expected_c:
        output = output.T  # (C, N) -> (N, C)

    # 前 4 列是 cx, cy, w, h; 后 num_classes 列是类别置信度
    boxes = output[:, :4]
    scores = output[:, 4:4 + num_classes]

    if apply_sigmoid:
        print(f"  [debug] sigmoid前 score: min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}", flush=True)
        scores = sigmoid(scores)

    # 取每个检测框的最大类别置信度
    class_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)

    # 置信度过滤
    mask = max_scores > conf_thres
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    print(f"  [debug] 总网格点: {output.shape[0]}, 过阈值({conf_thres}): {len(boxes)}", flush=True)

    if len(boxes) == 0:
        return []

    # top-k 截断: 按 score 降序取前 top_k 个, 防止 NMS 过慢
    if len(boxes) > top_k:
        top_idx = np.argsort(max_scores)[::-1][:top_k]
        boxes = boxes[top_idx]
        max_scores = max_scores[top_idx]
        class_ids = class_ids[top_idx]
        print(f"  [debug] top-{top_k} 截断后: {len(boxes)}", flush=True)

    # cxcywh -> xyxy
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # NMS (按类别分别做)
    detections = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = xyxy[cls_mask]
        cls_scores = max_scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_thres)
        for idx in keep:
            detections.append([
                cls_boxes[idx, 0], cls_boxes[idx, 1],
                cls_boxes[idx, 2], cls_boxes[idx, 3],
                cls_scores[idx], int(cls_id)
            ])
    return detections


def nms(boxes, scores, iou_thres):
    """
    非极大值抑制 (NMS): 去除重叠的冗余检测框

    参数:
        boxes: (N, 4) xyxy 格式
        scores: (N,) 置信度
        iou_thres: IoU 阈值

    返回:
        keep: 保留的索引列表
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thres]
    return keep


def scale_boxes(detections, ratio, dw, dh, orig_w, orig_h):
    """
    将 640x640 坐标系下的检测框还原到原图坐标系

    参数:
        detections: yolov8_decode 输出
        ratio: letterbox 缩放比例
        dw, dh: letterbox 填充量
        orig_w, orig_h: 原图宽高

    返回:
        scaled: list of [x1, y1, x2, y2, score, class_id] (原图坐标)
    """
    scaled = []
    for det in detections:
        x1 = (det[0] - dw) / ratio
        y1 = (det[1] - dh) / ratio
        x2 = (det[2] - dw) / ratio
        y2 = (det[3] - dh) / ratio
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))
        scaled.append([int(x1), int(y1), int(x2), int(y2), det[4], det[5]])
    return scaled


def draw_detections(img, detections, class_names=COCO_NAMES, line_width=2):
    """
    在图像上绘制检测结果 (检测框 + 类别标签 + 置信度)

    参数:
        img: 输入图像 (BGR)
        detections: list of [x1, y1, x2, y2, score, class_id]
        class_names: 类别名称列表
        line_width: 框线宽度

    返回:
        img: 绘制后的图像
    """
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        color = tuple(int(c) for c in _COLORS[cls_id % 80])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        label = f'{class_names[cls_id]} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def align_up(size, align):
    """DVPP 对齐函数: 向上取整到 align 的倍数"""
    return (size + align - 1) // align * align

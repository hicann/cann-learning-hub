# -*- coding: utf-8 -*-
"""
yolo_video_detect_opencv.py - 昇腾香橙派视频 YOLO 目标检测 (OpenCV 纯 CPU 路径)
============================================================================
运行环境: 昇腾香橙派 AIPro (Ascend 310B4 NPU)
依赖: CANN Toolkit, AscendCL (acl), OpenCV, NumPy

功能:
  1. 使用 OpenCV VideoCapture 读取视频帧
  2. 逐帧使用 OpenCV letterbox 缩放 (CPU)
  3. 使用纯 OM 模型推理 (float32 输入, 无 AIPP)
  4. YOLOv8 后处理 (解码 + NMS)
  5. 绘制检测结果，合成结果视频保存到 output/

用途: 与 yolo_video_detect.py (DVPP+AIPP) 对比，验证硬件加速效果

用法:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 yolo_video_detect_opencv.py [视频路径]

  默认视频: ../images/dog.mp4
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_postprocess import (
    letterbox, yolov8_decode, yolov8_decode_raw, scale_boxes, draw_detections, COCO_NAMES
)

try:
    import acl
except ImportError:
    print("[ERROR] 无法导入 acl 模块，请先加载 CANN 环境:")
    print("  source /usr/local/Ascend/ascend-toolkit/set_env.sh")
    sys.exit(1)

ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

MODEL_PATH = "../output/yolov8n_pure.om"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "../images/dog.mp4"
OUTPUT_DIR = "../output"
INPUT_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
NUM_CLASSES = 80


class OpenCVVideoDetector:
    """
    OpenCV + 纯 OM 视频 YOLO 检测器 (无 DVPP, 无 AIPP)

    逐帧处理流程:
      视频帧(BGR) -> letterbox缩放(CPU) -> BGR->RGB(CPU) -> 归一化/255(CPU)
      -> HWC->CHW(CPU) -> float32 -> 纯OM推理 -> YOLO后处理 -> 绘制结果

    对比说明:
      - 预处理全在 CPU, 输入为 float32 (4字节)
      - vs DVPP+AIPP: 预处理在 NPU 硬件, 输入为 uint8 (1字节)
    """

    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path
        self._init_acl()
        self._load_model()

    def _init_acl(self):
        ret = acl.init()
        assert ret in (0, 100002)
        acl.rt.set_device(self.device_id)
        self.context, _ = acl.rt.create_context(self.device_id)
        self.stream, _ = acl.rt.create_stream()
        print(f"[OK] ACL 初始化成功")

    def _load_model(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        assert ret == 0
        self.model_desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.model_desc, self.model_id)
        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.num_outputs = acl.mdl.get_num_outputs(self.model_desc)
        self.output_sizes = []
        self.output_dims_list = []
        for i in range(self.num_outputs):
            sz = acl.mdl.get_output_size_by_index(self.model_desc, i)
            self.output_sizes.append(sz)
            dr = acl.mdl.get_output_dims(self.model_desc, i)
            if isinstance(dr, tuple) and isinstance(dr[0], dict):
                dims = tuple(dr[0]['dims'])
            elif isinstance(dr, dict):
                dims = tuple(dr['dims'])
            else:
                dims = tuple(dr)
                if dims[0] == len(dims) - 1:
                    dims = dims[1:]
            self.output_dims_list.append(dims)
        self.input_dev, _ = acl.rt.malloc(self.input_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        self.in_dataset = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(self.in_dataset,
            acl.create_data_buffer(self.input_dev, self.input_size))
        self.output_devs = []
        self.out_bufs = []
        self.out_dataset = acl.mdl.create_dataset()
        for i in range(self.num_outputs):
            dev, _ = acl.rt.malloc(self.output_sizes[i], ACL_MEM_MALLOC_NORMAL_ONLY)
            buf = acl.create_data_buffer(dev, self.output_sizes[i])
            acl.mdl.add_dataset_buffer(self.out_dataset, buf)
            self.output_devs.append(dev)
            self.out_bufs.append(buf)
        print(f"[OK] 纯 OM 模型加载成功: {self.model_path}")

    def detect_frame(self, frame):
        """OpenCV 预处理 + 纯 OM 推理"""
        orig_h, orig_w = frame.shape[:2]

        # letterbox (CPU)
        img640, ratio, (dw, dh) = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))

        # BGR->RGB + 归一化 + HWC->CHW + float32 (全 CPU)
        rgb = cv2.cvtColor(img640, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = normalized.transpose(2, 0, 1).reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
        input_data = np.ascontiguousarray(chw)

        # 纯 OM 推理
        acl.rt.memcpy(self.input_dev, self.input_size,
                      input_data.ctypes.data, self.input_size,
                      ACL_MEMCPY_HOST_TO_DEVICE)
        ret = acl.mdl.execute(self.model_id, self.in_dataset, self.out_dataset)
        assert ret == 0
        acl.rt.synchronize_stream(self.stream)

        # 后处理 (读取所有输出)
        all_outputs = []
        for i in range(self.num_outputs):
            out_np = np.zeros(self.output_sizes[i], dtype=np.uint8)
            acl.rt.memcpy(out_np.ctypes.data, self.output_sizes[i],
                          self.output_devs[i], self.output_sizes[i],
                          ACL_MEMCPY_DEVICE_TO_HOST)
            out_data = out_np.view(np.float32)
            out_count = 1
            for d in self.output_dims_list[i]:
                out_count *= d
            out_data = out_data[:out_count].reshape(self.output_dims_list[i])
            all_outputs.append(out_data)
        detections = yolov8_decode_raw(all_outputs, CONF_THRES, IOU_THRES)
        scaled_dets = scale_boxes(detections, ratio, dw, dh, orig_w, orig_h)
        return scaled_dets

    def cleanup(self):
        for buf in self.out_bufs:
            acl.destroy_data_buffer(buf)
        acl.mdl.destroy_dataset(self.in_dataset)
        acl.mdl.destroy_dataset(self.out_dataset)
        acl.rt.free(self.input_dev)
        for dev in self.output_devs:
            acl.rt.free(dev)
        acl.mdl.destroy_desc(self.model_desc)
        acl.mdl.unload(self.model_id)
        acl.rt.destroy_stream(self.stream)
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  昇腾香橙派视频 YOLO 目标检测 (OpenCV 纯 CPU 路径)")
    print("  硬件: Ascend 310B4 NPU (仅推理用 NPU, 预处理全 CPU)")
    print("=" * 60)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {VIDEO_PATH}")
    print(f"  帧数: {total}, 分辨率: {orig_w}x{orig_h}, 帧率: {fps:.1f} fps")

    detector = OpenCVVideoDetector(MODEL_PATH)

    result_path = os.path.join(OUTPUT_DIR, "opencv_video_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, fps if fps > 0 else 25,
                             (orig_w, orig_h))

    frame_idx = 0
    total_time = 0
    print("\n开始逐帧检测...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.time()
        detections = detector.detect_frame(frame)
        infer_ms = (time.time() - t0) * 1000
        total_time += infer_ms

        result_frame = draw_detections(frame, detections)
        cur_fps = 1000.0 / infer_ms if infer_ms > 0 else 0
        cv2.putText(result_frame, f"Frame {frame_idx}/{total}  {infer_ms:.1f}ms  {cur_fps:.1f}fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.write(result_frame)

        if frame_idx % 50 == 0 or frame_idx == total:
            print(f"  Frame {frame_idx}/{total}: {infer_ms:.2f} ms, "
                  f"{len(detections)} 个目标, {cur_fps:.1f} fps")

    cap.release()
    writer.release()

    avg_ms = total_time / frame_idx if frame_idx > 0 else 0
    avg_fps = frame_idx / (total_time / 1000) if total_time > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"  视频检测完成！")
    print(f"  总帧数: {frame_idx}")
    print(f"  平均耗时: {avg_ms:.2f} ms/帧")
    print(f"  平均帧率: {avg_fps:.1f} fps")
    print(f"  结果视频: {result_path} ({os.path.getsize(result_path)//1024} KB)")
    print(f"{'=' * 60}")

    detector.cleanup()


if __name__ == '__main__':
    main()

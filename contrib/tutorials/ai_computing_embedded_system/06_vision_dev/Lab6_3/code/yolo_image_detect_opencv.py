# -*- coding: utf-8 -*-
"""
yolo_image_detect_opencv.py - 昇腾香橙派图片 YOLO 目标检测 (OpenCV 纯 CPU 路径)
============================================================================
运行环境: 昇腾香橙派 AIPro (Ascend 310B4 NPU)
依赖: CANN Toolkit, AscendCL (acl), OpenCV, NumPy

功能:
  1. 使用 OpenCV 软件解码图片 (CPU)
  2. 使用 OpenCV letterbox 缩放到 640x640 (CPU)
  3. 使用纯 OM 模型推理 (float32 输入, 无 AIPP)
  4. YOLOv8 后处理 (解码 + NMS)
  5. 绘制检测结果并保存到 output/

用途: 与 yolo_image_detect.py (DVPP+AIPP) 对比，验证硬件加速效果

用法:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 yolo_image_detect_opencv.py
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

# === 常量 ===
ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

MODEL_PATH = "../output/yolov8n_pure.om"   # 纯 OM 模型 (无 AIPP)
IMAGE_DIR = "../images"
OUTPUT_DIR = "../output"
INPUT_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
NUM_CLASSES = 80

TEST_IMAGES = [
    "dog1.jpg", "dog2.jpg", "cat1.jpg", "cat2.jpg",
]


class OpenCVDetector:
    """
    OpenCV + 纯 OM YOLO 检测器 (无 DVPP, 无 AIPP)

    处理流程:
      图片 -> OpenCV解码(BGR) -> letterbox缩放 -> BGR->RGB -> 归一化/255
      -> HWC->CHW -> float32 -> 纯OM推理 -> YOLO后处理 -> 绘制结果

    对比说明:
      - 解码: OpenCV CPU (vs DVPP 硬件)
      - 缩放: OpenCV CPU (vs DVPP VPC 硬件)
      - 归一化: NumPy CPU (vs AIPP 硬件)
      - 输入类型: float32 4字节 (vs AIPP uint8 1字节, 带宽 4x)
    """

    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path
        self._init_acl()
        self._load_model()

    def _init_acl(self):
        ret = acl.init()
        assert ret in (0, 100002), f"acl.init failed: {ret}"
        ret = acl.rt.set_device(self.device_id)
        assert ret == 0
        self.context, ret = acl.rt.create_context(self.device_id)
        assert ret == 0
        self.stream, ret = acl.rt.create_stream()
        assert ret == 0
        print(f"[OK] ACL 初始化成功 (device {self.device_id})")

    def _load_model(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        assert ret == 0, f"load model failed: {ret}"
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
        self.in_buf = acl.create_data_buffer(self.input_dev, self.input_size)
        acl.mdl.add_dataset_buffer(self.in_dataset, self.in_buf)

        self.output_devs = []
        self.out_bufs = []
        self.out_dataset = acl.mdl.create_dataset()
        for i in range(self.num_outputs):
            dev, _ = acl.rt.malloc(self.output_sizes[i], ACL_MEM_MALLOC_NORMAL_ONLY)
            buf = acl.create_data_buffer(dev, self.output_sizes[i])
            acl.mdl.add_dataset_buffer(self.out_dataset, buf)
            self.output_devs.append(dev)
            self.out_bufs.append(buf)

        print(f"[OK] 模型加载成功: {self.model_path}")
        print(f"     输入大小: {self.input_size} bytes ({self.input_size/1024/1024:.2f} MB, float32)")
        print(f"     输出数: {self.num_outputs}")

    def detect(self, image_path):
        """
        OpenCV 纯 CPU 预处理 + 纯 OM 推理

        预处理全在 CPU:
          1. OpenCV imread 解码 (CPU 哈夫曼解码 + IDCT)
          2. letterbox 缩放 (CPU 双线性插值)
          3. BGR -> RGB 色彩转换 (CPU)
          4. 归一化 /255 (CPU 浮点除法)
          5. HWC -> CHW 转置 (CPU 内存重排)
          6. float32 类型转换 (CPU)
        """
        timing = {}

        # Step 1: OpenCV 解码 (CPU)
        t0 = time.time()
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        orig_h, orig_w = orig_img.shape[:2]
        timing['opencv_decode'] = (time.time() - t0) * 1000

        # Step 2: letterbox 缩放 (CPU)
        t0 = time.time()
        img640, ratio, (dw, dh) = letterbox(orig_img, (INPUT_SIZE, INPUT_SIZE))
        timing['letterbox'] = (time.time() - t0) * 1000

        # Step 3: BGR->RGB + 归一化 + HWC->CHW + float32 (全 CPU)
        t0 = time.time()
        rgb = cv2.cvtColor(img640, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = normalized.transpose(2, 0, 1).reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
        input_data = np.ascontiguousarray(chw)
        timing['preprocess'] = (time.time() - t0) * 1000

        # Step 4: 纯 OM 推理 (float32 输入)
        t0 = time.time()
        acl.rt.memcpy(self.input_dev, self.input_size,
                      input_data.ctypes.data, self.input_size,
                      ACL_MEMCPY_HOST_TO_DEVICE)
        ret = acl.mdl.execute(self.model_id, self.in_dataset, self.out_dataset)
        assert ret == 0, f"execute failed: {ret}"
        acl.rt.synchronize_stream(self.stream)
        timing['inference'] = (time.time() - t0) * 1000

        # Step 5: 后处理 (读取所有输出)
        t0 = time.time()
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
        timing['postprocess'] = (time.time() - t0) * 1000

        result_img = draw_detections(orig_img.copy(), scaled_dets)
        return result_img, scaled_dets, timing

    def cleanup(self):
        acl.destroy_data_buffer(self.in_buf)
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
        print("[OK] 所有资源已释放")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  昇腾香橙派图片 YOLO 目标检测 (OpenCV 纯 CPU 路径)")
    print("  硬件: Ascend 310B4 NPU (仅推理用 NPU, 预处理全 CPU)")
    print("=" * 60)

    detector = OpenCVDetector(MODEL_PATH)

    all_results = []
    for img_name in TEST_IMAGES:
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"[SKIP] {img_path} 不存在")
            continue

        print(f"\n--- 处理: {img_name} ---")
        result_img, detections, timing = detector.detect(img_path)

        total_ms = sum(timing.values())
        print(f"  OpenCV 解码: {timing['opencv_decode']:.2f} ms")
        print(f"  letterbox:   {timing['letterbox']:.2f} ms")
        print(f"  预处理:      {timing['preprocess']:.2f} ms")
        print(f"  推理:        {timing['inference']:.2f} ms")
        print(f"  后处理:      {timing['postprocess']:.2f} ms")
        print(f"  总耗时:      {total_ms:.2f} ms")
        print(f"  检测到 {len(detections)} 个目标:")
        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            print(f"    {COCO_NAMES[cls_id]}: {score:.2f}  [{x1},{y1},{x2},{y2}]")

        out_path = os.path.join(OUTPUT_DIR, f"opencv_{img_name}")
        cv2.imwrite(out_path, result_img)
        print(f"  结果已保存: {out_path}")
        all_results.append({'name': img_name, 'timing': timing,
                            'detections': detections, 'total': total_ms})

    print("\n" + "=" * 60)
    print("  OpenCV 图片检测汇总")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['name']}: {r['total']:.2f} ms, {len(r['detections'])} 个目标")
    if all_results:
        avg_ms = sum(r['total'] for r in all_results) / len(all_results)
        print(f"  平均耗时: {avg_ms:.2f} ms/张")
    print("[OK] OpenCV 图片检测完成！")

    detector.cleanup()


if __name__ == '__main__':
    main()

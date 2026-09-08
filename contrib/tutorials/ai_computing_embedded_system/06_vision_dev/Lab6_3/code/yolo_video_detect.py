# -*- coding: utf-8 -*-
"""
yolo_video_detect.py - 昇腾香橙派视频 YOLO 目标检测 (DVPP + AIPP 硬件加速路径)
============================================================================
运行环境: 昇腾香橙派 AIPro (Ascend 310B4 NPU)
依赖: CANN Toolkit, AscendCL (acl), OpenCV, NumPy

功能:
  1. 使用 OpenCV VideoCapture 读取视频帧
  2. 逐帧使用 DVPP 硬件解码 + VPC 缩放
  3. 使用 AIPP-OM 模型推理 (uint8 输入)
  4. YOLOv8 后处理 (解码 + NMS)
  5. 绘制检测结果，合成结果视频保存到 output/

用法:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 yolo_video_detect.py [视频路径]

  默认视频: ../images/dog.mp4
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_postprocess import (
    letterbox, yolov8_decode, yolov8_decode_raw, scale_boxes, draw_detections,
    COCO_NAMES, align_up
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
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1

MODEL_PATH = "../output/yolov8n_aipp.om"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "../images/dog.mp4"
OUTPUT_DIR = "../output"
INPUT_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
NUM_CLASSES = 80


class DVPPAIPPVideoDetector:
    """
    DVPP + AIPP 视频 YOLO 检测器

    逐帧处理流程:
      视频帧(BGR) -> 编码JPEG -> DVPP硬件解码 -> DVPP VPC缩放
      -> AIPP-OM推理 -> YOLO后处理 -> 绘制结果 -> 写入结果视频

    注意: 视频帧由 OpenCV 从 mp4 中解码得到 (BGR)，
    然后编码为 JPEG 送入 DVPP 做硬件解码+缩放。
    在实际系统中可使用昇腾硬件视频解码 VDEC 直接解码 H.264/H.265，
    本脚本演示 DVPP 图像处理管线在视频帧上的应用。
    """

    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path
        self._init_acl()
        self._load_model()
        self._init_dvpp()

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
        print(f"[OK] AIPP-OM 模型加载成功: {self.model_path}")

    def _init_dvpp(self):
        self.dvpp_desc = acl.media.dvpp_create_channel_desc()
        ret = acl.media.dvpp_create_channel(self.dvpp_desc)
        assert ret == 0
        print("[OK] DVPP 通道创建成功")

    def _dvpp_decode_and_resize(self, jpeg_data):
        """DVPP 硬件 JPEG 解码 + VPC 缩放到 640x6640"""
        jpeg_ptr = acl.util.bytes_to_ptr(jpeg_data)
        width, height, _, ret = acl.media.dvpp_jpeg_get_image_info(
            jpeg_ptr, len(jpeg_data))
        assert ret == 0

        # JPEG -> Device
        dev_jpeg, ret = acl.media.dvpp_malloc(len(jpeg_data))
        assert ret == 0
        acl.rt.memcpy(dev_jpeg, len(jpeg_data), jpeg_ptr,
                      len(jpeg_data), ACL_MEMCPY_HOST_TO_DEVICE)

        # 解码 JPEG -> YUV420SP
        aw, ah = align_up(width, 128), align_up(height, 16)
        yuv_size = (aw * ah * 3) // 2
        dev_yuv, ret = acl.media.dvpp_malloc(yuv_size)
        assert ret == 0
        yuv_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(yuv_desc, dev_yuv)
        acl.media.dvpp_set_pic_desc_format(yuv_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        acl.media.dvpp_set_pic_desc_width(yuv_desc, width)
        acl.media.dvpp_set_pic_desc_height(yuv_desc, height)
        acl.media.dvpp_set_pic_desc_width_stride(yuv_desc, aw)
        acl.media.dvpp_set_pic_desc_height_stride(yuv_desc, ah)
        acl.media.dvpp_set_pic_desc_size(yuv_desc, yuv_size)
        ret = acl.media.dvpp_jpeg_decode_async(
            self.dvpp_desc, dev_jpeg, len(jpeg_data), yuv_desc, self.stream)
        assert ret == 0
        acl.rt.synchronize_stream(self.stream)

        # VPC 缩放到 640x640
        dst_aw, dst_ah = align_up(INPUT_SIZE, 16), align_up(INPUT_SIZE, 2)
        dst_size = (dst_aw * dst_ah * 3) // 2
        dst_desc = acl.media.dvpp_create_pic_desc()
        dev_out, ret = acl.media.dvpp_malloc(dst_size)
        assert ret == 0
        acl.media.dvpp_set_pic_desc_data(dst_desc, dev_out)
        acl.media.dvpp_set_pic_desc_format(dst_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        acl.media.dvpp_set_pic_desc_width(dst_desc, INPUT_SIZE)
        acl.media.dvpp_set_pic_desc_height(dst_desc, INPUT_SIZE)
        acl.media.dvpp_set_pic_desc_width_stride(dst_desc, dst_aw)
        acl.media.dvpp_set_pic_desc_height_stride(dst_desc, dst_ah)
        acl.media.dvpp_set_pic_desc_size(dst_desc, dst_size)
        resize_cfg = acl.media.dvpp_create_resize_config()
        ret = acl.media.dvpp_vpc_resize_async(
            self.dvpp_desc, yuv_desc, dst_desc, resize_cfg, self.stream)
        assert ret == 0
        acl.rt.synchronize_stream(self.stream)

        # YUV -> RGB (拷回 Host, 用于 AIPP-OM 的 uint8 输入)
        yuv_np = np.zeros(dst_size, dtype=np.uint8)
        acl.rt.memcpy(yuv_np.ctypes.data, dst_size, dev_out,
                      dst_size, ACL_MEMCPY_DEVICE_TO_HOST)
        # NV12: 前 dst_ah 行为 Y, 后 dst_ah/2 行为交错 UV
        yuv_buf = yuv_np.reshape(dst_ah * 3 // 2, dst_aw)
        yuv420 = yuv_buf[:INPUT_SIZE * 3 // 2, :INPUT_SIZE]
        bgr = cv2.cvtColor(yuv420, cv2.COLOR_YUV420sp2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 释放资源
        acl.media.dvpp_free(dev_jpeg)
        acl.media.dvpp_free(dev_yuv)
        acl.media.dvpp_free(dev_out)
        acl.media.dvpp_destroy_pic_desc(yuv_desc)
        acl.media.dvpp_destroy_pic_desc(dst_desc)
        acl.media.dvpp_destroy_resize_config(resize_cfg)

        return rgb

    def detect_frame(self, frame):
        """对单帧执行 DVPP+AIPP 检测"""
        orig_h, orig_w = frame.shape[:2]

        # BGR -> JPEG (DVPP 要求 JPEG)
        _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        jpeg_data = jpeg_bytes.tobytes()

        # DVPP 解码 + 缩放
        rgb640 = self._dvpp_decode_and_resize(jpeg_data)

        # 准备 uint8 输入 (AIPP-OM 期望 RGB888 HWC 交错排列)
        input_data = np.ascontiguousarray(rgb640.astype(np.uint8))

        # AIPP-OM 推理
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

        # 坐标还原
        ratio_w, ratio_h = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE
        scaled_dets = []
        for det in detections:
            x1 = int(det[0] * ratio_w)
            y1 = int(det[1] * ratio_h)
            x2 = int(det[2] * ratio_w)
            y2 = int(det[3] * ratio_h)
            scaled_dets.append([x1, y1, x2, y2, det[4], det[5]])

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
        acl.media.dvpp_destroy_channel(self.dvpp_desc)
        acl.media.dvpp_destroy_channel_desc(self.dvpp_desc)
        acl.rt.destroy_stream(self.stream)
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  昇腾香橙派视频 YOLO 目标检测 (DVPP + AIPP 路径)")
    print("  硬件: Ascend 310B4 NPU")
    print("=" * 60)

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {VIDEO_PATH}")
    print(f"  帧数: {total}, 分辨率: {orig_w}x{orig_h}, 帧率: {fps:.1f} fps")

    # 创建检测器
    detector = DVPPAIPPVideoDetector(MODEL_PATH)

    # 创建结果视频写入器
    result_path = os.path.join(OUTPUT_DIR, "dvpp_aipp_video_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, fps if fps > 0 else 25,
                             (orig_w, orig_h))

    # 逐帧检测
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

        # 在画面上叠加 FPS 信息
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

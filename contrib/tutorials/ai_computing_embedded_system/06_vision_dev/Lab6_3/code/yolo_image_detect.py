# -*- coding: utf-8 -*-
"""
yolo_image_detect.py - 昇腾香橙派图片 YOLO 目标检测 (DVPP + AIPP 硬件加速路径)
============================================================================
运行环境: 昇腾香橙派 AIPro (Ascend 310B4 NPU)
依赖: CANN Toolkit, AscendCL (acl), OpenCV, NumPy

功能:
  1. 使用 DVPP 硬件解码 JPEG 图片 -> YUV420SP
  2. 使用 DVPP VPC 硬件缩放到 640x640
  3. 使用 AIPP-OM 模型推理 (uint8 输入, 硬件归一化)
  4. YOLOv8 后处理 (解码 + NMS)
  5. 绘制检测结果并保存到 output/

用法:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 yolo_image_detect.py

对比脚本: yolo_image_detect_opencv.py (纯 OpenCV + CPU 路径)
"""

import os
import sys
import time
import numpy as np
import cv2

# 导入后处理工具模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_postprocess import (
    letterbox, yolov8_decode, yolov8_decode_raw, scale_boxes, draw_detections,
    COCO_NAMES, align_up
)

# === AscendCL 导入 ===
try:
    import acl
except ImportError:
    print("[ERROR] 无法导入 acl 模块，请先加载 CANN 环境:")
    print("  source /usr/local/Ascend/ascend-toolkit/set_env.sh")
    sys.exit(1)

# === 常量 ===
ACL_SUCCESS = 0
ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1

MODEL_PATH = "../output/yolov8n_aipp.om"   # AIPP-OM 模型路径
IMAGE_DIR = "../images"                     # 输入图片目录
OUTPUT_DIR = "../output"                    # 输出目录
INPUT_SIZE = 640                            # YOLO 输入尺寸 640x640
CONF_THRES = 0.25
IOU_THRES = 0.45
NUM_CLASSES = 80

# 测试图片列表
TEST_IMAGES = [
    "dog1.jpg", "dog2.jpg", "cat1.jpg", "cat2.jpg",
]


class DVPPAIPPDetector:
    """
    DVPP + AIPP YOLO 检测器

    处理流程:
      JPEG/PNG -> (OpenCV转JPEG) -> DVPP硬件解码 -> DVPP VPC缩放
      -> AIPP-OM推理(uint8输入) -> YOLO后处理 -> 绘制结果
    """

    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path
        self._init_acl()
        self._load_model()
        self._init_dvpp()

    def _init_acl(self):
        """初始化 AscendCL 框架"""
        ret = acl.init()
        assert ret in (0, 100002), f"acl.init failed: {ret}"
        ret = acl.rt.set_device(self.device_id)
        assert ret == 0, f"set_device failed: {ret}"
        self.context, ret = acl.rt.create_context(self.device_id)
        assert ret == 0
        self.stream, ret = acl.rt.create_stream()
        assert ret == 0
        print(f"[OK] ACL 初始化成功 (device {self.device_id})")

    def _load_model(self):
        """加载 AIPP-OM 模型"""
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        assert ret == 0, f"load model failed: {ret}"
        self.model_desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.model_desc, self.model_id)

        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        self.num_outputs = acl.mdl.get_num_outputs(self.model_desc)

        # 解析每个输出的 dims
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

        # 诊断打印
        num_in = acl.mdl.get_num_inputs(self.model_desc)
        print(f"  [diag] 输入数: {num_in}, 输出数: {self.num_outputs}")
        for i in range(num_in):
            print(f"  [diag] 输入[{i}]: size={acl.mdl.get_input_size_by_index(self.model_desc, i)}, "
                  f"dims={acl.mdl.get_input_dims(self.model_desc, i)}")
        for i in range(self.num_outputs):
            print(f"  [diag] 输出[{i}]: size={self.output_sizes[i]}, dims={self.output_dims_list[i]}")

        # 分配输入设备内存
        self.input_dev, _ = acl.rt.malloc(self.input_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        self.in_dataset = acl.mdl.create_dataset()
        self.in_buf = acl.create_data_buffer(self.input_dev, self.input_size)
        acl.mdl.add_dataset_buffer(self.in_dataset, self.in_buf)

        # 分配所有输出设备内存
        self.output_devs = []
        self.out_bufs = []
        self.out_dataset = acl.mdl.create_dataset()
        for i in range(self.num_outputs):
            dev, _ = acl.rt.malloc(self.output_sizes[i], ACL_MEM_MALLOC_NORMAL_ONLY)
            buf = acl.create_data_buffer(dev, self.output_sizes[i])
            acl.mdl.add_dataset_buffer(self.out_dataset, buf)
            self.output_devs.append(dev)
            self.out_bufs.append(buf)

        total_out = sum(self.output_sizes)
        print(f"[OK] 模型加载成功: {self.model_path}")
        print(f"     输入大小: {self.input_size} bytes ({self.input_size/1024/1024:.2f} MB, uint8)")
        print(f"     输出数: {self.num_outputs}, 总大小: {total_out} bytes")

    def _init_dvpp(self):
        """初始化 DVPP 通道"""
        self.dvpp_desc = acl.media.dvpp_create_channel_desc()
        ret = acl.media.dvpp_create_channel(self.dvpp_desc)
        assert ret == 0, f"dvpp_create_channel failed: {ret}"
        print("[OK] DVPP 通道创建成功")

    def _decode_jpeg(self, jpeg_data):
        """
        DVPP 硬件 JPEG 解码: JPEG -> YUV420SP (NV12)

        参数:
            jpeg_data: JPEG 文件字节流

        返回:
            dev_yuv: Device 上的 YUV 数据指针
            yuv_desc: 图片描述符
            width, height: 原始图片宽高
            decode_ms: 解码耗时
        """
        jpeg_ptr = acl.util.bytes_to_ptr(jpeg_data)
        width, height, _, ret = acl.media.dvpp_jpeg_get_image_info(
            jpeg_ptr, len(jpeg_data))
        assert ret == 0

        # JPEG 数据拷贝到 Device
        dev_jpeg, ret = acl.media.dvpp_malloc(len(jpeg_data))
        assert ret == 0
        acl.rt.memcpy(dev_jpeg, len(jpeg_data), jpeg_ptr,
                      len(jpeg_data), ACL_MEMCPY_HOST_TO_DEVICE)

        # 分配 YUV 输出内存 (对齐)
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

        t0 = time.time()
        ret = acl.media.dvpp_jpeg_decode_async(
            self.dvpp_desc, dev_jpeg, len(jpeg_data), yuv_desc, self.stream)
        assert ret == 0
        acl.rt.synchronize_stream(self.stream)
        decode_ms = (time.time() - t0) * 1000

        acl.media.dvpp_free(dev_jpeg)
        return dev_yuv, yuv_desc, width, height, decode_ms

    def _vpc_resize(self, src_desc, src_w, src_h, dst_w, dst_h):
        """
        DVPP 硬件 VPC 缩放: YUV420SP -> YUV420SP (目标尺寸)

        参数:
            src_desc: 源图片描述符
            src_w, src_h: 源宽高
            dst_w, dst_h: 目标宽高

        返回:
            dev_out: Device 上的缩放结果
            resize_ms: 缩放耗时
        """
        dst_aw, dst_ah = align_up(dst_w, 16), align_up(dst_h, 2)
        dst_size = (dst_aw * dst_ah * 3) // 2

        dst_desc = acl.media.dvpp_create_pic_desc()
        dev_out, ret = acl.media.dvpp_malloc(dst_size)
        assert ret == 0
        acl.media.dvpp_set_pic_desc_data(dst_desc, dev_out)
        acl.media.dvpp_set_pic_desc_format(dst_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        acl.media.dvpp_set_pic_desc_width(dst_desc, dst_w)
        acl.media.dvpp_set_pic_desc_height(dst_desc, dst_h)
        acl.media.dvpp_set_pic_desc_width_stride(dst_desc, dst_aw)
        acl.media.dvpp_set_pic_desc_height_stride(dst_desc, dst_ah)
        acl.media.dvpp_set_pic_desc_size(dst_desc, dst_size)

        resize_cfg = acl.media.dvpp_create_resize_config()
        t0 = time.time()
        ret = acl.media.dvpp_vpc_resize_async(
            self.dvpp_desc, src_desc, dst_desc, resize_cfg, self.stream)
        assert ret == 0
        acl.rt.synchronize_stream(self.stream)
        resize_ms = (time.time() - t0) * 1000

        acl.media.dvpp_destroy_pic_desc(dst_desc)
        acl.media.dvpp_destroy_resize_config(resize_cfg)
        return dev_out, resize_ms

    def _yuv_to_rgb(self, dev_yuv, width, height):
        """
        将 Device 上的 YUV420SP 数据拷贝回 Host 并转为 RGB (用于可视化)
        注意: AIPP-OM 推理时不需要此转换，AIPP 硬件自动处理
        此函数仅用于将 DVPP 解码结果转回 Host 做可视化绘制
        """
        aw, ah = align_up(width, 128), align_up(height, 16)
        yuv_size = (aw * ah * 3) // 2
        yuv_np = np.zeros(yuv_size, dtype=np.uint8)
        acl.rt.memcpy(yuv_np.ctypes.data, yuv_size, dev_yuv,
                      yuv_size, ACL_MEMCPY_DEVICE_TO_HOST)
        # YUV420SP (NV12) -> BGR
        # NV12 总大小 = aw * ah * 3/2, 前 ah 行为 Y, 后 ah/2 行为交错 UV
        yuv_buf = yuv_np.reshape(ah * 3 // 2, aw)
        yuv420 = yuv_buf[:height * 3 // 2, :width]
        bgr = cv2.cvtColor(yuv420, cv2.COLOR_YUV420sp2BGR)
        return bgr

    def detect(self, image_path):
        """
        对单张图片执行完整检测流程:
          1. PNG -> JPEG 转换 (DVPP 仅支持 JPEG)
          2. DVPP 硬件解码 JPEG -> YUV420SP
          3. DVPP VPC 硬件缩放到 640x640
          4. YUV -> RGB 转换 + letterbox (CPU)
          5. AIPP-OM 推理 (uint8 输入)
          6. YOLOv8 后处理 (解码 + NMS)
          7. 坐标还原 + 可视化

        参数:
            image_path: 图片路径 (PNG/JPG)

        返回:
            result_img: 绘制了检测框的图像
            detections: 检测结果列表
            timing: 各步骤耗时字典
        """
        timing = {}

        # Step 1: 读取图片并转为 JPEG (DVPP 要求 JPEG 输入)
        t0 = time.time()
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        orig_h, orig_w = orig_img.shape[:2]
        # 编码为 JPEG
        _, jpeg_bytes = cv2.imencode('.jpg', orig_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        jpeg_data = jpeg_bytes.tobytes()
        timing['png_to_jpeg'] = (time.time() - t0) * 1000

        # Step 2: DVPP 硬件 JPEG 解码
        dev_yuv, yuv_desc, dec_w, dec_h, timing['dvpp_decode'] = \
            self._decode_jpeg(jpeg_data)

        # Step 3: DVPP VPC 硬件缩放到 640x640
        dev_resized, timing['dvpp_resize'] = \
            self._vpc_resize(yuv_desc, dec_w, dec_h, INPUT_SIZE, INPUT_SIZE)

        # Step 4: YUV -> BGR -> RGB (CPU, 用于可视化原图)
        # 注意: AIPP-OM 期望 uint8 RGB888 HWC 交错排列 (非 NCHW)
        t0 = time.time()
        bgr_img = self._yuv_to_rgb(dev_resized, INPUT_SIZE, INPUT_SIZE)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        input_data = np.ascontiguousarray(rgb_img.astype(np.uint8))
        timing['yuv_to_rgb'] = (time.time() - t0) * 1000

        # Step 5: AIPP-OM 推理
        print("  [step] 推理中...", end="", flush=True)
        t0 = time.time()
        acl.rt.memcpy(self.input_dev, self.input_size,
                      input_data.ctypes.data, self.input_size,
                      ACL_MEMCPY_HOST_TO_DEVICE)
        ret = acl.mdl.execute(self.model_id, self.in_dataset, self.out_dataset)
        assert ret == 0, f"execute failed: {ret}"
        acl.rt.synchronize_stream(self.stream)
        timing['inference'] = (time.time() - t0) * 1000
        print(f" {timing['inference']:.1f}ms", flush=True)

        # Step 6: 拷贝所有输出回 Host 并后处理
        print("  [step] 后处理中...", end="", flush=True)
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
        timing['postprocess'] = (time.time() - t0) * 1000
        print(f" {timing['postprocess']:.1f}ms, 检测到 {len(detections)} 个目标", flush=True)

        # Step 7: 坐标还原到原图并绘制
        # DVPP 直接缩放到 640x640 (非 letterbox)，需按比例还原
        ratio_w, ratio_h = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE
        scaled_dets = []
        for det in detections:
            x1 = int(det[0] * ratio_w)
            y1 = int(det[1] * ratio_h)
            x2 = int(det[2] * ratio_w)
            y2 = int(det[3] * ratio_h)
            scaled_dets.append([x1, y1, x2, y2, det[4], det[5]])

        result_img = draw_detections(orig_img.copy(), scaled_dets)

        # 释放本张图片 DVPP 资源
        acl.media.dvpp_free(dev_yuv)
        acl.media.dvpp_free(dev_resized)
        acl.media.dvpp_destroy_pic_desc(yuv_desc)

        return result_img, scaled_dets, timing

    def cleanup(self):
        """释放所有资源"""
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
        acl.media.dvpp_destroy_channel(self.dvpp_desc)
        acl.media.dvpp_destroy_channel_desc(self.dvpp_desc)
        acl.rt.destroy_stream(self.stream)
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("[OK] 所有资源已释放")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  昇腾香橙派图片 YOLO 目标检测 (DVPP + AIPP 路径)")
    print("  硬件: Ascend 310B4 NPU")
    print("=" * 60)

    # 创建检测器
    detector = DVPPAIPPDetector(MODEL_PATH)

    # 依次处理测试图片
    all_results = []
    for img_name in TEST_IMAGES:
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"[SKIP] {img_path} 不存在")
            continue

        print(f"\n--- 处理: {img_name} ---")
        result_img, detections, timing = detector.detect(img_path)

        total_ms = sum(timing.values())
        print(f"  DVPP 解码: {timing['dvpp_decode']:.2f} ms")
        print(f"  DVPP 缩放: {timing['dvpp_resize']:.2f} ms")
        print(f"  推理:      {timing['inference']:.2f} ms")
        print(f"  后处理:    {timing['postprocess']:.2f} ms")
        print(f"  总耗时:    {total_ms:.2f} ms")
        print(f"  检测到 {len(detections)} 个目标:")
        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            print(f"    {COCO_NAMES[cls_id]}: {score:.2f}  [{x1},{y1},{x2},{y2}]")

        # 保存结果
        out_path = os.path.join(OUTPUT_DIR, f"dvpp_aipp_{img_name}")
        cv2.imwrite(out_path, result_img)
        print(f"  结果已保存: {out_path}")
        all_results.append({'name': img_name, 'timing': timing,
                            'detections': detections, 'total': total_ms})

    # 汇总
    print("\n" + "=" * 60)
    print("  DVPP + AIPP 图片检测汇总")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['name']}: {r['total']:.2f} ms, {len(r['detections'])} 个目标")
    if all_results:
        avg_ms = sum(r['total'] for r in all_results) / len(all_results)
        print(f"  平均耗时: {avg_ms:.2f} ms/张")
    print("[OK] DVPP + AIPP 图片检测完成！")

    detector.cleanup()


if __name__ == '__main__':
    main()

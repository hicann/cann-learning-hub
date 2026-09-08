# -*- coding: utf-8 -*-
"""
yolo_usb_camera_detect.py - 昇腾香橙派 USB 摄像头实时 YOLO 目标检测 + HDMI 显示
==================================================================================
运行环境: 昇腾香橙派 AIPro (Ascend 310B4 NPU)
硬件连接:
  - USB 摄像头 -> 香橙派 USB 接口 (视频采集)
  - HDMI 线 -> 香橙派 HDMI 接口 -> 显示器 (结果显示)

依赖: CANN Toolkit, AscendCL (acl), OpenCV, NumPy

功能:
  1. 通过 USB 接口读取摄像头视频帧 (OpenCV VideoCapture)
  2. 逐帧使用 DVPP 硬件解码 + VPC 缩放 (NPU 硬件加速)
  3. 使用 AIPP-OM 模型推理 (uint8 输入, 硬件归一化)
  4. YOLOv8 后处理 (解码 + NMS)
  5. 绘制检测框 + FPS 信息
  6. 通过 HDMI 接口实时显示到显示器 (OpenCV imshow / DRM 直显)

数据流:
  USB摄像头 -> USB接口 -> 香橙派内存 -> DVPP硬件解码 -> DVPP VPC缩放
  -> AIPP-OM推理(NPU) -> YOLO后处理 -> 绘制检测框 -> HDMI接口 -> 显示器

用法:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python3 yolo_usb_camera_detect.py [摄像头索引] [显示模式]

  参数:
    摄像头索引: 默认 0 (第一个 USB 摄像头)
    显示模式:   "opencv" (默认, 需要桌面环境) 或 "drm" (直显, 无需桌面)

示例:
  python3 yolo_usb_camera_detect.py 0 opencv    # 使用 OpenCV 窗口显示
  python3 yolo_usb_camera_detect.py 0 drm       # 使用 DRM 直显 (headless)
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
OUTPUT_DIR = "../output"
INPUT_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
NUM_CLASSES = 80

# 摄像头参数
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DISPLAY_MODE = sys.argv[2] if len(sys.argv) > 2 else "opencv"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


class USBCameraYOLODetector:
    """
    USB 摄像头实时 YOLO 检测器 (DVPP + AIPP 硬件加速)

    完整数据流:
      USB摄像头 --(USB接口)--> 香橙派内存(Host)
        -> OpenCV采集BGR帧
        -> 编码JPEG
        -> DVPP硬件解码(JPEG->YUV420SP) [NPU]
        -> DVPP VPC硬件缩放(->640x640)  [NPU]
        -> AIPP-OM推理(uint8输入)       [NPU AI Core]
        -> YOLOv8后处理(解码+NMS)       [CPU]
        -> 绘制检测框+FPS               [CPU]
        --(HDMI接口)--> 显示器实时显示

    三种实现方式对比:
      1. DVPP+AIPP (本脚本): 硬件解码+缩放+归一化, 最快, uint8输入
      2. OpenCV+AIPP: 软件解码缩放, 硬件归一化, 中等
      3. 纯OpenCV+纯OM: 全软件预处理, float32输入, 最慢
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
        assert ret == 0
        self.context, _ = acl.rt.create_context(self.device_id)
        self.stream, _ = acl.rt.create_stream()
        print(f"[OK] ACL 初始化成功 (device {self.device_id})")

    def _load_model(self):
        """加载 AIPP-OM 模型"""
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
        print(f"[OK] AIPP-OM 模型加载: {self.model_path}")
        print(f"     输入: {self.input_size} bytes ({self.input_size/1024/1024:.2f} MB, uint8)")

    def _init_dvpp(self):
        """初始化 DVPP 通道"""
        self.dvpp_desc = acl.media.dvpp_create_channel_desc()
        ret = acl.media.dvpp_create_channel(self.dvpp_desc)
        assert ret == 0
        print("[OK] DVPP 通道创建成功")

    def _dvpp_process(self, jpeg_data):
        """
        DVPP 硬件处理: JPEG解码 -> YUV420SP -> VPC缩放到640x640 -> RGB

        返回:
            rgb640: 640x640 RGB uint8 图像 (用于 AIPP-OM 推理输入)
            decode_ms: 解码耗时
            resize_ms: 缩放耗时
        """
        jpeg_ptr = acl.util.bytes_to_ptr(jpeg_data)
        width, height, _, ret = acl.media.dvpp_jpeg_get_image_info(
            jpeg_ptr, len(jpeg_data))
        assert ret == 0

        # JPEG -> Device
        dev_jpeg, ret = acl.media.dvpp_malloc(len(jpeg_data))
        assert ret == 0
        acl.rt.memcpy(dev_jpeg, len(jpeg_data), jpeg_ptr,
                      len(jpeg_data), ACL_MEMCPY_HOST_TO_DEVICE)

        # DVPP JPEG 解码 -> YUV420SP
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

        # DVPP VPC 缩放 -> 640x640
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

        t0 = time.time()
        ret = acl.media.dvpp_vpc_resize_async(
            self.dvpp_desc, yuv_desc, dst_desc, resize_cfg, self.stream)
        assert ret == 0
        acl.rt.synchronize_stream(self.stream)
        resize_ms = (time.time() - t0) * 1000

        # YUV420SP -> BGR -> RGB (拷回 Host)
        yuv_np = np.zeros(dst_size, dtype=np.uint8)
        acl.rt.memcpy(yuv_np.ctypes.data, dst_size, dev_out,
                      dst_size, ACL_MEMCPY_DEVICE_TO_HOST)
        yuv_buf = yuv_np.reshape(dst_ah * 3 // 2, dst_aw)
        yuv420 = yuv_buf[:INPUT_SIZE * 3 // 2, :INPUT_SIZE]
        bgr = cv2.cvtColor(yuv420, cv2.COLOR_YUV420sp2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 释放
        acl.media.dvpp_free(dev_jpeg)
        acl.media.dvpp_free(dev_yuv)
        acl.media.dvpp_free(dev_out)
        acl.media.dvpp_destroy_pic_desc(yuv_desc)
        acl.media.dvpp_destroy_pic_desc(dst_desc)
        acl.media.dvpp_destroy_resize_config(resize_cfg)

        return rgb, decode_ms, resize_ms

    def detect_frame(self, frame):
        """
        对单帧执行完整检测:
          BGR帧 -> JPEG编码 -> DVPP解码 -> DVPP缩放 -> AIPP推理 -> 后处理

        返回:
            detections: 检测结果列表 [x1,y1,x2,y2,score,cls_id]
            timing: 各步骤耗时
        """
        timing = {}
        orig_h, orig_w = frame.shape[:2]

        # BGR -> JPEG (DVPP 要求 JPEG 输入)
        t0 = time.time()
        _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        jpeg_data = jpeg_bytes.tobytes()
        timing['jpeg_encode'] = (time.time() - t0) * 1000

        # DVPP 硬件解码 + 缩放
        rgb640, timing['dvpp_decode'], timing['dvpp_resize'] = \
            self._dvpp_process(jpeg_data)

        # 准备 uint8 输入 (AIPP-OM 期望 RGB888 HWC 交错排列, AIPP 硬件做归一化)
        input_data = np.ascontiguousarray(rgb640.astype(np.uint8))

        # AIPP-OM 推理
        t0 = time.time()
        acl.rt.memcpy(self.input_dev, self.input_size,
                      input_data.ctypes.data, self.input_size,
                      ACL_MEMCPY_HOST_TO_DEVICE)
        ret = acl.mdl.execute(self.model_id, self.in_dataset, self.out_dataset)
        assert ret == 0
        acl.rt.synchronize_stream(self.stream)
        timing['inference'] = (time.time() - t0) * 1000

        # 后处理 (读取所有输出)
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

        # 坐标还原到原图
        ratio_w, ratio_h = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE
        scaled_dets = []
        for det in detections:
            x1 = int(det[0] * ratio_w)
            y1 = int(det[1] * ratio_h)
            x2 = int(det[2] * ratio_w)
            y2 = int(det[3] * ratio_h)
            scaled_dets.append([x1, y1, x2, y2, det[4], det[5]])

        return scaled_dets, timing

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


class DRMDisplay:
    """
    DRM 直显模块: 无需桌面环境，直接通过 HDMI 显示到显示器
    适用于香橙派 headless 模式 (无 X11 桌面)

    原理:
      使用 Linux DRM (Direct Rendering Manager) 子系统
      直接操作 HDMI 显示帧缓冲区 (framebuffer)
      适合嵌入式设备无桌面环境下的实时显示

    注意: 需要 pydrm 或直接操作 /dev/dri/card0
    本类为示意实现，实际使用时需安装 pydrm 或使用 C 扩展
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._init_drm()

    def _init_drm(self):
        """初始化 DRM 显示"""
        try:
            import pydrm
            self.drm = pydrm.DRM()
            self.drm.open("/dev/dri/card0")
            print(f"[OK] DRM 初始化成功: {self.width}x{self.height}")
        except ImportError:
            print("[WARN] pydrm 未安装, DRM 直显不可用")
            print("       安装方法: pip3 install pydrm")
            print("       或使用 OpenCV 显示模式: python3 yolo_usb_camera_detect.py 0 opencv")
            self.drm = None

    def show(self, frame):
        """显示一帧到 HDMI"""
        if self.drm is not None:
            self.drm.display(frame)

    def close(self):
        if self.drm is not None:
            self.drm.close()


def main():
    global DISPLAY_MODE
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  昇腾香橙派 USB 摄像头实时 YOLO 目标检测 + HDMI 显示")
    print("  硬件: Ascend 310B4 NPU")
    print("  数据流: USB摄像头 -> DVPP+AIPP(NPU) -> HDMI显示器")
    print("=" * 65)
    print(f"  摄像头索引: {CAMERA_INDEX}")
    print(f"  显示模式:   {DISPLAY_MODE}")
    print(f"  模型:       {MODEL_PATH}")
    print()

    # === Step 1: 打开 USB 摄像头 ===
    print("[Step 1] 打开 USB 摄像头...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {CAMERA_INDEX}")
        print("  请检查:")
        print("  1. USB 摄像头已插入香橙派 USB 接口")
        print("  2. 摄像头设备: ls /dev/video*")
        print("  3. 摄像头权限: sudo chmod 666 /dev/video*")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  摄像头分辨率: {cam_w}x{cam_h}, 帧率: {cam_fps:.1f} fps")

    # === Step 2: 初始化检测器 (DVPP + AIPP) ===
    print("\n[Step 2] 初始化 DVPP + AIPP 检测器...")
    detector = USBCameraYOLODetector(MODEL_PATH)

    # === Step 3: 初始化显示模块 ===
    print(f"\n[Step 3] 初始化 HDMI 显示 (模式: {DISPLAY_MODE})...")
    drm_display = None
    if DISPLAY_MODE == "drm":
        drm_display = DRMDisplay(cam_w, cam_h)
        if drm_display.drm is None:
            print("  回退到 OpenCV 显示模式")
            DISPLAY_MODE = "opencv"
    if DISPLAY_MODE == "opencv":
        cv2.namedWindow("YOLO Detection (Orange Pi)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Detection (Orange Pi)", cam_w, cam_h)
        print("  OpenCV 显示窗口已创建 (通过 HDMI 输出到显示器)")

    # === Step 4: 实时检测循环 ===
    print("\n[Step 4] 开始实时检测 (按 'q' 退出, 's' 保存截图)...")
    print("  数据流: USB摄像头 -> DVPP解码 -> DVPP缩放 -> AIPP推理 -> HDMI显示")

    frame_count = 0
    fps_history = []
    fps_avg = 0
    last_time = time.time()

    # 用于保存检测录像
    record = False
    writer = None

    try:
        while True:
            # 从 USB 摄像头读取一帧
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 摄像头读取失败，重试...")
                continue

            frame_count += 1

            # YOLO 检测 (DVPP + AIPP 硬件加速)
            detections, timing = detector.detect_frame(frame)

            # 绘制检测结果
            result_frame = draw_detections(frame, detections)

            # 计算 FPS
            now = time.time()
            instant_fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0
            last_time = now
            fps_history.append(instant_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            fps_avg = np.mean(fps_history)

            # 在画面上叠加信息
            total_ms = sum(timing.values())
            info_lines = [
                f"FPS: {fps_avg:.1f}  Frame: {frame_count}",
                f"DVPP decode: {timing['dvpp_decode']:.1f}ms  resize: {timing['dvpp_resize']:.1f}ms",
                f"Inference: {timing['inference']:.1f}ms  Post: {timing['postprocess']:.1f}ms",
                f"Total: {total_ms:.1f}ms  Objects: {len(detections)}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(result_frame, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 打印检测到的目标
            if detections:
                det_str = ", ".join(
                    f"{COCO_NAMES[d[5]]}:{d[4]:.2f}" for d in detections)
                print(f"\r  Frame {frame_count}: {det_str}", end="", flush=True)

            # 录像
            if record and writer is not None:
                writer.write(result_frame)

            # === HDMI 显示 ===
            if DISPLAY_MODE == "opencv":
                cv2.imshow("YOLO Detection (Orange Pi)", result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  用户按 'q' 退出")
                    break
                elif key == ord('s'):
                    snap_path = os.path.join(OUTPUT_DIR, f"usb_camera_snap_{frame_count}.jpg")
                    cv2.imwrite(snap_path, result_frame)
                    print(f"\n  截图已保存: {snap_path}")
                elif key == ord('r'):
                    record = not record
                    if record:
                        rec_path = os.path.join(OUTPUT_DIR, "usb_camera_record.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(rec_path, fourcc, 30, (cam_w, cam_h))
                        print(f"\n  开始录像: {rec_path}")
                    else:
                        if writer:
                            writer.release()
                            writer = None
                        print("\n  停止录像")
            elif DISPLAY_MODE == "drm":
                drm_display.show(result_frame)

    except KeyboardInterrupt:
        print("\n  用户中断 (Ctrl+C)")

    # === 清理资源 ===
    print("\n\n[Cleanup] 释放资源...")
    cap.release()
    if writer is not None:
        writer.release()
    if DISPLAY_MODE == "opencv":
        cv2.destroyAllWindows()
    elif DISPLAY_MODE == "drm" and drm_display:
        drm_display.close()
    detector.cleanup()

    print(f"\n{'=' * 65}")
    print(f"  USB 摄像头实时检测结束")
    print(f"  总帧数: {frame_count}")
    print(f"  平均 FPS: {fps_avg:.1f}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""准备 YOLOv8n ONNX 模型。

仓库不再随附 yolov8n.onnx / mini_yolo.onnx 等模型文件（ONNX 不属于仓库准入文件类型）。
本脚本优先从直连地址下载预导出的 yolov8n.onnx；若直连失败，则通过 ultralytics
获取官方 yolov8n.pt 权重并导出为 yolov8n.onnx，放到当前目录。

用法：
    python3 download_yolov8n.py            # 默认输出 yolov8n.onnx 到当前目录
    python3 download_yolov8n.py --out /path/yolov8n.onnx

直连下载无需额外依赖；ultralytics 回退方式需 pip install ultralytics onnx
"""
import argparse, os, sys, urllib.request

DEFAULT_OUT = "yolov8n.onnx"
DIRECT_URL = "https://www.qmpan.com/f/1LXkCp/yolov8n.onnx"


def download_direct(out_path):
    print(f"[INFO] 正在从直连地址下载 yolov8n.onnx: {DIRECT_URL}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(DIRECT_URL, out_path)
    print(f"[OK] 已下载: {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")


def export_via_ultralytics(out_path):
    from ultralytics import YOLO
    print("[INFO] 直连下载失败，回退至 ultralytics 导出方式...")
    weight = "yolov8n.pt"
    model = YOLO(weight)
    model.export(format="onnx", imgsz=640, simplify=True, opset=12)
    exported = "yolov8n.onnx"
    if exported != out_path and os.path.exists(exported):
        os.replace(exported, out_path)
    print(f"[OK] 已导出 ONNX 模型: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 ONNX 文件路径")
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[SKIP] {args.out} 已存在，跳过下载。")
        return

    try:
        download_direct(args.out)
        return
    except Exception as e:
        print(f"[WARN] 直连下载失败: {e}")

    try:
        export_via_ultralytics(args.out)
    except ImportError:
        print("[ERROR] 直连下载失败，且未安装 ultralytics 回退依赖。", file=sys.stderr)
        print("        请 pip install ultralytics onnx 后重运行，或手动从直连地址下载：",
              file=sys.stderr)
        print(f"        {DIRECT_URL}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

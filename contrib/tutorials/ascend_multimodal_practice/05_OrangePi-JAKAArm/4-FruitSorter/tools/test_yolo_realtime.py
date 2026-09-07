#!/usr/bin/env python3
"""
实时测试YOLO模型识别准确率
按Q退出
"""
import cv2
import numpy as np
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use custom YOLO detector with OM model
from perception.yolo_detector import YoloDetector
print("Using YOLO detector with OM model")

# Class names for color palette (matching best-fruits.om model)
CLASSES = ['cavocado', 'lemon', 'pear', 'mango', 'persimmon']

# Generate fixed color palette for each class
np.random.seed(42)
COLOR_PALETTE = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_box(img, box, score, class_id, class_name):
    """Draws a bounding box on the image"""
    # Retrieve the color for the class ID
    color = COLOR_PALETTE[class_id % len(CLASSES)]
    
    # Draw the bounding box on the image
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

    # Create the label text with class name and score
    label = f'{class_name}: {score:.2f}'

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Calculate the position of the label text
    label_x = box[0]
    label_y = box[1] - 10 if box[1] - 10 > label_height else box[1] + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img,
        (int(label_x), int(label_y - label_height)),
        (int(label_x + label_width), int(label_y + label_height)),
        color,
        cv2.FILLED,
    )

    # Draw the label text on the image
    cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def test_with_custom_detector():
    """使用自定义检测器测试"""
    print("\nUsing custom YOLO detector")
    
    # 初始化检测器 (使用OM模型)
    detector = YoloDetector("models/best-fruits.om")
    if not detector.is_ready():
        print("Detector failed to initialize")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nPress 'Q' to quit")
    print("Press 'S' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 运行检测
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = (time.time() - start_time) * 1000
        
        # 创建显示图像
        display = frame.copy()
        
        # 绘制检测结果
        for det in detections:
            cx, cy = det['bbox'][:2]
            w, h = det['width'], det['height']
            conf = det['conf']
            class_name = det['class']
            class_id = det['class_id']
            
            # 计算边界框坐标
            xmin = int(cx - w / 2)
            ymin = int(cy - h / 2)
            xmax = int(cx + w / 2)
            ymax = int(cy + h / 2)
            
            # 绘制边界框和标签
            draw_box(display, [xmin, ymin, xmax, ymax], conf, class_id, class_name)
        
        # 显示信息
        fps_text = f"FPS: {1000/inference_time:.1f}" if inference_time > 0 else "FPS: --"
        cv2.putText(display, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Detections: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示
        cv2.imshow("Custom YOLO Detection", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"custom_detection_{timestamp}.jpg"
            cv2.imwrite(filename, display)
            print(f"Frame saved as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("YOLO Model Accuracy Test (OM Model)")
    print("="*60)
    test_with_custom_detector()

if __name__ == "__main__":
    main()
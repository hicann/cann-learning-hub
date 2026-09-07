# file: tools/calib_camera_auto.py

import cv2
import numpy as np
import csv
import os
import time

def show_detected_circles(frame, circles, title="Detected Circles"):
    """显示检测到的圆，方便调试"""
    display = frame.copy()
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for i, (x, y, r) in enumerate(circles):
            # 绘制圆
            cv2.circle(display, (x, y), r, (0, 255, 0), 2)  # 绿色圆
            cv2.circle(display, (x, y), 2, (0, 0, 255), 3)  # 红色圆心
            
            # 显示序号
            cv2.putText(display, str(i+1), (x-10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 显示坐标和半径
            info = f"({x},{y}) r={r}"
            cv2.putText(display, info, (x-30, y+r+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # 显示统计信息
    count = len(circles[0]) if circles is not None else 0
    cv2.putText(display, f"Detected: {count} circles", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow(title, display)
    return display

def auto_detect_and_sort_debug(frame):
    """带调试信息的自动检测"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 显示预处理图像
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    print("\n=== 开始检测 ===")
    print(f"图像尺寸: {frame.shape}")
    
    # 非常宽松的参数
    params = {
        'dp': 1.1,
        'minDist': 50,
        'param1': 30,
        'param2': 18,  # 这个值越小，检测到的圆越多
        'minRadius': 10,
        'maxRadius': 50
    }
    
    print(f"使用参数: {params}")
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=params['dp'],
        minDist=params['minDist'],
        param1=params['param1'],
        param2=params['param2'],
        minRadius=params['minRadius'],
        maxRadius=params['maxRadius']
    )
    
    # 显示检测结果
    if circles is not None:
        circles_rounded = np.round(circles[0, :]).astype("int")
        print(f"检测到 {len(circles_rounded)} 个圆:")
        
        for i, (x, y, r) in enumerate(circles_rounded):
            print(f"  圆{i+1}: 中心({x}, {y}), 半径{r}")
        
        # 显示图像
        display = show_detected_circles(frame, circles, "All Detected Circles")
        
        # 询问是否继续
        print("\n按 'c' 继续筛选为9个圆，按 'q' 退出")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                # 筛选最好的9个圆
                best_circles = select_best_9_circles(circles_rounded, blurred)
                print(f"\n筛选后剩余 {len(best_circles)} 个圆:")
                
                for i, (x, y, r) in enumerate(best_circles):
                    print(f"  圆{i+1}: 中心({x}, {y}), 半径{r}")
                
                # 显示筛选结果
                best_circles_array = np.array([best_circles], dtype=np.float32)
                show_detected_circles(frame, best_circles_array, "Best 9 Circles")
                
                # 排序
                if len(best_circles) == 9:
                    sorted_points = sort_circles(best_circles)
                    return True, sorted_points
                else:
                    print(f"筛选后只有 {len(best_circles)} 个圆，不足9个")
                    return False, None
                    
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False, None
    else:
        print("未检测到任何圆")
        return False, None
    
    cv2.destroyAllWindows()
    return False, None

def select_best_9_circles(circles, gray_image):
    """从多个圆中筛选出最好的9个"""
    if len(circles) <= 9:
        return circles
    
    print(f"\n从 {len(circles)} 个圆中筛选最好的9个...")
    
    scores = []
    for (x, y, r) in circles:
        # 创建一个掩码
        mask = np.zeros(gray_image.shape[:2], dtype="uint8")
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # 计算掩码区域内的边缘强度
        edges = cv2.Canny(gray_image, 50, 150)
        edge_in_mask = cv2.bitwise_and(edges, edges, mask=mask)
        
        # 得分：边缘强度/周长（理想圆边缘强度高）
        edge_strength = np.sum(edge_in_mask) / (2 * np.pi * r) if r > 0 else 0
        
        # 额外的得分：圆形度（周长/面积比）
        area = np.pi * r * r
        perimeter = 2 * np.pi * r
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 综合得分
        total_score = edge_strength * 0.7 + circularity * 0.3
        scores.append(total_score)
        
        print(f"  圆({x},{y}) r={r}: 边缘得分={edge_strength:.2f}, 圆形度={circularity:.2f}, 总分={total_score:.2f}")
    
    # 选择得分最高的9个圆
    circles_with_scores = list(zip(circles, scores))
    circles_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n得分排名:")
    for i, ((x, y, r), score) in enumerate(circles_with_scores):
        print(f"  第{i+1}名: 圆({x},{y}) r={r}, 得分={score:.2f}")
    
    best_circles = [c for c, s in circles_with_scores[:9]]
    
    # 按坐标排序
    best_circles = sorted(best_circles, key=lambda x: (x[1], x[0]))
    
    return best_circles

def sort_circles(circles):
    """排序9个圆点"""
    # 按 y 坐标分组为3行
    circles_sorted_by_y = sorted(circles, key=lambda x: x[1])
    
    # 计算行间距
    y_coords = [c[1] for c in circles_sorted_by_y]
    row_gap = (y_coords[-1] - y_coords[0]) / 2  # 大致行间距
    
    rows = [[], [], []]
    current_row = 0
    current_y = circles_sorted_by_y[0][1]
    
    for circle in circles_sorted_by_y:
        if abs(circle[1] - current_y) > row_gap * 0.5:
            current_row += 1
            current_y = circle[1]
        
        if current_row < 3:
            rows[current_row].append(circle)
    
    # 每行内按 x 排序
    for i in range(3):
        rows[i] = sorted(rows[i], key=lambda x: x[0])
    
    # 合并
    sorted_circles = rows[0] + rows[1] + rows[2]
    points = [[int(c[0]), int(c[1])] for c in sorted_circles]
    
    print("\n排序结果:")
    for i, (x, y) in enumerate(points):
        row = i // 3 + 1
        col = i % 3 + 1
        print(f"  点{i+1} (行{row},列{col}): [{x}, {y}]")
    
    return points

def run_camera_calib():
    """
    摄像头九点标定 - 实时预览 + 鼠标点击
    """
    os.makedirs("data", exist_ok=True)

    # 打开摄像头
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    if not cap.isOpened():
        print("尝试索引1...")
        cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    # 配置摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 预热
    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    print("=== 摄像头九点标定（实时预览模式） ===")
    print("操作说明：")
    print("  - 实时预览摄像头画面")
    print("  - 按 'c' 截取当前画面进行标定")
    print("  - 按 'q' 退出标定")
    print()

    manual_points = []
    captured_frame = None
    is_capturing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal manual_points
        if event == cv2.EVENT_LBUTTONDOWN and is_capturing:
            if len(manual_points) < 9:
                manual_points.append([x, y])
                print(f"点击第 {len(manual_points)} 点: [{x}, {y}]")
                
                if len(manual_points) == 9:
                    print("\n已点击9个点，按 's' 保存，按 'r' 重新截图，按 'q' 取消")
    
    cv2.namedWindow("Camera Calibration")
    cv2.setMouseCallback("Camera Calibration", mouse_callback)
    
    while True:
        if not is_capturing:
            # 实时预览模式
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 读取帧失败")
                time.sleep(0.1)
                continue
            
            display = frame.copy()
            cv2.putText(display, "Press 'c' to capture, 'q' to quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera Calibration", display)
        else:
            # 标定模式 - 显示截取的画面
            display = captured_frame.copy()
            
            # 绘制已点击的点
            for i, (px, py) in enumerate(manual_points):
                cv2.circle(display, (px, py), 15, (0, 255, 255), -1)
                cv2.putText(display, str(i+1), (px-15, py-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            
            # 显示状态
            status = f"Points: {len(manual_points)}/9 | 's'=save, 'r'=recapture, 'q'=quit"
            cv2.putText(display, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Camera Calibration", display)
        
        k = cv2.waitKey(30) & 0xFF
        
        if k == ord('q') or k == 27:  # q 或 ESC 退出
            print("标定取消")
            break
        elif k == ord('c') and not is_capturing:
            # 截取当前画面
            ret, captured_frame = cap.read()
            if ret:
                is_capturing = True
                manual_points = []
                print("\n已截取画面，请依次点击9个圆点中心（按行顺序：1-2-3, 4-5-6, 7-8-9）")
        elif k == ord('r') and is_capturing:
            # 重新截取
            is_capturing = False
            manual_points = []
            print("\n返回实时预览，按 'c' 重新截取")
        elif k == ord('s') and is_capturing and len(manual_points) == 9:
            save_points(manual_points, captured_frame)
            print("[SUCCESS] 摄像头标定完成！")
            break

    cap.release()
    cv2.destroyAllWindows()

def save_points(points, frame):

    csv_path = "data/cam_point.csv"
    img_path = "data/cam_point_review.jpg"
    

    height = frame.shape[0]
    corrected_points = []
    
    
    for i, (x, y) in enumerate(points):

        y_corrected = y
        corrected_points.append([x, y_corrected])
        
        print(f"  点{i+1}: ({x:.0f}, {y:.0f}) -> ({x:.0f}, {y_corrected:.0f})")
    
    # 保存修正后的坐标
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(corrected_points)
    
    print(f"\n[CAMERA] 九点坐标已保存到 {csv_path} ")
    
    # 保存复核图（显示原始坐标）
    disp = frame.copy()
    for i, (x, y) in enumerate(points):  # 使用原始坐标显示
        cv2.circle(disp, (int(x), int(y)), 15, (0, 255, 0), 5)
        cv2.putText(disp, str(i+1), (int(x)-15, int(y)-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    
    cv2.imwrite(img_path, disp)
    
    print("\n保存的点坐标:")
    for i, (x, y_corrected) in enumerate(corrected_points):
        row = i // 3 + 1
        col = i % 3 + 1
        print(f"  点{i+1} (行{row},列{col}): [{x:.0f}, {y_corrected:.0f}]")

# 测试函数
def test_calibration():
    """测试标定"""
    print("测试标定功能...")
    run_camera_calib()

if __name__ == "__main__":
    test_calibration()
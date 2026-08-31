"""
摄像头调试模块 - 实时预览、检测、抓取命令处理
"""
import cv2
import threading
import time
import os

# 检测缓存阈值
DETECTION_THRESHOLD = 10

# 水果名称映射：中文 -> 英文（按模型类别）
FRUIT_MAP = {
    '牛油果': 'cavocado',  # class 0 -> 篮子1
    '柠檬': 'lemon',       # class 1 -> 篮子2
    '梨': 'pear',          # class 2 -> 篮子3
    '芒果': 'mango',       # class 3 -> 篮子4
    '柿子': 'persimmon'    # class 4 -> 篮子5
}

# 英文 -> 中文映射（用于显示）
FRUIT_MAP_REVERSE = {v: k for k, v in FRUIT_MAP.items()}


class DetectionCache:
    """检测缓存管理类"""
    
    def __init__(self, threshold=DETECTION_THRESHOLD):
        self.cache = {}
        self.threshold = threshold
        self.ready_fruits = []  # 已确认的水果列表（按确认顺序）
    
    def update(self, detections):
        """更新检测缓存
        detections: list of dict, each dict has 'class', 'bbox', 'conf', etc.
        """
        # 统计每个类别的检测
        detected_classes = set()
        for det in detections:
            class_name = det['class']
            detected_classes.add(class_name)
            
            if class_name not in self.cache:
                self.cache[class_name] = {
                    'count': 0,
                    'last_position': None,
                    'last_confidence': None
                }
            
            self.cache[class_name]['count'] += 1
            self.cache[class_name]['last_position'] = det['bbox']  # [center_x, center_y]
            self.cache[class_name]['last_confidence'] = det['conf']
            
            # 检查是否刚达到阈值（首次确认）
            if self.cache[class_name]['count'] == self.threshold:
                if class_name not in self.ready_fruits:
                    self.ready_fruits.append(class_name)
                    chinese_name = FRUIT_MAP_REVERSE.get(class_name, class_name)
                    print(f"\n🍎 检测确认: {chinese_name}")
                    self._print_ready_list()
    
    def _print_ready_list(self):
        """打印可抓取水果列表"""
        if self.ready_fruits:
            print("=" * 40)
            print("🍎 可抓取水果列表：")
            for i, fruit in enumerate(self.ready_fruits):
                chinese_name = FRUIT_MAP_REVERSE.get(fruit, fruit)
                print(f"  按 {i+1} 抓取 {chinese_name}")
            print("=" * 40)
    
    def is_ready(self, class_name):
        """检查某个类别是否达到检测阈值"""
        if class_name in self.cache:
            return self.cache[class_name]['count'] >= self.threshold
        return False
    
    def get_ready_fruit_by_index(self, index):
        """根据索引获取已确认的水果"""
        if 0 <= index < len(self.ready_fruits):
            return self.ready_fruits[index]
        return None
    
    def get_position(self, class_name):
        """获取某个类别的缓存位置"""
        if class_name in self.cache and self.cache[class_name]['last_position']:
            return self.cache[class_name]['last_position']
        return None
    
    def get_info(self, class_name):
        """获取某个类别的完整缓存信息"""
        return self.cache.get(class_name, None)
    
    def clear(self):
        """清空缓存"""
        self.cache = {}
        self.ready_fruits = []


def draw_detections(frame, detections, detection_cache):
    """在画面上绘制检测结果"""
    if detections:
        for det in detections:
            class_name = det['class']
            bbox = det['bbox']  # [center_x, center_y]
            conf = det['conf']
            width = det.get('width', 100)
            height = det.get('height', 100)
            
            # 计算边界框
            center_x, center_y = int(bbox[0]), int(bbox[1])
            xmin = int(center_x - width / 2)
            ymin = int(center_y - height / 2)
            xmax = int(center_x + width / 2)
            ymax = int(center_y + height / 2)
            
            # 已确认的水果用绿色，未确认的用黄色
            is_ready = detection_cache.is_ready(class_name)
            color = (0, 255, 0) if is_ready else (0, 255, 255)
            
            # 绘制边界框
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # 绘制中心点
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 显示类别和置信度
            label_text = f"{class_name} ({conf:.2f})"
            cv2.putText(frame, label_text, (xmin, ymin - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 绘制检测进度（右上角）
    y_offset = 60
    for class_name, cache_info in detection_cache.cache.items():
        count = cache_info['count']
        if count >= DETECTION_THRESHOLD:
            status = f"[OK] {class_name}: {count}"
            color = (0, 255, 0)
        else:
            status = f"[..] {class_name}: {count}/{DETECTION_THRESHOLD}"
            color = (0, 255, 255)
        
        cv2.putText(frame, status, (frame.shape[1] - 200, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
    
    # 绘制可抓取列表（左下角）- 只显示英文避免乱码
    if detection_cache.ready_fruits:
        y_offset = frame.shape[0] - 30 - len(detection_cache.ready_fruits) * 25
        cv2.putText(frame, "Press number to grab:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        for i, fruit in enumerate(detection_cache.ready_fruits):
            text = f"{i+1}: {fruit}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
    
    return frame


def open_camera_with_detection(camera, detector, agent, asr=None):
    """
    打开摄像头进行实时检测
    
    Args:
        camera: Camera 对象
        detector: YoloDetector 对象
        agent: TaskAgent 对象
        asr: 语音识别对象（可选）
    """
    print("=" * 50)
    print("🎥 摄像头检测模式")
    print("=" * 50)
    print("操作说明:")
    print("  'd' - 开始/停止连续检测")
    print("  's' - 单次检测")
    print("  '1-9' - 抓取对应编号的水果")
    print("  'r' - 清空检测缓存，重新检测")
    print("  'q' / ESC - 退出")
    print()
    print(f"检测确认阈值: {DETECTION_THRESHOLD} 次")
    print("水果确认后会显示编号，按对应数字键抓取")
    print("=" * 50)
    
    # 状态变量
    continuous_detect = False
    frame_count = 0
    last_detections = []
    detection_cache = DetectionCache()
    running = True
    
    print("[SYSTEM] 摄像头画面已打开，按 'd' 开始检测...")
    
    # 主循环
    while running:
        # 获取帧
        frame = camera.get_frame()
        if frame is None:
            print("[ERROR] 读取帧失败")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        display_frame = frame.copy()
        
        # 连续检测模式 - 每2帧检测一次
        if continuous_detect and frame_count % 2 == 0:
            try:
                detections = detector.detect(frame)
                last_detections = detections
                detection_cache.update(detections)
            except Exception as e:
                print(f"[ERROR] 检测失败: {e}")
        
        # 绘制检测结果
        display_frame = draw_detections(display_frame, last_detections, detection_cache)
        
        # 显示状态栏（英文避免乱码）
        status_text = f"Detect: {'ON' if continuous_detect else 'OFF'} | 'd'=detect 's'=single 'r'=reset 'q'=quit"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 显示画面
        cv2.imshow('Camera Detection', display_frame)
        
        # 处理按键
        key = cv2.waitKey(30) & 0xFF
        
        # 退出
        if key == ord('q') or key == 27:
            print("[SYSTEM] 退出摄像头模式")
            running = False
            break
        
        # 开始/停止检测
        elif key == ord('d'):
            continuous_detect = not continuous_detect
            if continuous_detect:
                detection_cache.clear()
                print("[SYSTEM] ▶ 开始连续检测...")
            else:
                print("[SYSTEM] ⏸ 停止连续检测")
                if detection_cache.ready_fruits:
                    detection_cache._print_ready_list()
        
        # 单次检测
        elif key == ord('s'):
            print("[SYSTEM] 执行单次检测...")
            try:
                detections = detector.detect(frame)
                last_detections = detections
                detection_cache.update(detections)
                if detections:
                    detected_classes = [d['class'] for d in detections]
                    print(f"[SYSTEM] 检测到: {set(detected_classes)}")
                else:
                    print("[SYSTEM] 未检测到任何物体")
            except Exception as e:
                print(f"[ERROR] 检测失败: {e}")
        
        # 重置检测缓存
        elif key == ord('r'):
            detection_cache.clear()
            last_detections = []
            print("[SYSTEM] 🔄 检测缓存已清空")
        
        # 数字键 1-9 抓取对应水果
        elif ord('1') <= key <= ord('9'):
            fruit_index = key - ord('1')  # 0-8
            print(f"[DEBUG] 按下数字键 {fruit_index+1}, ready_fruits={detection_cache.ready_fruits}")
            fruit_english = detection_cache.get_ready_fruit_by_index(fruit_index)
            
            if fruit_english:
                fruit_chinese = FRUIT_MAP_REVERSE.get(fruit_english, fruit_english)
                position = detection_cache.get_position(fruit_english)
                
                if position:
                    print(f"\n[SYSTEM] 🎯 抓取 {fruit_index+1}: {fruit_english} ({fruit_chinese})")
                    print(f"[SYSTEM] 位置: ({position[0]:.1f}, {position[1]:.1f})")
                    
                    # 检查机械臂是否初始化
                    if agent.arm is None:
                        print("[ERROR] 机械臂未初始化！请检查机械臂连接。")
                        continue
                    
                    # 停止检测
                    continuous_detect = False
                    cv2.destroyWindow('Camera Detection')
                    
                    # 执行抓取放置（一体化流程）
                    try:
                        success = agent.pick_and_place(position, fruit_english)
                        if success:
                            # 复位到HOME
                            agent.reset_arm()
                            print(f"\n[SYSTEM] ✅ {fruit_chinese} 完成！")
                        else:
                            print(f"[ERROR] 抓取失败")
                    except Exception as e:
                        print(f"[ERROR] 抓取过程出错: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # 抓取完成后清空缓存，准备下一轮检测
                    detection_cache.clear()
                    last_detections = []
                    print("\n[SYSTEM] 按 'd' 继续检测，按 'q' 退出")
            else:
                print(f"[WARN] 编号 {fruit_index+1} 无效，当前可抓取: {len(detection_cache.ready_fruits)} 个")
    
    # 清理
    cv2.destroyAllWindows()
    print("[SYSTEM] 摄像头已关闭")

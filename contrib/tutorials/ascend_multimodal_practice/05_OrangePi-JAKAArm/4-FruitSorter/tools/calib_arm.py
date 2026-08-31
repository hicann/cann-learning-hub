# file: calib_arm.py
import sys
import os
import time
import csv


def parse_direction_input(direction):
    """解析方向输入字符串（如 x15z-10y30），返回坐标列表 [dx, dy, dz]"""
    x_y_z = {"x": 0, "y": 1, "z": 2}
    direc = ["0", "0", "0"]
    now_i = 0
    
    for char in direction:
        if char in x_y_z:
            now_i = x_y_z[char]
        else:
            if direc[now_i][0] == "0":
                direc[now_i] = direc[now_i][1:]
            direc[now_i] += char
    
    return list(map(float, direc))


def refer_move(robot, distances):
    """相对移动机械臂"""
    try:
        now_pos = robot.get_tcp_position()[1]
        now_pos2 = robot.get_joint_position()[1]
        
        # 计算新位置
        new_pos = now_pos.copy()
        new_pos[0] += distances[0]
        new_pos[1] += distances[1]
        new_pos[2] += distances[2]
        
        way_to_new = robot.kine_inverse(now_pos2, new_pos)[1]
        robot.joint_move(way_to_new, 0, True, 3)  # ABS=0
        print(f"移动完成: dx={distances[0]}, dy={distances[1]}, dz={distances[2]}")
    except Exception as e:
        print(f"无法移动到该位置: {e}")


def run_arm_calib(arm):
    """
    机械臂九点标定 - 支持键盘输入坐标移动
    arm: JakaArm对象
    """
    print("[ARM_CALIB] Starting arm 9-point calibration")
    print("=" * 60)
    
    # 确保在项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        import jkrc
    except ImportError as e:
        print(f"[ERROR] Failed to import jkrc: {e}")
        print("Please install jkrc library")
        return
    
    print("=== 机械臂九点标定坐标采集 ===")
    print("操作说明：")
    print("  方式1: 手动拖动机械臂到目标点")
    print("  方式2: 输入坐标移动，如: x30 或 x10y-5z20")
    print()
    print("命令说明：")
    print("  x30      - X轴正向移动30mm")
    print("  x-20     - X轴负向移动20mm")
    print("  x10y5z-3 - 同时移动多个轴")
    print("  p        - 记录当前点")
    print("  pos      - 显示当前位置")
    print("  quit     - 退出标定")
    print()
    print("按摄像头九点顺序记录！（1-2-3, 4-5-6, 7-8-9）")
    print("=" * 60)

    points = []
    recorded_count = 0
    
    try:
        # 使用传入的arm对象（已经初始化）
        robot = arm.robot
        
        print(f"\n当前已记录 0/9 个点，准备记录第1个点")
        
        while recorded_count < 9:
            try:
                cmd = input(f"[第{recorded_count+1}点/共9点] 输入命令: ").strip().lower()
                
                if cmd == 'quit' or cmd == 'q':
                    print("用户退出标定")
                    break
                    
                elif cmd == 'p' or cmd == 'point' or cmd == '':
                    # 记录当前点
                    ret1 = robot.get_tcp_position()
                    if ret1[0] == 0:
                        x, y = ret1[1][0], ret1[1][1]
                        print(f"✅ 第 {recorded_count+1} 点: [{x:.3f}, {y:.3f}]")
                        points.append([x, y])
                        recorded_count += 1
                        
                        if recorded_count < 9:
                            print(f"进度: {recorded_count}/9，准备记录第{recorded_count+1}个点...")
                        else:
                            print("✅ 已记录全部9个点！")
                    else:
                        print("❌ 获取TCP位置失败")
                        
                elif cmd == 'pos':
                    # 显示当前位置
                    ret1 = robot.get_tcp_position()
                    if ret1[0] == 0:
                        pos = ret1[1]
                        print(f"当前TCP位置: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
                    else:
                        print("获取位置失败")
                        
                elif cmd == 'hand':
                    # 手动模式提示
                    print("请手动拖动机械臂到目标位置，然后输入 'p' 记录")
                    
                else:
                    # 尝试解析为移动命令
                    try:
                        distances = parse_direction_input(cmd)
                        if any(d != 0 for d in distances):
                            refer_move(robot, distances)
                        else:
                            print("无效命令，请输入如 x30 或 p")
                    except Exception as e:
                        print(f"命令格式错误: {e}")
                        print("示例: x30, y-10, x5y10z-3, p(记录点)")
                    
            except KeyboardInterrupt:
                print("\n用户中断标定")
                break
            except Exception as e:
                print(f"错误: {e}")
                time.sleep(0.5)
        
    except Exception as e:
        print(f"机械臂连接错误: {e}")
        return
    
    if len(points) == 9:
        # 确保data目录存在
        os.makedirs("data", exist_ok=True)
        
        # 保存到CSV
        csv_path = "data/arm_point.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(points)
        
        print(f"\n[SUCCESS] 机械臂九点坐标已保存到 {csv_path}")
        print("点坐标:")
        for i, (x, y) in enumerate(points):
            row = i // 3 + 1
            col = i % 3 + 1
            print(f"  点{i+1} (行{row},列{col}): [{x:.3f}, {y:.3f}]")
        
        # 验证文件
        if os.path.exists(csv_path):
            print(f"文件验证: {csv_path} 存在，大小: {os.path.getsize(csv_path)} bytes")
        else:
            print("警告: 文件保存失败")
            
    else:
        print(f"\n[WARNING] 只采集了 {len(points)} 个点，未完成标定！")
        if len(points) > 0:
            # 保存已有的点
            os.makedirs("data", exist_ok=True)
            csv_path = "data/arm_point_partial.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(points)
            print(f"部分坐标已保存到 {csv_path}")
    
    print("=" * 60)
    print("机械臂标定程序结束")
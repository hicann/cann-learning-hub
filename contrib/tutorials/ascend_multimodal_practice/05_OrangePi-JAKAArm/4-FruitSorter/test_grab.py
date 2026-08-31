#!/usr/bin/env python3
"""
测试抓取脚本 - 最小化测试
直接使用关节移动，参考 5.move_with_gripper.py
"""
import jkrc
import time
import wiringpi
import numpy as np
import csv
import os

ABS = 0  # 绝对位置

# 夹爪控制
GRIPPER_PIN = 19
DUTY_OPEN = 21
DUTY_CLOSE = 30

def init_gripper():
    wiringpi.wiringPiSetup()
    wiringpi.pinMode(GRIPPER_PIN, 1)
    wiringpi.softPwmCreate(GRIPPER_PIN, 10, 100)

def open_gripper():
    print("[GRIPPER] 打开夹爪")
    wiringpi.softPwmWrite(GRIPPER_PIN, DUTY_OPEN)

def close_gripper():
    print("[GRIPPER] 闭合夹爪")
    wiringpi.softPwmWrite(GRIPPER_PIN, DUTY_CLOSE)

def load_calibration_matrix():
    """加载手眼标定矩阵"""
    matrix_file = 'data/config_relation_matrix.csv'
    if os.path.exists(matrix_file):
        with open(matrix_file, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]  # 过滤空行
            matrix = np.array([[float(x) for x in row] for row in rows])
            print(f"[CALIB] 标定矩阵 shape: {matrix.shape}")
            print(f"[CALIB] 标定矩阵:\n{matrix}")
            return matrix
    else:
        print("[CALIB] 标定矩阵文件不存在！")
        return None

def pixel_to_arm(pixel_x, pixel_y, matrix):
    """像素坐标转机械臂坐标（3x2矩阵）"""
    # matrix是3x2: [[a, b], [c, d], [e, f]]
    # x = px*a + py*c + e
    # y = px*b + py*d + f
    x = pixel_x * matrix[0, 0] + pixel_y * matrix[1, 0] + matrix[2, 0]
    y = pixel_x * matrix[0, 1] + pixel_y * matrix[1, 1] + matrix[2, 1]
    return x, y

def move_to_xyz(rc, x, y, z, speed=0.25):
    """移动到指定XYZ位置（保持当前姿态）"""
    # 获取当前关节和TCP
    ret = rc.get_joint_position()
    if ret[0] != 0:
        print("获取关节失败")
        return False
    current_joint = ret[1]
    
    ret = rc.get_tcp_position()
    if ret[0] != 0:
        print("获取TCP失败")
        return False
    current_tcp = ret[1]
    
    # 保持当前姿态
    target_pose = [x, y, z, current_tcp[3], current_tcp[4], current_tcp[5]]
    print(f"目标: ({x:.1f}, {y:.1f}, {z:.1f}), 姿态: ({current_tcp[3]:.2f}, {current_tcp[4]:.2f}, {current_tcp[5]:.2f})")
    
    # 逆解
    ret = rc.kine_inverse(current_joint, target_pose)
    if ret[0] != 0:
        print(f"逆解失败: {ret[0]}")
        return False
    
    # 关节移动
    ret = rc.joint_move(ret[1], ABS, True, speed)
    if ret[0] != 0:
        print(f"移动失败: {ret[0]}")
        return False
    
    print("移动成功")
    return True

def main():
    print("=" * 50)
    print("测试抓取脚本（最小化）")
    print("=" * 50)
    
    # 初始化夹爪
    init_gripper()
    
    # 连接机械臂
    print("\n[1] 连接机械臂...")
    rc = jkrc.RC("10.5.5.100")
    ret = rc.login()
    print(f"    Login: {ret}")
    if ret[0] != 0:
        print("登录失败！")
        return
    
    ret = rc.power_on()
    print(f"    Power on: {ret}")
    
    ret = rc.enable_robot()
    print(f"    Enable: {ret}")
    
    # 获取当前位置
    print("\n[2] 当前位置...")
    ret = rc.get_tcp_position()
    if ret[0] == 0:
        pos = ret[1]
        print(f"    TCP位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        print(f"    姿态: ({pos[3]:.2f}, {pos[4]:.2f}, {pos[5]:.2f})")
    
    ret = rc.get_joint_position()
    if ret[0] == 0:
        joint = ret[1]
        print(f"    关节角度: {[f'{j:.3f}' for j in joint]}")
    
    # 加载标定矩阵
    print("\n[3] 加载标定矩阵...")
    matrix = load_calibration_matrix()
    
    # 测试坐标转换
    print("\n[4] 测试坐标转换...")
    pixel_x, pixel_y = 401.5, 236.5  # 柿子位置
    if matrix is not None:
        arm_x, arm_y = pixel_to_arm(pixel_x, pixel_y, matrix)
        print(f"    像素: ({pixel_x}, {pixel_y})")
        print(f"    机械臂: ({arm_x:.1f}, {arm_y:.1f})")
    
    # 定义关节位置（参考5.move_with_gripper.py）
    # 这些是预设的安全位置，需要根据实际情况调整
    print("\n[5] 定义移动位置...")
    
    # HOME位置（关节角度）
    home_joint = [0, 0, 1.57, 0, 1.57, 0]  # 大约的HOME位置
    
    # 获取当前关节作为起点
    ret = rc.get_joint_position()
    current_joint = ret[1] if ret[0] == 0 else home_joint
    
    print("\n请选择测试模式：")
    print("  1. 测试夹爪开合")
    print("  2. 测试关节移动（使用5.move_with_gripper.py的位置）")
    print("  3. 测试笛卡尔移动（当前位置附近）")
    print("  4. 手动示教抓取测试")
    print("  5. 测试抓取（选择像素位置）")
    print("  6. 显示当前位置")
    print("  7. 移动到工作区域（使用预设关节角度）")
    print("  q. 退出")
    
    while True:
        choice = input("\n请选择: ").strip()
        
        if choice == '1':
            print("\n测试夹爪...")
            open_gripper()
            time.sleep(2)
            close_gripper()
            time.sleep(2)
            open_gripper()
            print("夹爪测试完成")
            
        elif choice == '2':
            print("\n测试关节移动...")
            # 使用参考代码中的位置
            pos_1 = [1.5272618380704825, -0.06282712377329037, 1.4938506502674533, -0.03254433684272177, 1.6331557989724899, 1.8475597429984356]
            pos_2 = [1.5353991266175793, 0.5577759504703221, 0.8695564348272197, -0.03253235261805742, 1.6500415715245655, 1.8475597429984356]
            
            print(f"移动到pos_1...")
            ret = rc.joint_move(pos_1, 0, True, 0.25)
            print(f"结果: {ret}")
            time.sleep(1)
            
            print(f"移动到pos_2...")
            ret = rc.joint_move(pos_2, 0, True, 0.25)
            print(f"结果: {ret}")
            
        elif choice == '3':
            print("\n测试笛卡尔移动（在当前位置附近）...")
            ret = rc.get_tcp_position()
            if ret[0] == 0:
                pos = ret[1]
                # 在当前位置上方50mm
                target = [pos[0], pos[1], pos[2] + 50, pos[3], pos[4], pos[5]]
                print(f"当前: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                print(f"目标: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
                
                ret = rc.linear_move(target, 0, True, 50)
                print(f"linear_move结果: {ret}")
                
                if ret[0] != 0:
                    print("linear_move失败，尝试使用逆解+关节移动...")
                    ret_joint = rc.get_joint_position()
                    if ret_joint[0] == 0:
                        ret_ik = rc.kine_inverse(ret_joint[1], target)
                        print(f"逆解结果: {ret_ik}")
                        if ret_ik[0] == 0:
                            ret_move = rc.joint_move(ret_ik[1], 0, True, 0.25)
                            print(f"joint_move结果: {ret_move}")
            
        elif choice == '4':
            print("\n完整抓取测试...")
            print("请手动示教以下位置的关节角度：")
            print("  - 物体上方位置")
            print("  - 抓取位置")
            print("  - 放置位置")
            
            # 获取当前位置作为参考
            ret = rc.get_joint_position()
            if ret[0] == 0:
                print(f"\n当前关节角度: {ret[1]}")
            
            input("将机械臂移动到【物体上方】，按回车记录...")
            ret = rc.get_joint_position()
            pos_above = ret[1] if ret[0] == 0 else None
            print(f"记录: {pos_above}")
            
            input("将机械臂移动到【抓取位置】，按回车记录...")
            ret = rc.get_joint_position()
            pos_grab = ret[1] if ret[0] == 0 else None
            print(f"记录: {pos_grab}")
            
            input("将机械臂移动到【放置位置】，按回车记录...")
            ret = rc.get_joint_position()
            pos_place = ret[1] if ret[0] == 0 else None
            print(f"记录: {pos_place}")
            
            if pos_above and pos_grab and pos_place:
                print("\n开始执行抓取...")
                
                # 1. 打开夹爪
                open_gripper()
                time.sleep(1)
                
                # 2. 移动到上方
                print("移动到上方...")
                rc.joint_move(pos_above, 0, True, 0.25)
                time.sleep(0.5)
                
                # 3. 下降抓取
                print("下降抓取...")
                rc.joint_move(pos_grab, 0, True, 0.25)
                time.sleep(0.5)
                
                # 4. 闭合夹爪
                close_gripper()
                time.sleep(2)
                
                # 5. 提起
                print("提起...")
                rc.joint_move(pos_above, 0, True, 0.25)
                time.sleep(0.5)
                
                # 6. 移动到放置位置
                print("移动到放置位置...")
                rc.joint_move(pos_place, 0, True, 0.25)
                time.sleep(0.5)
                
                # 7. 打开夹爪
                open_gripper()
                time.sleep(1)
                
                print("抓取测试完成！")
            
        elif choice == '5':
            print("\n测试抓取...")
            if matrix is None:
                print("标定矩阵未加载！")
                continue
            
            # 使用标定范围内的中心点测试（更可靠）
            # 标定范围：像素X=[223-378], Y=[87-240]
            # 中心点约 (300, 160)
            print("选择测试位置：")
            print("  a. 标定中心点 (300, 160) - 推荐先测试")
            print("  b. 柿子位置 (401.5, 236.5) - 可能超出范围")
            print("  c. 手动输入像素坐标")
            sub = input("选择: ").strip().lower()
            
            if sub == 'a':
                pixel_x, pixel_y = 300, 160
            elif sub == 'b':
                pixel_x, pixel_y = 401.5, 236.5
            elif sub == 'c':
                try:
                    pixel_x = float(input("像素X: "))
                    pixel_y = float(input("像素Y: "))
                except:
                    print("输入无效")
                    continue
            else:
                continue
            
            arm_x, arm_y = pixel_to_arm(pixel_x, pixel_y, matrix)
            print(f"像素: ({pixel_x}, {pixel_y}) -> 机械臂: ({arm_x:.1f}, {arm_y:.1f})")
            
            # 高度参数
            safe_z = 250
            grab_z = 180
            place_y_offset = 100  # 放置时Y方向偏移
            
            print(f"\n将执行以下动作：")
            print(f"  1. 打开夹爪")
            print(f"  2. 移动到 ({arm_x:.1f}, {arm_y:.1f}, {safe_z})")
            print(f"  3. 下降到 ({arm_x:.1f}, {arm_y:.1f}, {grab_z})")
            print(f"  4. 闭合夹爪")
            print(f"  5. 提起到 ({arm_x:.1f}, {arm_y:.1f}, {safe_z})")
            print(f"  6. 平移到 ({arm_x:.1f}, {arm_y + place_y_offset:.1f}, {safe_z})")
            print(f"  7. 打开夹爪")
            
            confirm = input("\n确认执行？(y/n): ").strip().lower()
            if confirm != 'y':
                print("取消")
                continue
            
            # 执行抓取
            print("\n[1] 打开夹爪")
            open_gripper()
            time.sleep(1.5)
            
            print(f"\n[2] 移动到物体上方 ({arm_x:.1f}, {arm_y:.1f}, {safe_z})")
            if not move_to_xyz(rc, arm_x, arm_y, safe_z):
                print("移动失败！")
                continue
            time.sleep(0.5)
            
            print(f"\n[3] 下降到抓取高度 ({arm_x:.1f}, {arm_y:.1f}, {grab_z})")
            if not move_to_xyz(rc, arm_x, arm_y, grab_z):
                print("下降失败！")
                continue
            time.sleep(0.5)
            
            print("\n[4] 闭合夹爪")
            close_gripper()
            time.sleep(2.5)
            
            print(f"\n[5] 提起 ({arm_x:.1f}, {arm_y:.1f}, {safe_z})")
            if not move_to_xyz(rc, arm_x, arm_y, safe_z):
                print("提起失败！")
            time.sleep(0.5)
            
            print(f"\n[6] 平移到放置位置 ({arm_x:.1f}, {arm_y + place_y_offset:.1f}, {safe_z})")
            if not move_to_xyz(rc, arm_x, arm_y + place_y_offset, safe_z):
                print("平移失败！")
            time.sleep(0.5)
            
            print("\n[7] 打开夹爪释放")
            open_gripper()
            time.sleep(1.5)
            
            print("\n抓取测试完成！")
            
        elif choice == '6':
            print("\n当前位置：")
            ret = rc.get_tcp_position()
            if ret[0] == 0:
                pos = ret[1]
                print(f"  TCP: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                print(f"  姿态: ({pos[3]:.4f}, {pos[4]:.4f}, {pos[5]:.4f})")
            ret = rc.get_joint_position()
            if ret[0] == 0:
                print(f"  关节: {[f'{j:.4f}' for j in ret[1]]}")
        
        elif choice == '7':
            print("\n移动到工作区域...")
            print("使用5.move_with_gripper.py中的pos_1位置")
            # 这是参考代码中的第一个位置，应该在工作区域内
            work_pos = [1.5272618380704825, -0.06282712377329037, 1.4938506502674533, 
                       -0.03254433684272177, 1.6331557989724899, 1.8475597429984356]
            
            print(f"目标关节角度: {[f'{j:.3f}' for j in work_pos]}")
            confirm = input("确认移动？(y/n): ").strip().lower()
            if confirm == 'y':
                ret = rc.joint_move(work_pos, ABS, True, 0.25)
                print(f"移动结果: {ret}")
                if ret[0] == 0:
                    # 显示新位置
                    ret = rc.get_tcp_position()
                    if ret[0] == 0:
                        pos = ret[1]
                        print(f"新TCP位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            
        elif choice == 'q':
            break
        else:
            print("无效选择")
    
    print("\n测试结束")

if __name__ == "__main__":
    main()

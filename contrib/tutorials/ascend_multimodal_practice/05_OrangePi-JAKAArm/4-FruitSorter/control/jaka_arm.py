# control/jaka_arm.py
import jkrc
import numpy as np
import os
import csv
import time
from collections import deque

ABS = 0

class JakaArm:
    def __init__(self, ip="10.5.5.100", data_dir="data"):
        self.ip = ip
        self.data_dir = data_dir
        
        # 抓取参数
        self.safe_z = 250           # 安全高度
        self.grasp_height = 180     # 抓取高度（调高避免怼到桌面）
        self.approach_height = 250  # 接近高度
        self.place_height = 220     # 放置高度
        
        # HOME位置
        self.home_x = 182.615
        self.home_y = -6.000
        self.home_z = 250
        
        # 路径记忆
        self.path_to_target = deque(maxlen=20)  # 记录到目标的路径
        self.is_returning = False  # 是否在返回途中
        
        # 必须加载手眼矩阵
        self.eyehand_matrix = self._load_eyehand_matrix()
        if self.eyehand_matrix is None:
            print("[JAKA] 警告：未找到手眼标定矩阵，将使用简单映射")
            print("[JAKA] 建议执行：1.摄像头标定 2.机械臂标定 3.手眼标定")
            # 创建默认的简单映射矩阵
            self.eyehand_matrix = np.array([
                [0.3, 0.0],   # x = px*0.3
                [0.0, 0.3],   # y = py*0.3
                [150, 150]    # 偏移
            ], dtype=np.float32)
        
        # 连接机械臂
        self.robot = jkrc.RC(ip)
        self._init_robot()
        
        # 启动后直接移动到HOME位置
        self._go_to_home_position()
        
        print(f"[JAKA] 初始化完成")
        print(f"[JAKA] HOME位置: ({self.home_x:.1f}, {self.home_y:.1f}, {self.home_z}mm)")
        print(f"[JAKA] 安全高度: {self.safe_z}mm")

    def _init_robot(self):
        """初始化机械臂"""
        ret = self.robot.login()
        if ret[0] != 0:
            raise RuntimeError("JAKA login failed")

        ret = self.robot.power_on()
        if ret[0] != 0:
            raise RuntimeError("Power on failed")
            
        ret = self.robot.enable_robot()
        if ret[0] != 0:
            raise RuntimeError("Enable robot failed")
            
        print("[JAKA] 机械臂就绪")
        
        # 显示当前位置
        self._show_current_position()

    def _go_to_home_position(self):
        """移动到HOME位置"""
        print("[JAKA] ========== 移动到HOME位置 ==========")
        
        try:
            # 获取当前位置
            ret = self.robot.get_tcp_position()
            if ret[0] != 0:
                print("[JAKA] 获取当前位置失败")
                return
            current = ret[1]
            print(f"[JAKA] 当前位置: ({current[0]:.1f}, {current[1]:.1f}, {current[2]:.1f})")
            
            # 获取当前关节角度
            ret_joint = self.robot.get_joint_position()
            if ret_joint[0] != 0:
                print("[JAKA] 获取关节角度失败")
                return
            joint = ret_joint[1]
            
            # 1. 直接上升到HOME Z高度（保持当前XY）
            print(f"[JAKA] 步骤1: 上升到Z={self.home_z}mm")
            pose1 = [current[0], current[1], self.home_z, -np.pi, 0, -np.pi]
            print(f"[JAKA] 目标姿态1: {pose1}")
            ret = self.robot.kine_inverse(joint, pose1)
            print(f"[JAKA] 逆解结果: {ret[0]}")
            if ret[0] == 0:
                print(f"[JAKA] 执行关节移动...")
                move_ret = self.robot.joint_move(ret[1], ABS, True, 5)
                print(f"[JAKA] 移动结果: {move_ret}")
            else:
                print(f"[JAKA] 上升逆解失败，错误码: {ret[0]}")
            
            # 2. 平移到HOME XY位置
            print(f"[JAKA] 步骤2: 平移到XY=({self.home_x:.1f}, {self.home_y:.1f})")
            ret_joint = self.robot.get_joint_position()
            if ret_joint[0] == 0:
                joint = ret_joint[1]
                pose2 = [self.home_x, self.home_y, self.home_z, -np.pi, 0, -np.pi]
                print(f"[JAKA] 目标姿态2: {pose2}")
                ret = self.robot.kine_inverse(joint, pose2)
                print(f"[JAKA] 逆解结果: {ret[0]}")
                if ret[0] == 0:
                    print(f"[JAKA] 执行关节移动...")
                    move_ret = self.robot.joint_move(ret[1], ABS, True, 8)
                    print(f"[JAKA] 移动结果: {move_ret}")
                else:
                    print(f"[JAKA] 平移逆解失败，错误码: {ret[0]}")
            
            # 3. 开启夹爪
            print("[JAKA] 步骤3: 开启夹爪")
            try:
                from control.gripper_pwm import open_gripper
                open_gripper()
            except:
                print("[JAKA] 夹爪控制失败，继续...")
            
            # 显示最终位置
            ret = self.robot.get_tcp_position()
            if ret[0] == 0:
                final = ret[1]
                print(f"[JAKA] 最终位置: ({final[0]:.1f}, {final[1]:.1f}, {final[2]:.1f})")
                
            print("[JAKA] ========== HOME位置设置完成 ==========")
            
        except Exception as e:
            print(f"[JAKA] HOME位置设置失败: {e}")
            import traceback
            traceback.print_exc()

    def _load_eyehand_matrix(self):
        """加载手眼标定矩阵"""
        path = os.path.join(self.data_dir, "config_relation_matrix.csv")
        if not os.path.exists(path):
            print(f"[JAKA] 错误：未找到标定矩阵 {path}")
            return None
            
        try:
            with open(path, "r") as f:
                reader = csv.reader(f)
                matrix_data = list(reader)
                if matrix_data:
                    matrix = np.array(matrix_data, dtype=np.float32)
                    print(f"[JAKA] 手眼矩阵已加载")
                    return matrix
        except Exception as e:
            print(f"[JAKA] 加载矩阵失败: {e}")
            return None

    def pixel_to_arm_xy(self, px, py):
        """像素坐标转换为机械臂坐标"""
        if self.eyehand_matrix is None:
            raise RuntimeError("没有手眼矩阵，无法转换坐标")
        
        # 标定范围检查（基于9点标定的范围）
        # 摄像头标定范围约：X=223-378, Y=87-240
        CAM_X_MIN, CAM_X_MAX = 200, 450
        CAM_Y_MIN, CAM_Y_MAX = 50, 300
        
        if not (CAM_X_MIN <= px <= CAM_X_MAX and CAM_Y_MIN <= py <= CAM_Y_MAX):
            print(f"[JAKA] ⚠️ 警告：像素坐标({px:.1f}, {py:.1f})超出标定范围")
            print(f"[JAKA] 标定范围：X=[{CAM_X_MIN}-{CAM_X_MAX}], Y=[{CAM_Y_MIN}-{CAM_Y_MAX}]")
            print(f"[JAKA] 转换结果可能不准确！")
            
        try:
            # 使用3x2格式矩阵
            if self.eyehand_matrix.shape[1] == 2:
                x = px * self.eyehand_matrix[0, 0] + \
                    py * self.eyehand_matrix[1, 0] + \
                    self.eyehand_matrix[2, 0]
                y = px * self.eyehand_matrix[0, 1] + \
                    py * self.eyehand_matrix[1, 1] + \
                    self.eyehand_matrix[2, 1]
            else:
                # 3x3格式
                p_h = np.array([float(px), float(py), 1.0])
                result = self.eyehand_matrix @ p_h
                x, y = float(result[0]), float(result[1])
            
            # 机械臂工作范围检查（基于标定点范围）
            ARM_X_MIN, ARM_X_MAX = -150, 100
            ARM_Y_MIN, ARM_Y_MAX = -450, -200
            
            if not (ARM_X_MIN <= x <= ARM_X_MAX and ARM_Y_MIN <= y <= ARM_Y_MAX):
                print(f"[JAKA] ⚠️ 警告：机械臂坐标({x:.1f}, {y:.1f})可能超出工作范围")
                print(f"[JAKA] 建议范围：X=[{ARM_X_MIN}-{ARM_X_MAX}], Y=[{ARM_Y_MIN}-{ARM_Y_MAX}]")
            
            print(f"[JAKA] 像素({px:.1f}, {py:.1f}) -> 机械臂({x:.1f}, {y:.1f})")
            return x, y
            
        except Exception as e:
            print(f"[JAKA] 坐标转换失败: {e}")
            raise

    def _record_path_point(self):
        """记录当前路径点"""
        try:
            pos = self.get_current_position()
            if pos:
                # 只记录XY位置，Z高度是固定的
                point = (pos[0], pos[1])
                if len(self.path_to_target) == 0 or self._distance(point, self.path_to_target[-1]) > 10.0:
                    # 只有当距离大于10mm时才记录，避免重复点
                    self.path_to_target.append(point)
                    print(f"[JAKA] 记录路径点: ({point[0]:.1f}, {point[1]:.1f})")
                    print(f"[JAKA] 当前路径点数量: {len(self.path_to_target)}")
        except:
            pass

    def _distance(self, point1, point2):
        """计算两点距离"""
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _move_to_xyz(self, x=None, y=None, z=None, speed=10, max_step=100.0, record_path=True):
        """内部移动方法"""
        try:
            # 获取当前位置
            ret = self.robot.get_tcp_position()
            if ret[0] != 0:
                print(f"[JAKA] 获取当前位置失败")
                return False
                
            current = ret[1]
            current_x = float(current[0])
            current_y = float(current[1])
            current_z = float(current[2])
            
            # 设置目标位置
            target_x = float(x) if x is not None else current_x
            target_y = float(y) if y is not None else current_y
            target_z = float(z) if z is not None else current_z
            
            # 计算总距离
            total_dist = np.sqrt((target_x-current_x)**2 + 
                               (target_y-current_y)**2 + 
                               (target_z-current_z)**2)
            
            print(f"[JAKA] 目标位置: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
            print(f"[JAKA] 当前位置: ({current_x:.1f}, {current_y:.1f}, {current_z:.1f})")
            print(f"[JAKA] 总距离: {total_dist:.1f}mm")
            
            # 如果不是在返回途中且需要记录路径，记录起点
            if record_path and not self.is_returning:
                self._record_path_point()
            
            # 如果距离超过最大步长，需要分步移动
            if total_dist > max_step:
                print(f"[JAKA] 距离过大 ({total_dist:.1f}mm > {max_step}mm)，需要分步移动")
                
                # 计算中间点（沿直线移动）
                ratio = max_step / total_dist
                mid_x = current_x + (target_x - current_x) * ratio
                mid_y = current_y + (target_y - current_y) * ratio
                mid_z = current_z + (target_z - current_z) * ratio
                
                print(f"[JAKA] 第一步：移动到中间点 ({mid_x:.1f}, {mid_y:.1f}, {mid_z:.1f})")
                
                # 先移动到中间点
                if not self._move_direct_to_point(mid_x, mid_y, mid_z, speed, record_path):
                    print(f"[JAKA] 移动到中间点失败")
                    return False
                
                # 如果不是在返回途中且需要记录路径，记录中间点
                if record_path and not self.is_returning:
                    self._record_path_point()
                
                # 再从中间点移动到目标点
                print(f"[JAKA] 第二步：从中间点移动到目标点")
                return self._move_direct_to_point(target_x, target_y, target_z, speed, record_path)
            else:
                # 直接移动
                return self._move_direct_to_point(target_x, target_y, target_z, speed, record_path)
                
        except Exception as e:
            print(f"[JAKA] 移动异常: {e}")
            return False

    def _move_direct_to_point(self, x, y, z, speed, record_path=True):
        """直接移动到指定点"""
        try:
            # 尝试多种姿态
            poses_to_try = [
                [x, y, z, -np.pi, 0, -np.pi],      # 姿态1
                [x, y, z, -np.pi, 0, 0],           # 姿态2
                [x, y, z, 0, 0, -np.pi],           # 姿态3
                [x, y, z, -np.pi/2, 0, -np.pi/2],  # 姿态4
                [x, y, z, np.pi, 0, 0],            # 姿态5
                [x, y, z, 0, -np.pi/2, 0],         # 姿态6
            ]
            
            # 获取当前关节角度
            ret = self.robot.get_joint_position()
            if ret[0] != 0:
                print(f"[JAKA] 获取关节角度失败")
                return False
            joint = ret[1]
            
            for i, pose in enumerate(poses_to_try):
                print(f"[JAKA] 尝试姿态 {i+1}")
                
                ret = self.robot.kine_inverse(joint, pose)
                if ret[0] == 0:
                    print(f"[JAKA] 姿态 {i+1} 逆解成功")
                    
                    # 执行关节运动
                    ret = self.robot.joint_move(ret[1], ABS, True, speed)
                    if ret[0] == 0:
                        # 如果不是在返回途中且需要记录路径，记录到达点
                        if record_path and not self.is_returning:
                            self._record_path_point()
                        
                        # 根据距离估算等待时间
                        dist = np.sqrt((x-float(joint[0]))**2 + (y-float(joint[1]))**2 + (z-float(joint[2]))**2)
                        est_time = max(0.5, dist / 100.0)
                        print(f"[JAKA] 移动中，预计耗时 {est_time:.1f} 秒")
                        return True
                    else:
                        print(f"[JAKA] 关节移动失败: {ret[0]}")
                        continue
                else:
                    print(f"[JAKA] 姿态 {i+1} 逆解失败: {ret[0]}")
            
            print("[JAKA] 所有姿态都失败")
            return False
            
        except Exception as e:
            print(f"[JAKA] 直接移动异常: {e}")
            return False

    def move_xyz(self, x=None, y=None, z=None, speed=10):
        """公共移动接口"""
        return self._move_to_xyz(x, y, z, speed)

    def move_to_target_above(self, x, y, z):
        """
        简化移动：直接在安全高度平移到目标XY上方
        轨迹：当前位置 → 安全高度 → 目标XY上方 → 下降到z高度
        """
        print(f"[JAKA] 移动到目标上方 ({x:.1f}, {y:.1f}, z={z:.1f})")
        
        # 1. 先到安全高度（垂直提升）
        print("[JAKA] 步骤1: 提升到安全高度")
        if not self._simple_move_z(self.safe_z):
            print("[JAKA] 提升失败")
            return False
        
        # 2. 在安全高度水平移动到目标XY
        print(f"[JAKA] 步骤2: 水平移动到目标 ({x:.1f}, {y:.1f})")
        if not self._simple_move_xy(x, y):
            print("[JAKA] 水平移动失败")
            return False
        
        # 3. 垂直下降到目标高度
        print(f"[JAKA] 步骤3: 下降到高度 {z:.1f}")
        if not self._simple_move_z(z):
            print("[JAKA] 下降失败")
            self._simple_move_z(self.safe_z)
            return False
        
        print(f"[JAKA] ✅ 已到达目标位置")
        return True
    
    def _simple_move_z(self, z, speed=0.25):
        """简单垂直移动（只改变Z）"""
        current = self.get_current_position()
        if current is None:
            return False
        return self.move_to_position(current[0], current[1], z, speed)
    
    def _simple_move_xy(self, x, y, speed=0.25):
        """简单水平移动（只改变XY，保持当前Z）"""
        current = self.get_current_position()
        if current is None:
            return False
        return self.move_to_position(x, y, current[2], speed)

    def move_to_position(self, x, y, z, speed=0.25):
        """
        移动到指定位置（使用当前姿态进行逆解+关节移动）
        """
        print(f"[JAKA] 移动到 ({x:.1f}, {y:.1f}, {z:.1f})")
        try:
            # 获取当前关节角度
            ret = self.robot.get_joint_position()
            if ret[0] != 0:
                print(f"[JAKA] 获取关节角度失败")
                return False
            current_joint = ret[1]
            
            # 获取当前TCP姿态（保持姿态不变）
            ret = self.robot.get_tcp_position()
            if ret[0] != 0:
                print(f"[JAKA] 获取TCP位置失败")
                return False
            current_tcp = ret[1]
            
            # 使用当前姿态，只改变位置
            rx, ry, rz = current_tcp[3], current_tcp[4], current_tcp[5]
            target_pose = [x, y, z, rx, ry, rz]
            
            print(f"[JAKA] 目标姿态: pos=({x:.1f}, {y:.1f}, {z:.1f}), rot=({rx:.2f}, {ry:.2f}, {rz:.2f})")
            
            # 逆解
            ret = self.robot.kine_inverse(current_joint, target_pose)
            if ret[0] == 0:
                target_joint = ret[1]
                print(f"[JAKA] 逆解成功，执行关节移动...")
                ret2 = self.robot.joint_move(target_joint, ABS, True, speed)
                if ret2[0] == 0:
                    print(f"[JAKA] ✓ 到达")
                    return True
                else:
                    print(f"[JAKA] ✗ 关节移动失败: {ret2[0]}")
            else:
                print(f"[JAKA] ✗ 逆解失败: {ret[0]}，目标可能超出工作范围")
            
            return False
            
        except Exception as e:
            print(f"[JAKA] 移动异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def move_safe_z(self):
        """垂直移动到安全高度"""
        print(f"[JAKA] 垂直提升到安全高度 {self.safe_z}mm")
        return self._simple_move_z(self.safe_z)

    def return_to_home(self):
        """
        返回HOME位置
        """
        print("[JAKA] 返回HOME位置")
        
        try:
            # 获取当前位置
            current = self.get_current_position()
            if current is None:
                return False
            
            # 1. 先提升到安全高度（保持当前XY）
            print("[JAKA] 步骤1: 提升到安全高度")
            self.move_to_position(current[0], current[1], self.safe_z)
            
            # 2. 移动到HOME位置
            print(f"[JAKA] 步骤2: 移动到HOME ({self.home_x:.1f}, {self.home_y:.1f}, {self.home_z})")
            self.move_to_position(self.home_x, self.home_y, self.home_z)
            
            print("[JAKA] ✅ 已返回HOME位置")
            return True
            
        except Exception as e:
            print(f"[JAKA] 返回HOME异常: {e}")
            return False

    def _show_current_position(self):
        """显示当前位置"""
        try:
            pos = self.robot.get_tcp_position()[1]
            print(f"[JAKA] 当前位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        except:
            pass

    def get_current_position(self):
        """获取当前位置"""
        try:
            ret = self.robot.get_tcp_position()
            if ret[0] == 0:
                return [float(ret[1][0]), float(ret[1][1]), float(ret[1][2])]
        except:
            pass
        return None

    def record_point(self):
        """记录当前点"""
        try:
            ret = self.robot.get_tcp_position()
            if ret[0] == 0:
                pos = ret[1]
                print(f"[JAKA] 记录点: ({pos[0]:.1f}, {pos[1]:.1f})")
                return [float(pos[0]), float(pos[1])]
        except Exception as e:
            print(f"[JAKA] 记录点失败: {e}")
        return None


if __name__ == "__main__":
    print("测试JakaArm")
    try:
        arm = JakaArm()
        print("初始化成功")
        
        # 测试移动和返回
        print("\n测试移动和原路返回:")
        print("1. 移动到目标位置")
        success = arm.move_to_target_above(100, 100, 250)
        print(f"移动结果: {'成功' if success else '失败'}")
        
        if success:
            print("\n2. 原路返回HOME")
            success = arm.return_to_home()
            print(f"返回结果: {'成功' if success else '失败'}")
        
    except Exception as e:
        print(f"测试失败: {e}")
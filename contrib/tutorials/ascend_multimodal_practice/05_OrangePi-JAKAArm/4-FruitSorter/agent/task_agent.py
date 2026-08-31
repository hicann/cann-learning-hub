# agent/task_agent.py
from tools.eyehand_calib import run_eyehand_calib
from tools.calib_camera_auto import run_camera_calib
from tools.calib_arm import run_arm_calib
from control.gripper_pwm import open_gripper, close_gripper
import csv
import time
import os

ABS = 0  # 绝对位置模式

class TaskAgent:
    def __init__(self, camera, detector, arm, gripper=None):
        self.camera = camera
        self.detector = detector
        self.arm = arm
        self.gripper = gripper
        
        # 水果分类映射
        self.fruit_category_map = {
            "cavocado": 1, "avocado": 1,
            "lemon": 2,
            "pear": 3,
            "mango": 4,
            "persimmon": 5
        }
        
        # 抓取参数
        self.safe_z = 250           # 安全高度
        self.grasp_height = 170    # 抓取高度（降低20mm，参考5.move_with_gripper.py）
        self.place_height = 220     # 放置高度
        
        # 固定放置位置的关节角度（来自5.move_with_gripper.py的pos_4）
        self.place_joint_pos = [-0.058011400530368226, 0.2004662921025842, 1.3684597076043092, 
                                0.005553513365262088, 1.5503807592157914, 1.8257005172106524]
        
        # 目标区域坐标
        self.target_area = self._load_target_area()
        
        print(f"[AGENT] 初始化完成, 抓取高度: {self.grasp_height}mm")

    def _load_target_area(self):
        path = "data/target_area.csv"
        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.reader(f)
                positions = [[float(x) for x in row] for row in reader]
                return positions if len(positions) >= 5 else self._default_target_area()
        return self._default_target_area()

    def _default_target_area(self):
        return [[100, 100, 250], [200, 100, 250], [100, 200, 250], [200, 200, 250], [300, 200, 250]]

    def get_basket_for_fruit(self, fruit_name):
        return self.fruit_category_map.get(fruit_name.lower(), 1)

    def pick_and_place(self, position, fruit_name="fruit"):
        """
        简洁抓取放置流程（参考5.move_with_gripper.py）：
        1. 移动到物体上方 + 打开夹爪
        2. 下降抓取
        3. 闭合夹爪
        4. 提起 + 移动到固定放置位置（使用关节角度）
        5. 打开夹爪释放
        """
        print(f"\n[AGENT] ========== 抓取 {fruit_name} ==========")
        
        try:
            if self.arm is None:
                print("[AGENT] 错误：机械臂未初始化！")
                return False
            
            # 转换像素坐标到机械臂坐标
            pixel_x = float(position[0])
            pixel_y = float(position[1])
            arm_x, arm_y = self.arm.pixel_to_arm_xy(pixel_x, pixel_y)
            print(f"[AGENT] 像素: ({pixel_x:.0f}, {pixel_y:.0f}) → 机械臂: ({arm_x:.1f}, {arm_y:.1f})")
            
            # 1. 移动到物体上方 + 打开夹爪
            print(f"[1] 移动到物体上方")
            self.arm.move_to_position(arm_x, arm_y, self.safe_z)
            open_gripper()
            time.sleep(3)
            
            # 2. 下降抓取
            print(f"[2] 下降抓取")
            self.arm.move_to_position(arm_x, arm_y, self.grasp_height)
            time.sleep(1)
            
            # 3. 闭合夹爪
            print("[3] 闭合夹爪")
            close_gripper()
            time.sleep(3)
            
            # 4. 提起并移动到固定放置位置（使用关节角度直接移动）
            print(f"[4] 提起并移动到放置位置")
            self.arm.move_to_position(arm_x, arm_y, self.safe_z)
            time.sleep(1)
            
            # 使用关节角度直接移动到放置位置（来自5.move_with_gripper.py的pos_4）
            print(f"[4.1] 移动到固定放置位置")
            ret = self.arm.robot.joint_move(self.place_joint_pos, ABS, True, 0.25)
            if ret[0] != 0:
                print(f"[AGENT] 移动到放置位置失败: {ret[0]}")
            time.sleep(1)
            
            # 5. 打开夹爪释放
            print("[5] 打开夹爪释放")
            open_gripper()
            time.sleep(3)
            
            print(f"\n[AGENT] ✅ {fruit_name} 完成！")
            return True
            
        except Exception as e:
            print(f"[AGENT] 抓取失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def pick_fruit_to_safe_height(self, position, fruit_name="fruit"):
        """兼容旧接口，直接调用pick_and_place"""
        return self.pick_and_place(position, fruit_name)

    def place_fruit_simple(self):
        """兼容旧接口，已在pick_and_place中完成"""
        print("[AGENT] 放置已在抓取流程中完成")
        return True

    def place_fruit_to_basket(self, fruit_name="fruit"):
        """兼容旧接口，已在pick_and_place中完成"""
        print("[AGENT] 放置已在抓取流程中完成")
        return True

    def reset_arm(self):
        """复位机械臂"""
        print("\n[AGENT] 复位机械臂...")
        try:
            self.arm.return_to_home()
            print("[AGENT] 复位完成")
            return True
        except Exception as e:
            print(f"[AGENT] 复位失败: {e}")
            return False


if __name__ == "__main__":
    print("TaskAgent 模块")
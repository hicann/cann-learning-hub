# control/gripper_pwm.py
import wiringpi
import time

GRIPPER_PIN = 19
DUTY_OPEN = 21    # 打开夹爪（参考代码值）
DUTY_CLOSE = 27   # 闭合夹爪（参考代码值）

_wiringpi_initialized = False

def _setup_pwm():
    """每次操作前设置PWM（参考5.move_with_gripper.py的方式）"""
    global _wiringpi_initialized
    if not _wiringpi_initialized:
        wiringpi.wiringPiSetup()
        _wiringpi_initialized = True
    
    wiringpi.pinMode(GRIPPER_PIN, 1)
    # 初始值设为DUTY_OPEN(21)，启动时夹爪张开但不会太大
    wiringpi.softPwmCreate(GRIPPER_PIN, DUTY_OPEN, 100)

def init_gripper():
    """初始化夹爪"""
    try:
        _setup_pwm()
        print(f"[GRIPPER] 初始化成功")
    except Exception as e:
        print(f"[GRIPPER] 初始化失败: {e}")

def open_gripper():
    """打开夹爪"""
    print(f"[GRIPPER] 打开夹爪 (PWM={DUTY_OPEN})")
    try:
        _setup_pwm()
        wiringpi.softPwmWrite(GRIPPER_PIN, DUTY_OPEN)
        return True
    except Exception as e:
        print(f"[GRIPPER] 打开失败: {e}")
        return False

def close_gripper():
    """闭合夹爪"""
    print(f"[GRIPPER] 闭合夹爪 (PWM={DUTY_CLOSE})")
    try:
        _setup_pwm()
        wiringpi.softPwmWrite(GRIPPER_PIN, DUTY_CLOSE)
        return True
    except Exception as e:
        print(f"[GRIPPER] 闭合失败: {e}")
        return False

def test_gripper():
    """测试夹爪"""
    print("测试夹爪...")
    
    print("1. 打开夹爪")
    open_gripper()
    time.sleep(1)
    
    print("2. 闭合夹爪")
    close_gripper()
    time.sleep(1)
    
    print("3. 再次打开")
    open_gripper()
    
    print("测试完成")

if __name__ == "__main__":
    test_gripper()
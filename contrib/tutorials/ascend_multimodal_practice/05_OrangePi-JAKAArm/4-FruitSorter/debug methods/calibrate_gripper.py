#!/usr/bin/env python3
"""
夹爪校准脚本
用于测试和调整夹爪的PWM参数
"""

import wiringpi
import time
import sys

GRIPPER_PIN = 19

def calibrate_gripper():
    """夹爪校准函数"""
    print("夹爪校准程序")
    print("=" * 50)
    
    # 初始化wiringpi
    wiringpi.wiringPiSetup()
    wiringpi.pinMode(GRIPPER_PIN, 1)  # OUTPUT模式
    
    # 创建软PWM
    wiringpi.softPwmCreate(GRIPPER_PIN, 0, 100)
    
    try:
        while True:
            print("\n当前PWM范围: 0-100")
            print("常用值参考:")
            print("  10-15: 完全打开")
            print("  20-25: 完全关闭")
            print("  0: 最小")
            print("  100: 最大")
            
            try:
                value = int(input("\n输入PWM值(0-100)，输入-1退出: "))
                if value == -1:
                    break
                
                if 0 <= value <= 100:
                    print(f"设置PWM值为: {value}")
                    wiringpi.softPwmWrite(GRIPPER_PIN, value)
                    time.sleep(1)
                else:
                    print("错误: PWM值必须在0-100之间")
            except ValueError:
                print("错误: 请输入有效的数字")
                
    except KeyboardInterrupt:
        print("\n校准程序被中断")
    
    # 最后设置为打开状态
    wiringpi.softPwmWrite(GRIPPER_PIN, 10)
    print("\n校准完成，夹爪已设置为打开状态")

if __name__ == "__main__":
    print("注意: 此程序需要root权限运行")
    print("请使用: sudo python3 calibrate_gripper.py")
    calibrate_gripper()
# -*- coding: utf-8 -*-
"""
扩展实验2：软件 PWM 模拟 LED 呼吸灯

参考实现：通过快速切换 GPIO 高低电平模拟 PWM 信号，
实现 LED 渐亮渐暗的呼吸灯效果。

运行方式：python answer/extension2_pwm_breathing.py
输出文件：output/extension2_pwm_breathing_output.txt
"""
import os
import time
from enum import Enum
from dataclasses import dataclass

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---- 复用实验中的 GPIOController 基础定义 ----

class GPIOMode(Enum):
    SIMULATION = "simulation"
    HARDWARE = "hardware"

@dataclass
class GPIOEvent:
    pin: int
    value: int
    timestamp: float
    description: str

class GPIOController:
    def __init__(self, mode=GPIOMode.SIMULATION):
        self.mode = mode
        self._pins = {}
        self._events = []
        for pin in [17, 18, 22, 23, 24, 25]:
            self._pins[pin] = 0
        print("[GPIO] 仿真模式已启用")

    def write(self, pin, value):
        self._pins[pin] = value
        self._events.append(GPIOEvent(pin, value, time.time(), f"Write GPIO{pin}"))
        return True


# ---- 扩展实验2 核心实现 ----

def software_pwm(gpio: GPIOController, pin: int, duty_cycle: float,
                 duration: float, freq: float = 100):
    """
    软件PWM：通过快速切换GPIO高低电平模拟PWM信号

    参数：
        gpio: GPIOController 实例
        pin: 引脚编号
        duty_cycle: 占空比 (0.0 ~ 1.0)
        duration: 持续时间（秒）
        freq: PWM频率（Hz），默认100Hz
    """
    period = 1.0 / freq
    high_time = period * duty_cycle
    low_time = period - high_time
    cycles = int(duration * freq)

    for _ in range(cycles):
        if high_time > 0:
            gpio.write(pin, 1)
            time.sleep(high_time)
        if low_time > 0:
            gpio.write(pin, 0)
            time.sleep(low_time)


def breathing_led(gpio: GPIOController, pin: int,
                  cycles: int = 3, steps: int = 20):
    """
    LED呼吸灯效果：占空比从0渐变到1再回到0

    参数：
        gpio: GPIOController 实例
        pin: 引脚编号
        cycles: 呼吸周期数
        steps: 每个周期的步数（步数越多越平滑）
    """
    for cycle in range(cycles):
        print(f"\n  [呼吸周期 {cycle + 1}/{cycles}]")
        # 渐亮：占空比从 1/steps 到 1.0
        for i in range(1, steps + 1):
            duty = i / steps
            software_pwm(gpio, pin, duty, duration=0.05, freq=200)
            bar = "█" * i + "░" * (steps - i)
            print(f"    渐亮 [{bar}] 占空比={duty:.2f}")

        # 渐暗：占空比从 1.0 到 1/steps
        for i in range(steps, 0, -1):
            duty = i / steps
            software_pwm(gpio, pin, duty, duration=0.05, freq=200)
            bar = "█" * i + "░" * (steps - i)
            print(f"    渐暗 [{bar}] 占空比={duty:.2f}")


# ---- 测试 ----

def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extension2_pwm_breathing_output.txt")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()

    with redirect_stdout(buf):
        print("=" * 60)
        print("扩展实验2：软件 PWM 模拟 LED 呼吸灯")
        print("=" * 60)

        gpio = GPIOController(mode=GPIOMode.SIMULATION)

        print("\n[演示] LED呼吸灯效果 (3个周期, 每周期10步)")
        breathing_led(gpio, pin=18, cycles=3, steps=10)

        # 统计GPIO操作次数
        print(f"\n[统计] GPIO操作总次数: {len(gpio._events)}")
        print(f"[统计] 高电平次数: {sum(1 for e in gpio._events if e.value == 1)}")
        print(f"[统计] 低电平次数: {sum(1 for e in gpio._events if e.value == 0)}")
        print("=" * 60)
        print("[OK] 呼吸灯演示完成")

    output = buf.getvalue()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)
    try:
        print(output)
    except UnicodeEncodeError:
        print(output.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    print(f"\n输出已保存到: {output_file}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
扩展实验3：多 GPIO 引脚批量操作与状态显示

参考实现：扩展 GPIOController 支持批量设置多个引脚，
并用文本字符绘制 LED 状态条，模拟 8 个 LED 的跑马灯效果。

运行方式：python answer/extension3_batch_led.py
输出文件：output/extension3_batch_led_output.txt
"""
import os
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List

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
        for pin in [17, 18, 22, 23, 24, 25, 5, 6]:
            self._pins[pin] = 0
        print("[GPIO] 仿真模式已启用")

    def write(self, pin, value):
        self._pins[pin] = value
        self._events.append(GPIOEvent(pin, value, time.time(), f"Write GPIO{pin}"))
        return True

    def read(self, pin):
        return self._pins.get(pin, 0)


# ---- 扩展实验3 核心实现 ----

def batch_write(gpio: GPIOController, pin_values: Dict[int, int]):
    """
    批量设置多个GPIO引脚的电平

    参数：
        gpio: GPIOController 实例
        pin_values: 字典 {引脚号: 电平值}，如 {17: 1, 18: 0, 22: 1}
    """
    for pin, value in pin_values.items():
        gpio.write(pin, value)


def led_bar_display(gpio: GPIOController, pins: List[int]):
    """
    用文本字符绘制LED状态条

    参数：
        gpio: GPIOController 实例
        pins: 要显示的引脚列表

    返回：
        状态条字符串，如 [█ ▄ █ ▄ █ ▄ █ ▄]
    """
    segments = []
    for pin in pins:
        value = gpio.read(pin)
        segments.append("█" if value == 1 else "▄")
    bar = "[" + " ".join(segments) + "]"
    return bar


def running_led_demo(gpio: GPIOController, pins: List[int],
                     rounds: int = 2, step_delay: float = 0.2):
    """
    跑马灯效果：依次点亮每个LED，同时熄灭其他

    参数：
        gpio: GPIOController 实例
        pins: LED对应的引脚列表
        rounds: 循环轮数
        step_delay: 每步间隔秒数
    """
    num = len(pins)
    for r in range(rounds):
        print(f"\n  === 第 {r + 1} 轮跑马灯 ===")
        for i in range(num):
            # 批量设置：当前LED亮，其他灭
            pin_values = {p: (1 if j == i else 0) for j, p in enumerate(pins)}
            batch_write(gpio, pin_values)
            bar = led_bar_display(gpio, pins)
            print(f"    步骤{i + 1}: {bar}  (GPIO{pins[i]} 亮)")
            time.sleep(step_delay)
    # 最终全部熄灭
    batch_write(gpio, {p: 0 for p in pins})


# ---- 测试 ----

def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extension3_batch_led_output.txt")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()

    with redirect_stdout(buf):
        print("=" * 60)
        print("扩展实验3：多 GPIO 引脚批量操作与状态显示")
        print("=" * 60)

        gpio = GPIOController(mode=GPIOMode.SIMULATION)

        # 8个LED对应的引脚
        led_pins = [17, 18, 22, 23, 24, 25, 5, 6]

        print(f"\nLED引脚映射: {led_pins}")
        print(f"LED数量: {len(led_pins)}")

        # 测试批量写入
        print("\n[测试1] 批量写入")
        batch_write(gpio, {17: 1, 18: 1, 22: 0, 23: 1, 24: 0, 25: 0, 5: 1, 6: 0})
        bar = led_bar_display(gpio, led_pins)
        print(f"  状态条: {bar}")

        # 跑马灯演示
        print("\n[测试2] 8个LED跑马灯效果 (2轮)")
        running_led_demo(gpio, led_pins, rounds=2, step_delay=0.1)

        # 统计
        print(f"\n[统计] GPIO操作总次数: {len(gpio._events)}")
        print("=" * 60)
        print("[OK] 批量操作与跑马灯演示完成")

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

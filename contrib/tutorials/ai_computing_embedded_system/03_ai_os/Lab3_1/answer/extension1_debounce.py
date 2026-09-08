# -*- coding: utf-8 -*-
"""
扩展实验1：GPIO 按键防抖（软件去抖动）

参考实现：基于 GPIOController 实现软件防抖读取函数。
在真实硬件中，按键按下和释放瞬间会产生机械抖动（通常 5~20ms），
导致 GPIO 电平在短时间内反复跳变。软件防抖通过连续多次读取
相同电平才确认状态变化来消除抖动影响。

运行方式：python answer/extension1_debounce.py
输出文件：output/extension1_debounce_output.txt
"""
import os
import sys
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Callable, List

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
        self._callbacks = {}
        for pin in [17, 18, 22, 23, 24, 25]:
            self._pins[pin] = 0
        print("[GPIO] 仿真模式已启用")

    def write(self, pin, value):
        self._pins[pin] = value
        self._events.append(GPIOEvent(pin, value, time.time(), f"Write GPIO{pin}"))
        return True

    def read(self, pin):
        return self._pins.get(pin, 0)


# ---- 扩展实验1 核心实现 ----

def debounced_read(gpio: GPIOController, pin: int,
                   stable_count: int = 3, interval: float = 0.05) -> int:
    """
    软件防抖读取函数

    参数：
        gpio: GPIOController 实例
        pin: 引脚编号
        stable_count: 连续多少次相同电平才确认（默认3次）
        interval: 每次读取间隔秒数（默认0.05秒=50ms）

    返回：
        稳定后的电平值 (0 或 1)
    """
    last_value = gpio.read(pin)
    count = 1

    while count < stable_count:
        time.sleep(interval)
        current = gpio.read(pin)
        if current == last_value:
            count += 1
        else:
            last_value = current
            count = 1

    return last_value


# ---- 测试 ----

def main():
    # 重定向输出到文件
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extension1_debounce_output.txt")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()

    with redirect_stdout(buf):
        print("=" * 60)
        print("扩展实验1：GPIO 按键防抖（软件去抖动）")
        print("=" * 60)

        gpio = GPIOController(mode=GPIOMode.SIMULATION)

        # 测试1：写入高电平，防抖读取应返回1
        print("\n[测试1] 写入高电平后防抖读取")
        gpio.write(18, 1)
        value = debounced_read(gpio, 18, stable_count=3, interval=0.01)
        print(f"  写入: 1, 防抖读取: {value}, 结果: {'通过' if value == 1 else '失败'}")

        # 测试2：写入低电平，防抖读取应返回0
        print("\n[测试2] 写入低电平后防抖读取")
        gpio.write(18, 0)
        value = debounced_read(gpio, 18, stable_count=3, interval=0.01)
        print(f"  写入: 0, 防抖读取: {value}, 结果: {'通过' if value == 0 else '失败'}")

        # 测试3：多次切换
        print("\n[测试3] 多次切换电平")
        results = []
        for v in [1, 0, 1, 1, 0, 1, 0, 0]:
            gpio.write(18, v)
            read_v = debounced_read(gpio, 18, stable_count=2, interval=0.005)
            ok = "OK" if read_v == v else "FAIL"
            results.append((v, read_v, ok))
            print(f"  写入{v} -> 读取{read_v} [{ok}]")

        all_pass = all(r[2] == "OK" for r in results)
        print(f"\n总结: {'全部通过' if all_pass else '存在失败'}")
        print("=" * 60)

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

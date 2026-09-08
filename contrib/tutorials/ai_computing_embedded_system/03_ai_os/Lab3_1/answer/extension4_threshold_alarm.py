# -*- coding: utf-8 -*-
"""
扩展实验4：推理结果阈值判断控制 GPIO 输出

参考实现：执行 AI 推理，根据推理结果均值与阈值的比较
结果控制 GPIO 输出（报警指示灯），模拟端侧 AI 部署中
"推理结果驱动外设"的典型场景。

运行方式：python answer/extension4_threshold_alarm.py
输出文件：output/extension4_threshold_alarm_output.txt
"""
import os
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---- 复用实验中的基础定义 ----

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


class CANNRuntime:
    """简化版CANN Runtime（仿真模式）"""
    def __init__(self):
        self.mode = "simulation"
        self.stats = {'inference_calls': 0, 'total_time': 0}
        print("[CANN] 仿真模式已启用")

    def execute_inference(self, input_data: np.ndarray) -> np.ndarray:
        self.stats['inference_calls'] += 1
        start = time.time()
        # 模拟推理：矩阵乘 + ReLU
        data = input_data.reshape(1, -1) if input_data.ndim == 1 else input_data
        weight = np.random.randn(data.shape[-1], data.shape[-1]) * 0.1
        output = np.maximum(data @ weight, 0)
        time.sleep(0.05)
        elapsed = time.time() - start
        self.stats['total_time'] += elapsed
        return output.astype(np.float32)


# ---- 扩展实验4 核心实现 ----

def threshold_alarm_demo(gpio: GPIOController, runtime: CANNRuntime,
                         alarm_pin: int = 23, threshold: float = 0.0,
                         rounds: int = 5):
    """
    推理结果阈值判断控制GPIO输出

    参数：
        gpio: GPIOController 实例
        runtime: CANNRuntime 实例
        alarm_pin: 报警指示灯引脚
        threshold: 判断阈值
        rounds: 推理轮数
    """
    alarm_count = 0

    for i in range(rounds):
        # 1. 生成输入数据并执行推理
        input_data = np.random.randn(1, 256).astype(np.float32)
        result = runtime.execute_inference(input_data)

        # 2. 计算推理结果均值作为判断指标
        mean_value = float(result.mean())

        # 3. 阈值判断控制GPIO
        if mean_value > threshold:
            gpio.write(alarm_pin, 1)
            alarm_count += 1
            led_status = "亮 [报警]"
        else:
            gpio.write(alarm_pin, 0)
            led_status = "灭 [正常]"

        # 4. 打印本轮结果
        bar = "█" * int(mean_value * 50 + 10) if mean_value > 0 else "░" * 10
        print(f"  第{i+1}轮: 均值={mean_value:+.4f} | {bar} | GPIO{alarm_pin} {led_status}")

    return alarm_count


# ---- 测试 ----

def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extension4_threshold_alarm_output.txt")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()

    with redirect_stdout(buf):
        print("=" * 60)
        print("扩展实验4：推理结果阈值判断控制 GPIO 输出")
        print("=" * 60)

        gpio = GPIOController(mode=GPIOMode.SIMULATION)
        runtime = CANNRuntime()

        threshold = 0.0
        alarm_pin = 23
        print(f"\n配置: 报警引脚=GPIO{alarm_pin}, 阈值={threshold}")
        print(f"逻辑: 推理结果均值 > {threshold} → 点亮报警灯")
        print("\n开始执行5轮推理...\n")

        alarm_count = threshold_alarm_demo(
            gpio, runtime, alarm_pin=alarm_pin,
            threshold=threshold, rounds=5
        )

        # 统计
        print(f"\n[统计] 推理总次数: {runtime.stats['inference_calls']}")
        print(f"[统计] 推理总耗时: {runtime.stats['total_time']:.4f}秒")
        print(f"[统计] 报警触发次数: {alarm_count}/5")
        print(f"[统计] GPIO操作次数: {len(gpio._events)}")
        print("=" * 60)
        print("[OK] 阈值报警演示完成")

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

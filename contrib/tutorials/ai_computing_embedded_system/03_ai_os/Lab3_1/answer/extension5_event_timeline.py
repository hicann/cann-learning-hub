# -*- coding: utf-8 -*-
"""
扩展实验5：GPIO 事件时间线记录与可视化

参考实现：记录 GPIO 所有事件的时间戳，用文本字符绘制
事件时间线图，展示各引脚的电平变化历程。这是嵌入式
系统调试中常用的可视化手段。

运行方式：python answer/extension5_event_timeline.py
输出文件：output/extension5_event_timeline_output.txt
"""
import os
import time
import random
from enum import Enum
from dataclasses import dataclass
from typing import List

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

    def get_events(self) -> List[GPIOEvent]:
        return self._events


# ---- 扩展实验5 核心实现 ----

def draw_timeline(events: List[GPIOEvent], width: int = 60):
    """
    用文本字符绘制GPIO事件时间线

    参数：
        events: GPIO事件列表
        width: 时间线宽度（字符数）

    返回：
        时间线字符串
    """
    if not events:
        return "(无事件)"

    pins = sorted(set(e.pin for e in events))
    t_min = events[0].timestamp
    t_max = events[-1].timestamp
    t_range = t_max - t_min if t_max > t_min else 1.0

    lines = []
    header = "时间轴: " + "0".ljust(width // 2 - 4) + "|" + "结束".rjust(width // 2 - 2)
    lines.append(header)
    lines.append("       " + "-" * width)

    for pin in pins:
        pin_events = [e for e in events if e.pin == pin]
        if not pin_events:
            continue

        # 构建时间线
        timeline = ["─"] * width
        current_value = 0

        for e in pin_events:
            pos = int((e.timestamp - t_min) / t_range * (width - 1))
            pos = max(0, min(width - 1, pos))
            if e.value == 1:
                timeline[pos] = "┃"
                # 从此处开始用高电平符号填充到下一个跳变
                current_value = 1
            else:
                timeline[pos] = "┃"
                current_value = 0
            # 填充电平持续段
            for k in range(pos + 1, width):
                timeline[k] = "━" if current_value == 1 else "─"

        # 重新精确构建：逐段填充
        timeline = ["─"] * width
        pin_events_sorted = sorted(pin_events, key=lambda x: x.timestamp)
        state = 0
        last_pos = 0
        for e in pin_events_sorted:
            pos = int((e.timestamp - t_min) / t_range * (width - 1))
            pos = max(0, min(width - 1, pos))
            # 填充上一段
            char = "━" if state == 1 else "─"
            for k in range(last_pos, pos):
                timeline[k] = char
            timeline[pos] = "┃"
            state = e.value
            last_pos = pos + 1
        # 填充最后一段
        char = "━" if state == 1 else "─"
        for k in range(last_pos, width):
            timeline[k] = char

        lines.append(f"GPIO{pin:2d} |" + "".join(timeline) + "|")

    lines.append("       " + "-" * width)
    lines.append("图例: ━ 高电平持续  ─ 低电平持续  ┃ 电平跳变时刻")
    return "\n".join(lines)


def event_statistics(events: List[GPIOEvent]):
    """生成事件统计表"""
    pins = sorted(set(e.pin for e in events))
    lines = []
    lines.append(f"{'引脚':<10} {'操作次数':<10} {'高电平次数':<12} {'低电平次数':<12}")
    lines.append("-" * 50)
    for pin in pins:
        pin_events = [e for e in events if e.pin == pin]
        total = len(pin_events)
        high = sum(1 for e in pin_events if e.value == 1)
        low = sum(1 for e in pin_events if e.value == 0)
        lines.append(f"GPIO{pin:<5} {total:<12} {high:<14} {low:<14}")
    return "\n".join(lines)


# ---- 测试 ----

def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extension5_event_timeline_output.txt")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()

    with redirect_stdout(buf):
        print("=" * 60)
        print("扩展实验5：GPIO 事件时间线记录与可视化")
        print("=" * 60)

        gpio = GPIOController(mode=GPIOMode.SIMULATION)
        pins = [17, 18, 22, 23]

        # 执行一系列操作，加入随机间隔
        print("\n[1] 执行GPIO操作序列...")
        for _ in range(8):
            pin = random.choice(pins)
            value = random.choice([0, 1])
            gpio.write(pin, value)
            time.sleep(random.uniform(0.01, 0.1))

        # 获取所有事件
        events = gpio.get_events()
        print(f"   共记录 {len(events)} 个事件")

        # 绘制时间线
        print("\n[2] GPIO事件时间线可视化:")
        print()
        timeline = draw_timeline(events, width=60)
        print(timeline)

        # 统计表
        print("\n[3] 事件统计表:")
        print()
        stats = event_statistics(events)
        print(stats)

        print("\n" + "=" * 60)
        print("[OK] 事件时间线可视化完成")

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

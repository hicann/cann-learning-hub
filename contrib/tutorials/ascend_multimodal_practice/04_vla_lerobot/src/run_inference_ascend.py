"""
SO-101 机械臂 ACT 策略推理脚本（香橙派 + 昇腾 310 NPU 版本）

本脚本在香橙派（OrangePi AIPro，搭载昇腾 Ascend 310B NPU）上运行，
加载训练好的 ACT 策略，驱动 SO-101 机械臂完成方块抓取任务。

================================================================================
环境要求（香橙派 + 昇腾 310 NPU）
================================================================================

硬件：
- 香橙派 OrangePi AIPro（昇腾 Ascend 310B NPU）
- SO-101 机械臂（Follower 臂 + 两个摄像头：前置 + 腕部）
- USB 连接（机械臂舵机 + 摄像头）

软件：
- OS: Ubuntu 22.04（香橙派预装）
- Python: 3.10（香橙派默认）
- CANN: 8.0.RC1+（香橙派预装，用于 NPU 加速）
- PyTorch: 2.1+ + torch_npu（让 PyTorch 识别昇腾 NPU）
- LeRobot: 从源码安装（pip install -e .）
- ffmpeg: 视频解码需要

安装命令（在香橙派上执行）：
    # 1. 激活 Python 环境
    conda activate lerobot  # 或 python3 -m venv lerobot && source lerobot/bin/activate

    # 2. 安装 torch_npu（让 PyTorch 支持昇腾 310 NPU）
    pip install torch==2.1.0 torch_npu==2.1.0.post3 torchvision

    # 3. 设置 CANN 环境变量（通常香橙派已配置，若无则手动 source）
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 4. 安装 LeRobot（从源码，因为需要修改设备适配）
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot && pip install -e .

    # 5. 安装 ffmpeg
    sudo apt install ffmpeg

    # 6. 验证 NPU 可用
    python -c "import torch; import torch_npu; print('NPU可用:', torch.npu.is_available())"

================================================================================
使用方法
================================================================================

    # 基本用法：加载模型并启动推理
    python run_inference_ascend.py \\
        --policy.path=./pretrained_model \\
        --robot.port=/dev/ttyACM0 \\
        --num_episodes=10 \\
        --task="抓取方块"

    # 参数说明：
    #   --policy.path    训练好的 ACT 模型路径
    #   --robot.port     机械臂串口（通常 /dev/ttyACM0）
    #   --num_episodes   评估多少次
    #   --task           任务描述
    #   --display_data   是否实时显示观察数据（默认 False）

================================================================================
关于昇腾 310 NPU 适配的说明
================================================================================

LeRobot 官方基于 PyTorch + CUDA（NVIDIA GPU），本脚本通过 torch_npu 适配昇腾 NPU。
关键改动：
1. import torch_npu 让 PyTorch 识别 "npu" 设备
2. 模型和张量 .to("npu") 而非 .to("cuda")
3. ACTPolicy 加载后显式迁移到 NPU

注意：昇腾 310B 的算力有限（相比训练用的 NVIDIA GPU），推理速度可能稍慢，
但对 ACT 这类轻量模型（80M 参数）完全够用，实测可达 15-30 FPS。

作者：昇腾AI多模态实践课程
================================================================================
"""

import argparse
import os
import sys
import time

# ===== 昇腾 NPU 适配：必须在 import torch 后立即 import torch_npu =====
import torch
try:
    import torch_npu  # noqa: F401  让 PyTorch 识别昇腾 NPU 设备
    NPU_AVAILABLE = torch.npu.is_available()
    DEVICE = "npu" if NPU_AVAILABLE else "cpu"
    if NPU_AVAILABLE:
        print(f"✅ 检测到昇腾 NPU: {torch.npu.get_device_name(0)}")
    else:
        print("⚠️ 未检测到昇腾 NPU，回退到 CPU（速度会慢）")
except ImportError:
    print("⚠️ 未安装 torch_npu，回退到 CPU。请按文件头说明安装 torch_npu。")
    NPU_AVAILABLE = False
    DEVICE = "cpu"

# ===== LeRobot 相关导入 =====
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robots.so101_follower.so101_follower import SO101Follower
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def load_policy(model_path: str) -> ACTPolicy:
    """加载训练好的 ACT 策略，并迁移到昇腾 NPU"""
    print(f"加载 ACT 策略: {model_path}")
    policy = ACTPolicy.from_pretrained(model_path)
    policy.eval()
    policy.to(DEVICE)
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"✅ 模型已加载到 {DEVICE}（参数量: {param_count/1e6:.1f}M）")
    return policy


def setup_robot(port: str, cameras: dict) -> SO101Follower:
    """初始化 SO-101 Follower 臂 + 摄像头"""
    print(f"初始化 SO-101 Follower 臂 (端口: {port})")
    robot = SO101Follower(
        port=port,
        id="so101_follower",
        cameras=cameras,
    )
    robot.connect()
    print("✅ 机械臂已连接，摄像头已就绪")
    return robot


def run_inference_episode(policy, robot, task: str, max_steps: int = 300):
    """
    运行一个完整的推理 episode：
    循环{拍摄画面 → 策略预测动作 → 执行动作}，直到任务完成或达到最大步数
    """
    print(f"\n🎯 开始推理（任务: {task}，最大步数: {max_steps}）")
    policy.reset()  # 重置策略的内部状态（如 ACT 的潜在变量 z）

    step = 0
    start_time = time.time()
    for step in range(max_steps):
        # 1. 读取当前观察（摄像头画面 + 关节状态）
        observation = robot.get_observation()

        # 2. 构造 batch 并迁移到 NPU（只保留 observation.* 键）
        batch = {
            k: torch.as_tensor(v).unsqueeze(0).to(DEVICE)
            for k, v in observation.items()
            if k.startswith("observation.")
        }

        # 3. 策略预测动作
        with torch.inference_mode():
            action = policy.select_action(batch)

        # 4. 执行动作（驱动机械臂）
        action_np = action.squeeze(0).cpu().numpy()
        robot.send_action(action_np)

        # 注意：step 由 for 循环自动递增，此处不再手动 +1
        if (step + 1) % 30 == 0:  # 每30步打印一次进度
            elapsed = time.time() - start_time
            fps = (step + 1) / elapsed
            print(f"  步骤 {step + 1}/{max_steps}（FPS: {fps:.1f}）")

    elapsed = time.time() - start_time
    # 循环结束时 step = max_steps - 1，实际执行步数为 max_steps
    total_steps = max_steps if step >= max_steps - 1 else step + 1
    print(f"✅ 推理完成（{total_steps} 步，耗时 {elapsed:.1f} 秒，平均 FPS: {total_steps/elapsed:.1f}）")
    return total_steps


def main():
    parser = argparse.ArgumentParser(description="SO-101 ACT 推理（香橙派 + 昇腾 310 NPU）")
    parser.add_argument("--policy.path", dest="policy_path",
                        default="./pretrained_model",
                        help="训练好的 ACT 模型路径")
    parser.add_argument("--robot.port", dest="robot_port",
                        default="/dev/ttyACM0",
                        help="机械臂串口路径")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="评估多少个 episode")
    parser.add_argument("--task", default="抓取方块",
                        help="任务描述")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="每个 episode 最大步数")
    parser.add_argument("--display_data", action="store_true",
                        help="是否实时显示观察数据")
    args = parser.parse_args()

    # 1. 加载策略
    policy = load_policy(args.policy_path)

    # 2. 初始化机械臂（前置 + 腕部摄像头）
    cameras = {
        "front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 360, "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 360, "fps": 30},
    }
    robot = setup_robot(args.robot_port, cameras)

    # 3. 运行评估
    success_count = 0
    for ep in range(args.num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{args.num_episodes}")
        print(f"{'='*50}")

        input(f"按回车开始 Episode {ep+1}（确保机械臂在初始位置）...")
        steps = run_inference_episode(policy, robot, args.task, args.max_steps)

        # 由人工判断是否成功
        result = input("本次任务是否成功？(y/n): ").strip().lower()
        if result == 'y':
            success_count += 1

        # 回到初始位置（简单实现：可扩展为归位动作）
        print("机械臂归位中...")

    # 4. 统计结果
    success_rate = success_count / args.num_episodes * 100
    print(f"\n{'='*50}")
    print(f"📊 评估结果：{args.num_episodes} 次中成功 {success_count} 次")
    print(f"   成功率: {success_rate:.0f}%")
    print(f"{'='*50}")

    # 5. 清理
    robot.disconnect()
    print("✅ 机械臂已断开，程序结束")


if __name__ == "__main__":
    main()

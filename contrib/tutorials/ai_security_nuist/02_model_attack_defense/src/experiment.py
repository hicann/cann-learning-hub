"""实验配置与计划生成脚本。

这里先把常用参数组合列出来，
后面批量训练时直接按生成的命令执行。
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.poison import SUPPORTED_TRIGGER_TYPES
from src.utils import ensure_dir, save_json
import src.model as model_module


@dataclass
class ExperimentSetting:
    """单次实验配置。"""

    name: str
    seed: int
    poison_ratio: float
    trigger_size: int
    alpha: float
    trigger_type: str
    note: str


def parse_float_list(text: str) -> list[float]:
    """把逗号分隔字符串转成浮点数列表。"""

    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    """把逗号分隔字符串转成整数列表。"""

    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    """把逗号分隔字符串转成字符串列表。"""

    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """解析实验计划参数。"""

    parser = argparse.ArgumentParser(description="生成 BadNets 隐蔽性分析实验计划。")
    parser.add_argument("--poison-ratios", default="0.01,0.05,0.10")
    parser.add_argument("--trigger-sizes", default="2,4,6")
    parser.add_argument("--alphas", default="1.0,0.7,0.4")
    parser.add_argument("--trigger-types", default="square,checkerboard")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--device-target", default="Ascend")
    parser.add_argument("--ms-mode", default="GRAPH", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--position", default="bottom_right", choices=["bottom_right", "top_left"])
    parser.add_argument("--norm-type", default="group", choices=getattr(model_module, "SUPPORTED_NORM_TYPES", ("group", "batch", "none")))
    parser.add_argument("--poison-cache-dir", type=Path, default=PROJECT_ROOT / "src" / "outputs" / "poison_cache")
    parser.add_argument("--experiments-dir", type=Path, default=PROJECT_ROOT / "src" / "outputs" / "experiments")
    parser.add_argument("--num-parallel-workers", type=int, default=8)
    parser.add_argument("--prefetch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--output-path", type=Path, default=PROJECT_ROOT / "src" / "outputs" / "metrics" / "experiment_plan.json")
    parser.add_argument("--commands-path", type=Path, default=PROJECT_ROOT / "src" / "outputs" / "metrics" / "experiment_train_commands.sh")
    return parser.parse_args()


def format_float_for_name(value: float) -> str:
    """把浮点数转成适合目录名的短文本。"""

    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def make_run_name(trigger_type: str, poison_ratio: float, trigger_size: int, alpha: float, seed: int) -> str:
    """生成稳定、可读、能直接作为 save-dir 的实验名。"""

    poison_text = format_float_for_name(poison_ratio)
    alpha_text = format_float_for_name(alpha)
    return f"{trigger_type}_pr{poison_text}_ts{trigger_size}_a{alpha_text}_seed{seed}"


def build_experiment_grid(
    poison_ratios: list[float],
    trigger_sizes: list[int],
    alphas: list[float],
    trigger_types: list[str],
    seeds: list[int],
) -> list[ExperimentSetting]:
    """构造实验参数网格。"""

    settings: list[ExperimentSetting] = []
    for trigger_type in trigger_types:
        for seed in seeds:
            for poison_ratio in poison_ratios:
                for trigger_size in trigger_sizes:
                    for alpha in alphas:
                        settings.append(
                            ExperimentSetting(
                                name=make_run_name(trigger_type, poison_ratio, trigger_size, alpha, seed),
                                seed=seed,
                                poison_ratio=poison_ratio,
                                trigger_size=trigger_size,
                                alpha=alpha,
                                trigger_type=trigger_type,
                                note="比较随机种子、投毒比例、触发器大小、透明度和触发器形态对 Clean Accuracy / ASR 的影响。",
                            )
                        )
    return settings


def validate_grid_inputs(
    poison_ratios: list[float],
    trigger_sizes: list[int],
    alphas: list[float],
    trigger_types: list[str],
    seeds: list[int],
) -> None:
    """检查实验网格参数，尽早发现拼写或范围错误。"""

    if not poison_ratios or not trigger_sizes or not alphas or not trigger_types or not seeds:
        raise ValueError("poison-ratios、trigger-sizes、alphas、trigger-types、seeds 都不能为空")
    invalid_trigger_types = [item for item in trigger_types if item not in SUPPORTED_TRIGGER_TYPES]
    if invalid_trigger_types:
        raise ValueError(f"不支持的 trigger-type: {invalid_trigger_types}，可选值: {SUPPORTED_TRIGGER_TYPES}")
    for poison_ratio in poison_ratios:
        if not 0.0 <= poison_ratio <= 1.0:
            raise ValueError(f"poison_ratio 必须在 [0, 1] 内，当前为 {poison_ratio}")
    for trigger_size in trigger_sizes:
        if trigger_size <= 0:
            raise ValueError(f"trigger_size 必须为正整数，当前为 {trigger_size}")
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0, 1] 内，当前为 {alpha}")


def shell_join(parts: list[object]) -> str:
    """生成适合 Linux/Ascend 服务器复制执行的一行命令。"""

    return " ".join(shlex.quote(str(part)) for part in parts)


def build_train_command(setting: ExperimentSetting, args: argparse.Namespace) -> str:
    """根据实验配置生成完整训练命令。"""

    save_dir = args.experiments_dir / setting.name
    return shell_join(
        [
            args.python_bin,
            "src/train.py",
            "--task",
            "badnet",
            "--device-target",
            args.device_target,
            "--ms-mode",
            args.ms_mode,
            "--epochs",
            args.epochs,
            "--batch-size",
            args.batch_size,
            "--lr",
            args.lr,
            "--image-size",
            args.image_size,
            "--server-fast",
            "true",
            "--num-parallel-workers",
            args.num_parallel_workers,
            "--prefetch-size",
            args.prefetch_size,
            "--eval-train-acc",
            "false",
            "--eval-interval",
            args.eval_interval,
            "--cache-poisoned-train",
            "true",
            "--poison-cache-dir",
            args.poison_cache_dir.as_posix(),
            "--target-label",
            args.target_label,
            "--norm-type",
            args.norm_type,
            "--poison-rate",
            setting.poison_ratio,
            "--trigger-size",
            setting.trigger_size,
            "--alpha",
            setting.alpha,
            "--trigger-type",
            setting.trigger_type,
            "--position",
            args.position,
            "--seed",
            setting.seed,
            "--save-dir",
            save_dir.as_posix(),
        ]
    )


def save_command_file(commands: list[str], commands_path: Path) -> Path:
    """保存可直接在 Linux 服务器上执行的训练命令列表。"""

    ensure_dir(commands_path.parent)
    lines = [
        "#!/usr/bin/env bash",
        "set -e",
        "",
        "# created by src/experiment.py",
        "# run from the chapter root after src/data/train and src/data/test are ready",
        "",
        *commands,
        "",
    ]
    commands_path.write_text("\n".join(lines), encoding="utf-8")
    return commands_path


def main() -> dict[str, object]:
    """生成实验计划并保存到 outputs/metrics。"""

    args = parse_args()
    poison_ratios = parse_float_list(args.poison_ratios)
    trigger_sizes = parse_int_list(args.trigger_sizes)
    alphas = parse_float_list(args.alphas)
    trigger_types = parse_str_list(args.trigger_types)
    seeds = parse_int_list(args.seeds)
    validate_grid_inputs(poison_ratios, trigger_sizes, alphas, trigger_types, seeds)

    settings = build_experiment_grid(poison_ratios, trigger_sizes, alphas, trigger_types, seeds)
    ensure_dir(args.output_path.parent)
    commands = [build_train_command(setting, args) for setting in settings]
    save_command_file(commands, args.commands_path)

    payload = {
        "total_runs": len(settings),
        "base_train_args": {
            "device_target": args.device_target,
            "ms_mode": args.ms_mode,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "image_size": args.image_size,
            "target_label": args.target_label,
            "norm_type": args.norm_type,
            "position": args.position,
            "poison_cache_dir": args.poison_cache_dir.as_posix(),
            "experiments_dir": args.experiments_dir.as_posix(),
            "eval_train_acc": False,
            "eval_interval": args.eval_interval,
            "server_fast": True,
            "num_parallel_workers": args.num_parallel_workers,
            "prefetch_size": args.prefetch_size,
        },
        "commands_path": str(args.commands_path),
        "runs": [
            {
                **asdict(setting),
                "save_dir": (args.experiments_dir / setting.name).as_posix(),
                "train_command": command,
            }
            for setting, command in zip(settings, commands)
        ],
    }
    save_json(payload, args.output_path)

    print(f"[OK] 实验计划已生成: {args.output_path}")
    print(f"[OK] 训练命令已生成: {args.commands_path}")
    print(f"[信息] 总实验数量: {len(settings)}")
    for index, setting in enumerate(settings, start=1):
        print(
            f"  Run {index:02d} | name={setting.name} | seed={setting.seed} "
            f"| poison_ratio={setting.poison_ratio} "
            f"| trigger_size={setting.trigger_size} | alpha={setting.alpha} "
            f"| trigger_type={setting.trigger_type}"
        )

    print("[说明] 训练命令可以直接在项目根目录执行。训练结束后再跑 evaluate.py。")
    return payload


if __name__ == "__main__":
    main()

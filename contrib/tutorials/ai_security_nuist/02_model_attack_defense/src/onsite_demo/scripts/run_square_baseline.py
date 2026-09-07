from __future__ import annotations

import argparse
import sys
from pathlib import Path


ONSITE_ROOT = Path(__file__).resolve().parents[1]
if str(ONSITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ONSITE_ROOT))

from demo_lib.paths import ensure_dir, load_demo_config, make_run_timestamp, resolve_project_root
from demo_lib.subset import create_demo_subset
from demo_lib.training import run_stage_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Square Trigger Baseline / 白块触发器基线实验")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device-target", default="auto")
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    paths = resolve_project_root()
    config = load_demo_config()
    seed = int(args.seed if args.seed is not None else config["seed"])
    train_per_class = int(args.train_per_class or config["train_per_class"])
    test_per_class = int(args.test_per_class or config["test_per_class"])
    if not (paths["DEMO_SUBSET_ROOT"] / "subset_manifest.json").exists():
        create_demo_subset(paths["TRAIN_DIR"], paths["TEST_DIR"], paths["DEMO_SUBSET_ROOT"], train_per_class, test_per_class, seed)

    run_root = ensure_dir(args.run_dir or paths["DEMO_RUNS_ROOT"] / make_run_timestamp())
    result = run_stage_pipeline(
        stage_config=config["square"],
        run_root=run_root,
        subset_root=paths["DEMO_SUBSET_ROOT"],
        target_label=int(config["target_label"]),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device_target=args.device_target,
        ms_mode=str(config["ms_mode"]),
        seed=seed,
    )
    print(f"square clean accuracy：{result['eval_summary']['clean_accuracy']:.4f}")
    print(f"square ASR：{result['eval_summary']['attack_success_rate']:.4f}")
    print(f"输出目录：{result['output_dir']}")
    return result


if __name__ == "__main__":
    main()

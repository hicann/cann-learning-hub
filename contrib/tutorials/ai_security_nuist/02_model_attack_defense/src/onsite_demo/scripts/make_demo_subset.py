from __future__ import annotations

import argparse
import sys
from pathlib import Path


ONSITE_ROOT = Path(__file__).resolve().parents[1]
if str(ONSITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ONSITE_ROOT))

from demo_lib.paths import resolve_project_root
from demo_lib.subset import create_demo_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="创建课堂现场演示统一 demo subset")
    parser.add_argument("--train-per-class", type=int, default=5)
    parser.add_argument("--test-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    paths = resolve_project_root()
    manifest = create_demo_subset(
        train_dir=paths["TRAIN_DIR"],
        test_dir=paths["TEST_DIR"],
        output_dir=paths["DEMO_SUBSET_ROOT"],
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )
    manifest_path = paths["DEMO_SUBSET_ROOT"] / "subset_manifest.json"
    print(f"现场 demo train 样本数：{manifest['train_total']}")
    print(f"现场 demo test 样本数：{manifest['test_total']}")
    print(f"subset_manifest.json：{manifest_path}")
    return manifest


if __name__ == "__main__":
    main()

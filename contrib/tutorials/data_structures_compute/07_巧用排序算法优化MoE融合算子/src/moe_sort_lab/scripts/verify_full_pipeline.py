#!/usr/bin/env python3
"""Compare saved full-pipeline outputs with deterministic CPU references."""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def read(path: Path, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype=dtype).reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    data = args.data_dir
    meta = __import__("json").loads((data / "input" / "meta.json").read_text(encoding="utf-8"))
    t, h, k = meta["num_tokens"], meta["hidden_size"], meta["top_k"]
    ref_indices = read(data / "input" / "ref_topk_indices.bin", np.int32, (t, k))
    ref_unpermute = read(data / "input" / "ref_unpermute_out.bin", np.float16, (t, h))
    for name in ("topk", "quicksort", "heapsort"):
        root = data / "output" / "full_pipeline" / name
        indices = read(root / "indices.bin", np.int32, (t, k))
        output = read(root / "unpermute_out.bin", np.float16, (t, h))
        index_ok = np.array_equal(indices, ref_indices)
        value_ok = np.allclose(output.astype(np.float32), ref_unpermute.astype(np.float32), atol=2e-2, rtol=2e-2)
        print(f"{name}: indices={'PASS' if index_ok else 'FAIL'} unpermute={'PASS' if value_ok else 'FAIL'}")
        if not (index_ok and value_ok):
            raise SystemExit(1)


if __name__ == "__main__":
    main()

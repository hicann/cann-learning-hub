#!/usr/bin/env python3
import argparse, json, os, struct, numpy as np

def write_bin(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_tokens", type=int, default=1024)
    p.add_argument("--top_k", type=int, default=4)
    args = p.parse_args()

    N = args.num_tokens
    K = args.top_k
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(N).astype(np.float16)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input")
    write_bin(os.path.join(out_dir, "x.bin"), x)

    # CPU reference
    x_f32 = x.astype(np.float32)
    ref_sum = np.array([np.sum(x_f32)], dtype=np.float16)
    ref_max = np.array([np.max(x_f32)], dtype=np.float16)
    topk_idx = np.argsort(-x_f32)[:K].astype(np.int32)
    topk_val = x[topk_idx].astype(np.float16)

    write_bin(os.path.join(out_dir, "ref_sum.bin"), ref_sum)
    write_bin(os.path.join(out_dir, "ref_max.bin"), ref_max)
    write_bin(os.path.join(out_dir, "ref_topk_val.bin"), topk_val)
    write_bin(os.path.join(out_dir, "ref_topk_idx.bin"), topk_idx)

    meta = {"num_tokens": N, "top_k": K}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated data: N={N}, K={K}")
    print(f"  sum={float(ref_sum[0]):.4f}, max={float(ref_max[0]):.4f}")
    print(f"  topK values={topk_val.tolist()}, indices={topk_idx.tolist()}")

if __name__ == "__main__":
    main()

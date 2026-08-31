#!/usr/bin/env python3
"""生成注意力算子测试数据：q/k/v（float16）与 torch 参考输出 ref（float16）。

用法：python3 gen_data.py [seq_len ...] [--dim 64] [--seed 42]
默认生成 512/1024/2048/4096 四个序列长度的数据到 <lab>/data/。
"""
import argparse
import os
import struct

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(LAB_DIR, "data")


def write_bin(path, tensor):
    tensor = tensor.contiguous().view(-1)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{tensor.numel()}e", *tensor.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_lens", nargs="*", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(DATA_DIR, exist_ok=True)

    for seq_len in args.seq_lens:
        out_dir = os.path.join(DATA_DIR, str(seq_len))
        os.makedirs(out_dir, exist_ok=True)

        q = torch.randn(seq_len, args.dim, dtype=torch.float32) * 0.5
        k = torch.randn(seq_len, args.dim, dtype=torch.float32) * 0.5
        v = torch.randn(seq_len, args.dim, dtype=torch.float32) * 0.5

        # torch 参考：QK^T -> scale -> softmax -> AV（fp32 计算，输出转 fp16）
        scores = q @ k.transpose(-2, -1) * (1.0 / (args.dim ** 0.5))
        p = torch.softmax(scores, dim=-1)
        ref = p @ v

        write_bin(os.path.join(out_dir, "q.bin"), q.half())
        write_bin(os.path.join(out_dir, "k.bin"), k.half())
        write_bin(os.path.join(out_dir, "kt.bin"), k.transpose(-2, -1).contiguous().half())
        write_bin(os.path.join(out_dir, "v.bin"), v.half())
        write_bin(os.path.join(out_dir, "ref.bin"), ref.half())

        # 顺带保存中间结果参考，便于精度分析
        np.save(os.path.join(out_dir, "ref_fp32.npy"), ref.numpy())
        np.save(os.path.join(out_dir, "scores_fp32.npy"), scores.numpy())

        print(f"seq_len={seq_len}: q/k/v/ref half 已生成 -> {out_dir}")

    print("全部数据生成完成")


if __name__ == "__main__":
    main()

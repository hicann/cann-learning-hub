"""生成固定 seed 的 MoE Router 测试数据，落盘供 C++ 测试程序与 Python 用例加载。

每个 case 一个目录 data/<case_name>/，内容:
    x_f32.npy            float32 张量（承载 FP16 量化后的值），[N, D]
    wgate_f32.npy        float32 张量（承载 FP16 量化后的值），[D, E]
    x_fp16.bin           FP16 原始字节（C++ 侧直接加载），[N, D]
    wgate_fp16.bin       FP16 原始字节，[D, E]
    ref_topk_idx.npy     int32 参考索引，[N, K]
    ref_topk_weights.npy float32 参考权重，[N, K]
    meta.json            N/D/E/K/seed 等元信息

说明:
    - 接口数据类型为 FP16（设计简化，与 08 章 attention_custom 一致；
      910B + CANN 9.0.0 向量 Cast 不支持 BF16<->FP32，详见 moe_ref.make_inputs 注释）。
    - 朴素向量实现没有 cube/TopK 高阶库的对齐约束，w_gate 不再补零列，
      第二维即真实专家数 E（meta.json 中 E_pad 恒等于 E，保留字段仅为兼容）。
    - 覆盖边界形态: N 非 32 倍数、最小规模 N=16、E=4 边界、大 N 等。

用法:
    python tools/gen_test_data.py [--out data]
"""

import argparse
import json
import os

import numpy as np
import torch

from moe_ref import make_inputs, moe_router

# (case_name, N, D, E, K)
CASES = [
    ("case_128_512_16_2", 128, 512, 16, 2),    # 标准形态
    ("case_100_512_16_2", 100, 512, 16, 2),    # N 非 32 倍数
    ("case_16_256_8_2", 16, 256, 8, 2),        # 最小规模
    ("case_256_1024_4_4", 256, 1024, 4, 4),    # E=4 边界
    ("case_1024_1024_32_4", 1024, 1024, 32, 4),
    ("case_4096_2048_8_2", 4096, 2048, 8, 2),  # 大 N
    ("case_512_512_8_2", 512, 512, 8, 2),      # 多核切分
    ("case_129_256_16_2", 129, 256, 16, 2),    # N 非 32 倍数且 > 32
]


def fp16_bytes(t_f32: torch.Tensor) -> bytes:
    """float32 张量（FP16 值）-> FP16 原始字节（little-endian，行主序）。"""
    return t_f32.half().view(torch.int16).contiguous().numpy().tobytes()


def gen_case(case_name, N, D, E, K, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    x, W_gate = make_inputs(N, D, E, K, seed=seed)
    idx, w, _ = moe_router(x, W_gate, K)  # FP32 参考（真实 E 列）

    np.save(os.path.join(out_dir, "x_f32.npy"), x.numpy())
    np.save(os.path.join(out_dir, "wgate_f32.npy"), W_gate.numpy())
    with open(os.path.join(out_dir, "x_fp16.bin"), "wb") as f:
        f.write(fp16_bytes(x))
    with open(os.path.join(out_dir, "wgate_fp16.bin"), "wb") as f:
        f.write(fp16_bytes(W_gate))
    np.save(os.path.join(out_dir, "ref_topk_idx.npy"), idx.numpy())
    np.save(os.path.join(out_dir, "ref_topk_weights.npy"), w.numpy())
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"name": case_name, "N": N, "D": D, "E": E, "E_pad": E, "K": K, "seed": seed},
                  f, indent=2)
    print(f"[gen] {case_name}: N={N} D={D} E={E} K={K} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    for name, N, D, E, K in CASES:
        gen_case(name, N, D, E, K, os.path.join(args.out, name))
    print(f"[gen] done, {len(CASES)} cases under {args.out}/")


if __name__ == "__main__":
    main()

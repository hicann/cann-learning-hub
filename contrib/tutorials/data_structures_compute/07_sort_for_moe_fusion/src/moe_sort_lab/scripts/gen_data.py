#!/usr/bin/env python3
"""Generate deterministic MoE routing and token-movement inputs."""
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


TOP_K = 2


def write_bin(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)


def stable_topk(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tokens, experts = logits.shape
    indices = np.empty((tokens, TOP_K), dtype=np.int32)
    probs = np.empty((tokens, TOP_K), dtype=np.float16)
    for token in range(tokens):
        order = np.lexsort((np.arange(experts), -logits[token].astype(np.float32)))
        selected = order[:TOP_K]
        scores = logits[token, selected].astype(np.float32)
        shifted = scores - scores.min() + 1.0
        indices[token] = selected
        probs[token] = (shifted / shifted.sum()).astype(np.float16)
    return indices, probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()
    if args.top_k != TOP_K:
        raise SystemExit("This teaching package currently supports --top_k 2 only")

    rng = np.random.default_rng(202607)
    logits = rng.standard_normal((args.num_tokens, args.num_experts)).astype(np.float16)
    tokens = rng.standard_normal((args.num_tokens, args.hidden_size)).astype(np.float16)
    indices, probs = stable_topk(logits)

    pair_ids = np.arange(args.num_tokens * TOP_K, dtype=np.int32).reshape(args.num_tokens, TOP_K)
    expert_ids = indices.reshape(-1)
    order = np.lexsort((pair_ids.reshape(-1), expert_ids)).astype(np.int32)
    sorted_indices = (order // TOP_K).astype(np.int32)
    sorted_slots = (order % TOP_K).astype(np.int32)
    permuted_tokens = tokens[sorted_indices]
    permuted_probs = probs[sorted_indices, sorted_slots]
    expert_out = permuted_tokens.copy()  # identity expert for a routing-only lab
    unpermute = np.zeros_like(tokens, dtype=np.float32)
    for row, token in enumerate(sorted_indices):
        unpermute[token] += permuted_probs[row].astype(np.float32) * expert_out[row].astype(np.float32)

    root = Path(__file__).resolve().parents[1] / "data" / "input"
    write_bin(root / "logits.bin", logits)
    write_bin(root / "tokens.bin", tokens)
    write_bin(root / "sorted_order.bin", order)
    write_bin(root / "expert_out.bin", expert_out)
    write_bin(root / "ref_topk_indices.bin", indices)
    write_bin(root / "ref_topk_probs.bin", probs)
    write_bin(root / "ref_sorted_indices.bin", sorted_indices)
    write_bin(root / "ref_unpermute_out.bin", unpermute.astype(np.float16))
    (root / "meta.json").write_text(json.dumps({
        "num_tokens": args.num_tokens,
        "num_experts": args.num_experts,
        "hidden_size": args.hidden_size,
        "top_k": TOP_K,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Generated T={args.num_tokens}, E={args.num_experts}, H={args.hidden_size}, K={TOP_K}")


if __name__ == "__main__":
    main()

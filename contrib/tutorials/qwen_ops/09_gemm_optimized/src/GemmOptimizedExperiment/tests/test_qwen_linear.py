import argparse
import sys
from pathlib import Path

import torch
import torch_npu


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch_extension import load_torch_ops


def run_case(batch, seq, hidden, out, atol, rtol):
    torch.manual_seed(0)

    device = "npu:0"
    torch_npu.npu.set_device(device)

    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float32)
    linear = torch.nn.Linear(hidden, out, bias=False, dtype=torch.float32).to(device)

    golden = linear(x)

    x_2d = x.reshape(batch * seq, hidden).contiguous()

    # nn.Linear weight is [out, hidden], but GEMM needs B as [hidden, out].
    weight_for_gemm = linear.weight.t().contiguous()

    actual_2d = torch.ops.gemm_custom.gemm(x_2d, weight_for_gemm)
    actual = actual_2d.reshape(batch, seq, out)

    torch_npu.npu.synchronize()

    diff = (actual - golden).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print(f"shape=({batch}, {seq}, {hidden}) -> ({batch}, {seq}, {out})")
    print(f"max_abs_diff={max_abs:.8e}")
    print(f"mean_abs_diff={mean_abs:.8e}")

    ok = torch.allclose(actual, golden, atol=atol, rtol=rtol)

    if not ok:
        print("actual[0,0,:8] =", actual[0, 0, :8].detach().cpu())
        print("golden[0,0,:8] =", golden[0, 0, :8].detach().cpu())
        print("diff[0,0,:8]   =", diff[0, 0, :8].detach().cpu())

    print("PASS" if ok else "FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", "--in-features", dest="hidden", type=int, default=1024)
    parser.add_argument("--out", "--out-features", dest="out", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    load_torch_ops()

    ok = run_case(args.batch, args.seq, args.hidden, args.out, args.atol, args.rtol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
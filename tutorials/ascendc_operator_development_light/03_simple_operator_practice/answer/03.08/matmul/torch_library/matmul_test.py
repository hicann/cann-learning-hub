import argparse
from pathlib import Path
import sys

import torch
import torch_npu


def load_operator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path(__file__).resolve().parent / "build")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    library = build / "libascendc_matmul.so"
    assert library.is_file(), f"Build the operator first: {library}"
    torch.ops.load_library(str(library))
    return torch.ops.ascendc_matmul.matmul

def main():
    op = load_operator()
    torch.manual_seed(7)
    torch.set_num_threads(16)
    size = 8192
    a = (torch.rand(size, size) * 2 - 1).half()
    bt = (torch.rand(size, size) * 2 - 1).half()
    # CPU FP32 accumulation provides an independent full-matrix reference.
    reference = (a.float() @ bt.float().t()).half()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream), torch.no_grad():
        a_npu, bt_npu = a.npu(), bt.npu()
        output = op(a_npu, bt_npu)
    stream.synchronize()
    actual = output.cpu()
    assert output.shape == (size, size) and output.dtype == torch.float16
    assert output.device == a_npu.device
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(a_npu.cpu(), a, rtol=0, atol=0)
    torch.testing.assert_close(bt_npu.cpu(), bt, rtol=0, atol=0)
    print(f"random: max_error={(actual.float()-reference.float()).abs().max().item():.8g}")
    # A nonsymmetric permutation detects accidentally treating B_T as B.
    permutation = (torch.arange(size) + 17) % size
    bt.zero_()
    bt[torch.arange(size), permutation] = 1
    with torch.npu.stream(stream), torch.no_grad():
        bt_npu.copy_(bt)
        output = op(a_npu, bt_npu)
    stream.synchronize()
    torch.testing.assert_close(output.cpu(), a[:, permutation], rtol=0, atol=0)
    print("B_T layout: exact permutation result passed.")
    for bad_a, bad_b in [(a, bt_npu), (a_npu, bt_npu.t()),
                         (a_npu[:1], bt_npu), (a_npu, bt_npu.float())]:
        try:
            op(bad_a, bad_b)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    print("Matmul: all tests passed.")


if __name__ == "__main__":
    main()

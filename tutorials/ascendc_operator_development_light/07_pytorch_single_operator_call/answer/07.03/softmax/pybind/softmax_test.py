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
    sys.path.insert(0, str(build))
    import ascendc_softmax
    return ascendc_softmax.softmax

def main():
    op = load_operator()
    torch.manual_seed(7)
    for label, x in [
        ("random", torch.randn(128, 128)),
        ("large positive", torch.randn(128, 128) * 5 + 1000),
        ("large negative", torch.randn(128, 128) * 5 - 1000),
        ("constant", torch.full((128, 128), 3.0)),
    ]:
        reference = torch.softmax(x, dim=-1)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream), torch.no_grad():
            x_npu = x.npu()
            output = op(x_npu)
        stream.synchronize()
        actual = output.cpu()
        assert output.shape == x.shape and output.dtype == x.dtype
        assert output.device == x_npu.device
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.sum(-1), torch.ones(128), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x_npu.cpu(), x, rtol=0, atol=0)
        print(f"{label}: max_error={(actual-reference).abs().max().item():.8g}")
    for x in [torch.zeros(128, 128), torch.zeros(128, 128, device="npu", dtype=torch.float16),
              torch.zeros(64, 128, device="npu"), torch.zeros(128, 128, device="npu").t()]:
        try:
            op(x)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    print("Softmax: all tests passed.")


if __name__ == "__main__":
    main()

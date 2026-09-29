from add_modes import add_simd, add_simt
from course_validation import cpu_input, checked_input
import torch
torch.npu.init()

def run():
    for N, tile, blocks in [(64, 64, 1), (1179648, 16384, 36)]:
        for mode, factory in [("SimdVF", add_simd), ("SimtVF", add_simt)]:
            kernel = factory(N, tile, blocks)
            for case in ("random", "zero", "constant", "ramp"):
                a = cpu_input((N,), case)
                b = cpu_input((N,), "ramp").flip(0).contiguous()
                x, z = checked_input(a), checked_input(b)
                out = kernel(x, z); torch.npu.synchronize()
                actual = out.cpu(); torch.testing.assert_close(actual, a+b, rtol=0, atol=0)
                print(f"{mode} N={N} tile={tile} blocks={blocks} {case}: full-output PASS; max_abs_error={(actual-(a+b)).abs().max().item():.8g}", flush=True)
if __name__ == "__main__": run()

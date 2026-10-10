"""Chapter 2.2: inspect and run the T.print debug primitive."""
import torch
import tilelang
import tilelang.ascend.language as T


@T.prim_func
def print_probe(A: T.Tensor((64,), T.float32), C: T.Tensor((64,), T.float32)):
    with T.Kernel(1) as bx:
        work = T.alloc_shared((64,), T.float32)
        T.copy(A[:], work)
        T.print(None, msg="tilelang kernel reached")
        T.print(bx, msg="logical_block")
        # CopyOut follows the debug calls so the output remains the final device write.
        T.copy(work, C[:])


def run() -> None:
    kernel = tilelang.compile(
        print_probe,
        target="ascend",
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    source = kernel.get_kernel_source()
    assert "debug_print" in source
    print("Generated source contains:")
    for line in source.splitlines():
        if "debug_print" in line or "tilelang kernel" in line:
            print(line.strip())

    a = torch.arange(64, device="npu", dtype=torch.float32)
    out = kernel(a)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0.0, atol=0.0)
    print("T.print runtime output is emitted above; tensor output also matches.")
    print("Verification passed!")


if __name__ == "__main__":
    run()

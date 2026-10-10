import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
import torch
import torch_npu

tilelang.disable_cache()

N = 256
TILE = 128

def make_add():
    @T.prim_func
    def add(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(N // TILE) as bx:
            a = T.alloc_shared((TILE,), "float32")
            b = T.alloc_shared((TILE,), "float32")
            c = T.alloc_shared((TILE,), "float32")
            begin = bx * TILE
            source_begin = begin
            T.copy(A[begin:begin + TILE], a)
            T.copy(B[source_begin:source_begin + TILE], b)
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                for j in T.serial(TILE // 64):
                    va = S.vld(a[j * 64])
                    vb = S.vld(b[j * 64])
                    S.vsts(c[j * 64], S.vadd(va, vb, full), full)
            T.copy(c, C[begin:begin + TILE])
    return add


torch.npu.set_device(0)
kernel = tilelang.compile(make_add(), target="ascend", out_idx=-1)
generator = torch.Generator().manual_seed(20260918)
random_a = torch.randint(-32, 33, (N,), generator=generator).float()
random_b = torch.randint(-32, 33, (N,), generator=generator).float()
cases = [
    ("复现输入", random_a, random_b),
    ("全零输入", torch.zeros(N), torch.zeros(N)),
    ("常量输入", torch.full((N,), 2.0), torch.full((N,), -3.0)),
    ("另一组随机输入",
     torch.randint(-32, 33, (N,), generator=generator).float(),
     torch.randint(-32, 33, (N,), generator=generator).float()),
]
for name, a_cpu, b_cpu in cases:
    actual = kernel(a_cpu.npu(), b_cpu.npu()).cpu()
    expected = a_cpu + b_cpu
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"{name}：全部 {N} 个元素通过比较")
print("Verification passed!")

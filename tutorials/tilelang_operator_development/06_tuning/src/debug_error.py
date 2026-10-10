import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
import torch
import torch_npu

tilelang.disable_cache()

N = 256
TILE = 128

def make_add(shift_b=False, debug=False):
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
            source_begin = ((bx + 1) % 2) * TILE if shift_b else begin
            T.copy(A[begin:begin + TILE], a)
            T.copy(B[source_begin:source_begin + TILE], b)
            if debug:
                T.print(a, msg="A_after_copy")
                T.print(b, msg="B_after_copy")
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                for j in T.serial(TILE // 64):
                    va = S.vld(a[j * 64])
                    vb = S.vld(b[j * 64])
                    S.vsts(c[j * 64], S.vadd(va, vb, full), full)
            if debug:
                T.print(c, msg="C_before_writeback")
            T.copy(c, C[begin:begin + TILE])
    return add

torch.npu.set_device(0)
generator = torch.Generator().manual_seed(20260918)
a_cpu = torch.randint(-32, 33, (N,), generator=generator).float()
b_cpu = torch.randint(-32, 33, (N,), generator=generator).float()
a_npu, b_npu = a_cpu.npu(), b_cpu.npu()
torch.testing.assert_close(a_npu.cpu(), a_cpu, rtol=0, atol=0)
torch.testing.assert_close(b_npu.cpu(), b_cpu, rtol=0, atol=0)
print("输入检查：A、B 的设备数据与 CPU 输入完全一致", flush=True)
print("输入形状、类型：", tuple(a_cpu.shape), a_cpu.dtype, flush=True)
bad = tilelang.compile(make_add(shift_b=True, debug=True), target="ascend", out_idx=-1)
actual = bad(a_npu, b_npu).cpu()
expected = a_cpu + b_cpu
mismatches = (actual != expected).nonzero().flatten()
print("错误版本：不一致元素数 =", mismatches.numel(), flush=True)
i = int(mismatches[0])
print(f"首个差异：i={i}, actual={actual[i].item():.0f}, expected={expected[i].item():.0f}", flush=True)
print(f"任务与局部位置：bx={i // TILE}, local={i % TILE}", flush=True)
print(f"应读取的数据：A[{i}]={a_cpu[i].item():.0f}, B[{i}]={b_cpu[i].item():.0f}", flush=True)
print(f"供索引核对：B[128]={b_cpu[128].item():.0f}", flush=True)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)

"""2.4 Add：同一份 kernel 分别通过 JIT 和显式编译运行。"""
import torch
import tilelang
import tilelang.ascend.language as T


def _validate(n: int, num_blocks: int) -> int:
    if n <= 0 or num_blocks <= 0 or n % num_blocks:
        raise ValueError("n、num_blocks 必须为正数，且 n 必须能被 num_blocks 整除")
    tile = n // num_blocks
    if tile % 64:
        raise ValueError(f"tile={tile}，必须是 64 的倍数")
    return tile


def add_program(n: int, num_blocks: int):
    tile = _validate(n, num_blocks)

    @T.prim_func
    def main(
        a: T.Tensor((n,), T.float32),
        b: T.Tensor((n,), T.float32),
        c: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(num_blocks) as bx:
            # 每个逻辑块都有自己的 UB 工作区，只处理自己的 tile。
            a_ub = T.alloc_shared((tile,), T.float32)
            b_ub = T.alloc_shared((tile,), T.float32)
            c_ub = T.alloc_shared((tile,), T.float32)
            begin = bx * tile

            # CopyIn：把当前块负责的两个输入切片搬到 UB。
            T.copy(a[begin : begin + tile], a_ub)
            T.copy(b[begin : begin + tile], b_ub)

            # Compute：每轮从 UB 读取 64 个 float32，在寄存器中相加。
            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in T.serial(tile // 64):
                    a_vec = T.simd.vld(a_ub[i * 64])
                    b_vec = T.simd.vld(b_ub[i * 64])
                    c_vec = T.simd.vadd(a_vec, b_vec, mask)
                    T.simd.vsts(c_ub[i * 64], c_vec, mask)

            # CopyOut：只把当前块的结果写回 C 的对应区间。
            T.copy(c_ub, c[begin : begin + tile])

    return main


@tilelang.jit(out_idx=-1, target="ascend")
def make_add(n: int, num_blocks: int = 1):
    """根据 shape 和逻辑块数量构造可调用的 Add Kernel。"""
    return add_program(n, num_blocks)


def run() -> None:
    # 练习时只需修改这里的 num_blocks，再按顺序重跑 Notebook 两格。
    n, num_blocks = 2048, 2
    _validate(n, num_blocks)
    torch.manual_seed(31)
    a = torch.randn(n, device="npu", dtype=torch.float32)
    b = torch.randn(n, device="npu", dtype=torch.float32)

    # 入口一：调用被 @tilelang.jit 装饰的工厂函数。
    jit_kernel = make_add(n, num_blocks)
    out_jit = jit_kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out_jit, a + b, rtol=0.0, atol=0.0)

    # 入口二：显式编译同一份 add_program。
    compiled = tilelang.compile(add_program(n, num_blocks), target="ascend", out_idx=-1)
    out_compile = compiled(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out_compile, a + b, rtol=0.0, atol=0.0)

    source = compiled.get_kernel_source()
    print("JIT 和 compile 的输出均通过校验；下面是生成源码中的关键行：")
    for line in source.splitlines():
        if "copy_" in line or "simd_vf" in line or "vadd" in line or "vsts" in line:
            print(line.strip())
    print("Verification passed!")


if __name__ == "__main__":
    run()

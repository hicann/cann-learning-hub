"""第 2.5 节完整答案：Mul、SAXPY 和 2-D Add。"""

import torch
import tilelang
import tilelang.ascend.language as T


def _validate_1d(n: int, num_blocks: int) -> int:
    # 当前示例采用均分策略：每个逻辑块处理相同数量的元素。
    if n <= 0 or num_blocks <= 0 or n % num_blocks:
        raise ValueError("n and num_blocks must be positive and n must be divisible by num_blocks")
    tile = n // num_blocks

    # float32 SimdVF 每次处理 64 个元素，因此每块数据量必须是 64 的倍数。
    if tile % 64:
        raise ValueError(f"tile ({tile}) must be a multiple of 64 for float32 SimdVF")
    return tile


def make_mul(n: int = 1024, num_blocks: int = 4):
    """构造一维逐元素乘法 Kernel。"""
    # tile 表示一个逻辑块需要处理的元素数量。
    tile = _validate_1d(n, num_blocks)

    @T.prim_func
    def main(
        # x、y 是输入，out 是由 Kernel 写入的输出。
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        # 启动 num_blocks 份逻辑任务，bx 是当前逻辑块编号。
        with T.Kernel(num_blocks) as bx:
            # 当前块负责区间 [begin, begin + tile)。
            begin = bx * tile

            # 为当前块在 UB 中分配输入和输出工作区。
            x_ub = T.alloc_shared((tile,), T.float32)
            y_ub = T.alloc_shared((tile,), T.float32)
            out_ub = T.alloc_shared((tile,), T.float32)

            # CopyIn：只把当前块负责的数据从 GM 搬到 UB。
            T.copy(x[begin : begin + tile], x_ub)
            T.copy(y[begin : begin + tile], y_ub)

            # Compute：进入向量计算区域。
            with T.SimdVF():
                # pset(32) 为 32 位元素生成一个完整的向量掩码。
                mask = T.simd.pset(32)

                # 每轮处理 64 个 float32 元素，依次覆盖整个 tile。
                for i in T.serial(tile // 64):
                    # 从 UB 载入两个 64-lane 寄存器向量。
                    x_vec = T.simd.vld(x_ub[i * 64])
                    y_vec = T.simd.vld(y_ub[i * 64])

                    # 对应数学公式 out = x * y。
                    out_vec = T.simd.vmul(x_vec, y_vec, mask)

                    # 将乘法结果从寄存器写回 UB。
                    T.simd.vsts(out_ub[i * 64], out_vec, mask)

            # CopyOut：把当前 tile 的结果从 UB 写回 GM 中的对应区间。
            T.copy(out_ub, out[begin : begin + tile])

    return main


def make_saxpy(n: int = 1024, num_blocks: int = 4):
    """构造 out = alpha * x + y，其中 alpha 是单元素 Tensor。"""
    # SAXPY 与 Mul 使用相同的一维分块方式。
    tile = _validate_1d(n, num_blocks)

    @T.prim_func
    def main(
        # alpha 使用 shape=(1,) 的 Tensor 表示一个标量输入。
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.float32),
        alpha: T.Tensor((1,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(num_blocks) as bx:
            # 当前逻辑块仍然处理一个连续的一维区间。
            begin = bx * tile

            # x、y、alpha 和输出都需要对应的 UB 工作区。
            x_ub = T.alloc_shared((tile,), T.float32)
            y_ub = T.alloc_shared((tile,), T.float32)
            alpha_ub = T.alloc_shared((1,), T.float32)
            out_ub = T.alloc_shared((tile,), T.float32)

            # CopyIn：向量切片和标量都先从 GM 搬入 UB。
            T.copy(x[begin : begin + tile], x_ub)
            T.copy(y[begin : begin + tile], y_ub)
            T.copy(alpha[:], alpha_ub)

            with T.SimdVF():
                mask = T.simd.pset(32)

                # BRC_B32 将 alpha_ub[0] 广播到 64 个 float32 lane。
                alpha_vec = T.simd.vld(alpha_ub[0], "BRC_B32")

                # 逐组处理当前 tile，每组包含 64 个 float32 元素。
                for i in T.serial(tile // 64):
                    x_vec = T.simd.vld(x_ub[i * 64])
                    y_vec = T.simd.vld(y_ub[i * 64])

                    # 先计算 alpha * x，再将乘积与 y 相加。
                    scaled_vec = T.simd.vmul(alpha_vec, x_vec, mask)
                    out_vec = T.simd.vadd(scaled_vec, y_vec, mask)

                    # 将这一组 SAXPY 结果写回输出工作区。
                    T.simd.vsts(out_ub[i * 64], out_vec, mask)

            # 当前逻辑块完成后，把结果写回全局输出 Tensor。
            T.copy(out_ub, out[begin : begin + tile])

    return main


def _validate_2d(m: int, n: int, tile_m: int, tile_n: int) -> None:
    # 本例只处理能够被 tile 完整切分的二维矩阵。
    if min(m, n, tile_m, tile_n) <= 0 or m % tile_m or n % tile_n:
        raise ValueError("matrix and tile sizes must be positive and divisible")

    # 每次沿列方向处理 64 个 float32 元素，不在本节处理列尾块。
    if tile_n % 64:
        raise ValueError(f"tile_n ({tile_n}) must be a multiple of 64 for float32 SimdVF")


def make_2d_add(m: int = 16, n: int = 128, tile_m: int = 16, tile_n: int = 64):
    """构造使用矩形 tile 的二维 Add Kernel。"""
    _validate_2d(m, n, tile_m, tile_n)

    # 计算矩阵在行、列方向各自包含多少个 tile。
    row_tiles, col_tiles = m // tile_m, n // tile_n

    @T.prim_func
    def main(
        a: T.Tensor((m, n), T.float32),
        b: T.Tensor((m, n), T.float32),
        c: T.Tensor((m, n), T.float32),
    ):
        # 每个逻辑块负责一个二维 tile。
        with T.Kernel(row_tiles * col_tiles) as bid:
            # 将一维 bid 转换为当前 tile 左上角的行、列坐标。
            row0 = (bid // col_tiles) * tile_m
            col0 = (bid % col_tiles) * tile_n

            # UB 工作区的 shape 与当前二维 tile 完全一致。
            a_ub = T.alloc_shared((tile_m, tile_n), T.float32)
            b_ub = T.alloc_shared((tile_m, tile_n), T.float32)
            c_ub = T.alloc_shared((tile_m, tile_n), T.float32)

            # CopyIn：从 A、B 中截取当前二维 tile 并搬入 UB。
            T.copy(a[row0 : row0 + tile_m, col0 : col0 + tile_n], a_ub)
            T.copy(b[row0 : row0 + tile_m, col0 : col0 + tile_n], b_ub)

            with T.SimdVF():
                mask = T.simd.pset(32)

                # 外层循环依次选择 tile 中的每一行。
                for i in T.serial(tile_m):
                    # 内层循环每次处理当前行中的 64 个 float32 元素。
                    for k in T.serial(tile_n // 64):
                        a_vec = T.simd.vld(a_ub[i, k * 64])
                        b_vec = T.simd.vld(b_ub[i, k * 64])
                        c_vec = T.simd.vadd(a_vec, b_vec, mask)
                        T.simd.vsts(c_ub[i, k * 64], c_vec, mask)

            # CopyOut：将 c_ub 写回 C 中原 tile 对应的位置。
            T.copy(c_ub, c[row0 : row0 + tile_m, col0 : col0 + tile_n])

    return main


def run() -> None:
    # 固定随机种子，使每次运行使用相同的输入数据。
    torch.manual_seed(41)

    # Mul 和 SAXPY 共用同一组一维输入，便于只比较计算公式的变化。
    x = torch.randn(1024, device="npu", dtype=torch.float32)
    y = torch.randn(1024, device="npu", dtype=torch.float32)

    # 第 1 项：编译、执行并校验 Mul。断言通过后才继续 SAXPY。
    mul = tilelang.compile(make_mul(), target="ascend", out_idx=-1)
    mul_out = mul(x, y)
    # 等待设备计算完成，再比较整个输出 Tensor。
    torch.npu.synchronize()
    torch.testing.assert_close(mul_out, x * y, rtol=0.0, atol=0.0)
    print("Mul 校验通过：1024 个元素被分为 4 个 tile")

    # 第 2 项：增加标量 alpha，编译、执行并校验 SAXPY。
    alpha = torch.tensor([0.25], device="npu", dtype=torch.float32)
    saxpy = tilelang.compile(make_saxpy(), target="ascend", out_idx=-1)
    saxpy_out = saxpy(x, y, alpha)
    torch.npu.synchronize()
    torch.testing.assert_close(saxpy_out, alpha[0] * x + y, rtol=0.0, atol=0.0)
    print("SAXPY 校验通过：标量 alpha 参与全部元素计算")

    # 第 3 项：准备二维输入，编译、执行并校验 2-D Add。
    a2 = torch.randn(16, 128, device="npu", dtype=torch.float32)
    b2 = torch.randn(16, 128, device="npu", dtype=torch.float32)
    add2 = tilelang.compile(make_2d_add(), target="ascend", out_idx=-1)
    add2_out = add2(a2, b2)
    torch.npu.synchronize()
    torch.testing.assert_close(add2_out, a2 + b2, rtol=0.0, atol=0.0)
    print("2-D Add 校验通过：(16, 128) 被分为两个 (16, 64) tile")

    # 只有上面三个完整输出都通过比较，程序才会执行到这里。
    print("Verification passed!")


if __name__ == "__main__":
    run()

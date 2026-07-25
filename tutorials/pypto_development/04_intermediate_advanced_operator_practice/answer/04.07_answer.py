# 04.07 章节实践参考答案：动态 Residual + LayerNorm + GELU 近似

# 章节自测答案：
# 1. A
# 2. 真实有效的数据范围
# 3. A
# 4. LayerNorm；GELU
# 5. A

# 6. 编程题参考实现

HIDDEN_SIZE = 128
EPS = 1e-5
GELU_COEFF = 1.702


def residual_norm_gelu_golden(
    x: torch.Tensor,
    residual_input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    residual = x + residual_input
    mean = residual.mean(dim=-1, keepdim=True)
    var = ((residual - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (residual - mean) / torch.sqrt(var + eps)
    scaled = normalized * gamma + beta
    return scaled * torch.sigmoid(GELU_COEFF * scaled)


reset_pypto_notebook_state()


@pypto.frontend.jit(runtime_options={"run_mode": RUN_MODE})
def dynamic_residual_norm_gelu_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, HIDDEN_SIZE], pypto.DT_FP32),
    residual_input: pypto.Tensor([pypto.DYNAMIC, HIDDEN_SIZE], pypto.DT_FP32),
    gamma: pypto.Tensor([HIDDEN_SIZE], pypto.DT_FP32),
    beta: pypto.Tensor([HIDDEN_SIZE], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, HIDDEN_SIZE], pypto.DT_FP32),
):
    tile_b = 8
    batch = x.shape[0]
    pypto.set_vec_tile_shapes(tile_b, HIDDEN_SIZE)

    b_loop = (batch + tile_b - 1) // tile_b
    for b_idx in pypto.loop(b_loop):
        b_offset = b_idx * tile_b
        b_end = pypto.minimum(b_offset + tile_b, batch)
        valid_shape = [b_end - b_offset, HIDDEN_SIZE]

        x_view = pypto.view(x, [tile_b, HIDDEN_SIZE], [b_offset, 0], valid_shape=valid_shape)
        residual_view = pypto.view(
            residual_input,
            [tile_b, HIDDEN_SIZE],
            [b_offset, 0],
            valid_shape=valid_shape,
        )

        residual = x_view + residual_view
        mean = pypto.sum(residual, dim=-1, keepdim=True) / float(HIDDEN_SIZE)
        centered = residual - mean
        var = pypto.sum(centered * centered, dim=-1, keepdim=True) / float(HIDDEN_SIZE)
        normalized = centered / pypto.sqrt(var + EPS)
        scaled = normalized * gamma + beta
        activated = scaled * pypto.sigmoid(GELU_COEFF * scaled)

        pypto.assemble(activated, [b_offset, 0], out)


def main_dynamic_residual_norm_gelu():
    if device == "cpu":
        print("当前环境未执行 NPU 验证；NPU 环境中可执行本模块。")
        return

    for batch in [8, 13]:
        x = torch.randn((batch, HIDDEN_SIZE), dtype=torch.float32, device=device)
        residual_input = torch.randn((batch, HIDDEN_SIZE), dtype=torch.float32, device=device)
        gamma = torch.ones((HIDDEN_SIZE,), dtype=torch.float32, device=device)
        beta = torch.zeros((HIDDEN_SIZE,), dtype=torch.float32, device=device)
        out = torch.empty_like(x)

        dynamic_residual_norm_gelu_kernel(x, residual_input, gamma, beta, out)

        ref = residual_norm_gelu_golden(x, residual_input, gamma, beta)
        max_diff = (out - ref).abs().max().item()
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

        print(f"batch={batch} 验证通过")
        print("输出 shape:", tuple(out.shape))
        print("最大误差:", max_diff)


main_dynamic_residual_norm_gelu()

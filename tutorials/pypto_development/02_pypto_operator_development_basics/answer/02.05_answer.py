# 1. B
# 2. Tensor 计算逻辑
# 3. A
# 4. shape；dtype
# 5. A
# 6. Execute Graph

# 7. 编程题核心代码参考实现：把加法算子改写为 out = (x + y) * 0.5
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def half_add_kernel(x: pypto.Tensor[...], y: pypto.Tensor[...], out: pypto.Tensor[...]):
    pypto.set_vec_tile_shapes(32, 32)
    out[:] = pypto.mul(pypto.add(x, y), 0.5)

def test_half_add_kernel():
    shape = (64, 64)
    device = get_device()
    x = torch.randn(shape, dtype=torch.float, device=device)
    y = torch.randn(shape, dtype=torch.float, device=device)
    out = torch.empty(shape, dtype=torch.float, device=device)
    half_add_kernel(x, y, out)
    torch.testing.assert_close((x + y) * 0.5, out, atol=1e-3, rtol=1e-3)
    print("✓ Test completed successfully")
# 02.05 章节实践参考答案

# 1. B
# 2. Tensor 计算逻辑
# 3. A
# 4. shape；dtype
# 5. A
# 6. Execute Graph

# 7. 编程题参考实现：把加法算子改写为 out = (x + y) * 0.5

import numpy as np
import torch
from numpy.testing import assert_allclose

import pypto


RUN_MODE = pypto.RunMode.SIM


@pypto.frontend.jit(
    run_mode=RUN_MODE,
    input_tensor=[
        pypto.Tensor([1, 4, 1024], pypto.DT_FP32),
        pypto.Tensor([1, 4, 1024], pypto.DT_FP32),
        pypto.Tensor([1, 4, 1024], pypto.DT_FP32),
    ],
)
def half_add_kernel(x, y, out):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out[:] = pypto.mul(pypto.add(x, y), 0.5)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def test_half_add_kernel():
    input_data0 = torch.rand((1, 4, 1024), dtype=torch.float32)
    input_data1 = torch.rand((1, 4, 1024), dtype=torch.float32)
    output_data = torch.empty((1, 4, 1024), dtype=torch.float32)

    half_add_kernel(input_data0, input_data1, output_data)

    golden = (input_data0 + input_data1) * 0.5
    assert_allclose(to_numpy(output_data), to_numpy(golden), rtol=3e-3, atol=3e-3)
    print("half_add_kernel 验证通过")


test_half_add_kernel()

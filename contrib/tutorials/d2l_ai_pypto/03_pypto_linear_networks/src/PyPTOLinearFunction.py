import pypto
import torch

@pypto.frontend.jit()
def linear_forward_kernel(
    X: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    pypto.set_vec_tile_shapes(8, 8)
    out[:] = pypto.matmul(X, w, pypto.DT_FP32) + b

@pypto.frontend.jit()
def linear_backward_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    X: pypto.Tensor([], pypto.DT_FP32),
    w: pypto.Tensor([], pypto.DT_FP32),
    grad_X: pypto.Tensor([], pypto.DT_FP32),
    grad_w: pypto.Tensor([], pypto.DT_FP32),
    grad_b: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    grad_X[:] = pypto.matmul(grad_out, w, pypto.DT_FP32, b_trans=True)
    grad_w[:] = pypto.matmul(X, grad_out, pypto.DT_FP32, a_trans=True)
    pypto.set_vec_tile_shapes(*([8] * len(grad_out.shape)))
    grad_b[:] = pypto.sum(grad_out, dim=0, keepdim=False)

class PyPTOLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, w, b):
        X_c, w_c, b_c = X.contiguous(), w.contiguous(), b.contiguous()
        out = torch.empty((X.shape[0], 1), dtype=X.dtype, device=X.device)
        linear_forward_kernel(X_c, w_c, b_c, out)
        ctx.save_for_backward(X_c, w_c)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, w = ctx.saved_tensors
        grad_out_c = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        grad_X = torch.empty_like(X)
        grad_w = torch.empty_like(w)
        grad_b = torch.empty((1,), dtype=w.dtype, device=w.device)
        linear_backward_kernel(grad_out_c, X, w, grad_X, grad_w, grad_b)
        return grad_X, grad_w, grad_b
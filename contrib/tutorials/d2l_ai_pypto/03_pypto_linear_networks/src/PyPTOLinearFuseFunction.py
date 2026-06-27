import pypto
import torch

@pypto.frontend.jit()
def linear_forward_kernel(
    X: pypto.Tensor([], pypto.DT_FP32),
    W: pypto.Tensor([], pypto.DT_FP32),
    b_2d: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out[:] = pypto.matmul(X, W, pypto.DT_FP32,
                          extend_params={'bias_tensor': b_2d})

@pypto.frontend.jit()
def linear_backward_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    X: pypto.Tensor([], pypto.DT_FP32),
    W: pypto.Tensor([], pypto.DT_FP32),
    grad_X: pypto.Tensor([], pypto.DT_FP32),
    grad_W: pypto.Tensor([], pypto.DT_FP32),
    grad_b_2d: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    grad_X[:] = pypto.matmul(grad_out, W, pypto.DT_FP32, b_trans=True)
    grad_W[:] = pypto.matmul(X, grad_out, pypto.DT_FP32, a_trans=True)
    pypto.set_vec_tile_shapes(16, 16)
    grad_b_2d[:] = pypto.sum(grad_out, dim=0, keepdim=True)

class PyPTOLinearFuseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b):
        X_c, W_c = X.contiguous(), W.contiguous()
        b_2d = b.contiguous().view(1, -1)
        out = torch.empty((X.shape[0], W.shape[1]), dtype=X.dtype, device=X.device)
        linear_forward_kernel(X_c, W_c, b_2d, out)
        ctx.save_for_backward(X_c, W_c)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X, W = ctx.saved_tensors
        grad_out_c = grad_out.contiguous()
        grad_X = torch.empty_like(X)
        grad_W = torch.empty_like(W)
        grad_b_2d = torch.empty((1, W.shape[1]), dtype=W.dtype, device=W.device)
        linear_backward_kernel(grad_out_c, X, W, grad_X, grad_W, grad_b_2d)
        grad_b = grad_b_2d.view(-1)
        return grad_X, grad_W, grad_b
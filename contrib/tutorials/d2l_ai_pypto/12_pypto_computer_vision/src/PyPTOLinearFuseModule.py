import pypto
import torch
from torch import nn

@pypto.frontend.jit()
def linear_forward_kernel(
    X: pypto.Tensor([], pypto.DT_FP32),
    W: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    # bias 不在 kernel 内融合: extend_params bias_tensor 在多核切 K 时结果错误
    # (框架限制, 见 conv2d_aic), 由 host 加法叠加
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out[:] = pypto.matmul(X, W, pypto.DT_FP32)

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
        out = torch.empty((X.shape[0], W.shape[1]), dtype=X.dtype, device=X.device)
        linear_forward_kernel(X_c, W_c, out)
        out = out + b.view(1, -1)   # host 叠加 bias (kernel 融合有框架限制)
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


class PyPTOLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 修复: 原 torch.empty 未初始化 (巨大值/NaN) 导致 softmax 饱和、训练不下降
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        return PyPTOLinearFuseFunction.apply(X, self.weight, self.bias)
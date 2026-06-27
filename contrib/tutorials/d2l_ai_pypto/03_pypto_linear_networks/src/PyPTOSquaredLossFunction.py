import pypto
import torch

@pypto.frontend.jit()
def squared_loss_forward_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(*([8] * len(y_hat.shape)))
    diff = y_hat - y
    out[:] = diff ** 2 / 2

@pypto.frontend.jit()
def squared_loss_backward_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_y_hat: pypto.Tensor([], pypto.DT_FP32),
    grad_y: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(*([8] * len(y_hat.shape)))
    diff = y_hat - y
    grad_y_hat[:] = grad_out * diff
    grad_y[:] = grad_out * (0.0 - diff)

class PyPTOSquaredLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_hat, y):
        y_reshaped = y.reshape(y_hat.shape)
        out = torch.empty_like(y_hat)
        squared_loss_forward_kernel(y_hat, y_reshaped, out)
        ctx.save_for_backward(y_hat, y_reshaped)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        y_hat, y = ctx.saved_tensors
        grad_out_c = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        grad_y_hat = torch.empty_like(y_hat)
        grad_y = torch.empty_like(y)
        squared_loss_backward_kernel(grad_out_c, y_hat, y, grad_y_hat, grad_y)
        return grad_y_hat, None
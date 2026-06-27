import pypto
import torch

@pypto.frontend.jit()
def softmax_forward_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(16, 16)
    out[:] = pypto.softmax(x, dim=-1)

@pypto.frontend.jit()
def softmax_backward_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    y: pypto.Tensor([], pypto.DT_FP32),
    grad_x: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(16, 16)
    grad_x[:] = y * (grad_out - pypto.sum(grad_out * y, dim=-1, keepdim=True))

@pypto.frontend.jit()
def cross_entropy_forward_kernel(
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    y_1d: pypto.Tensor([], pypto.DT_INT32),
    out: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    y_2d = pypto.reshape(y_1d, [-1, 1])
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    selected = pypto.gather(y_hat, -1, y_2d)
    out[:] = pypto.neg(pypto.log(selected))

@pypto.frontend.jit()
def cross_entropy_backward_kernel(
    grad_out: pypto.Tensor([], pypto.DT_FP32),
    y_hat: pypto.Tensor([], pypto.DT_FP32),
    y_1d: pypto.Tensor([], pypto.DT_INT32),
    grad_y_hat: pypto.Tensor([], pypto.DT_FP32),
    num_classes: int,
):
    y_2d = pypto.reshape(y_1d, [-1, 1])
    pypto.set_vec_tile_shapes(8, ((num_classes + 7) // 8) * 8)
    selected = pypto.gather(y_hat, -1, y_2d)
    pypto.set_vec_tile_shapes(8, num_classes)
    one_hot_y = pypto.cast(pypto.one_hot(y_1d, num_classes), pypto.DT_FP32)
    pypto.set_vec_tile_shapes(8, 8)
    grad_y_hat[:] = -(grad_out / selected) * one_hot_y

class PyPTOSoftmaxCrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, num_classes):
        x_c = x.contiguous()
        # softmax
        softmax_out = torch.empty_like(x_c)
        softmax_forward_kernel(x_c, softmax_out)
        # cross_entropy on softmax output
        y_i32_1d = y.to(torch.int32).contiguous()
        ce_out = torch.empty((x.shape[0], 1), dtype=x.dtype, device=x.device)
        cross_entropy_forward_kernel(softmax_out, y_i32_1d, ce_out, num_classes)
        ctx.save_for_backward(softmax_out, y_i32_1d)
        ctx.num_classes = num_classes
        return ce_out.view(-1)

    @staticmethod
    def backward(ctx, grad_out):
        softmax_out, y_i32_1d = ctx.saved_tensors
        num_classes = ctx.num_classes
        # gradient of cross_entropy w.r.t. softmax_out
        grad_out_c = grad_out.view(-1, 1).contiguous()
        grad_ce = torch.empty_like(softmax_out)
        cross_entropy_backward_kernel(
            grad_out_c, softmax_out, y_i32_1d, grad_ce, num_classes
        )
        # gradient of softmax w.r.t. x
        grad_x = torch.empty_like(softmax_out)
        softmax_backward_kernel(grad_ce.contiguous(), softmax_out, grad_x)
        return grad_x, None, None
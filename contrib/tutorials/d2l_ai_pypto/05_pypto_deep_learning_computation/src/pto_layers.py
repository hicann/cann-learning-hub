import pypto
import torch
import torch.nn as nn
import math

# ========== 分块策略选择 ==========
def linear_tile_shapes(in_features, out_features):
    if max(in_features, out_features) <= 64:
        return [16, 16], [32, 64], [16, 16]
    return [32, 32], [64, 64], [64, 64]

def relu_tile_shapes(features):
    if features <= 64:
        return 16, 16
    return 32, 32

# ========== kernel 定义 ==========
def get_pypto_linear_kernel(in_features, out_features):
    m_tile, k_tile, n_tile = linear_tile_shapes(in_features, out_features)

    @pypto.frontend.jit
    def linear_kernel(
            x: pypto.Tensor([], pypto.DT_FP32),
            weight: pypto.Tensor([], pypto.DT_FP32),
            bias: pypto.Tensor([], pypto.DT_FP32),
            out: pypto.Tensor([], pypto.DT_FP32),
        ):
            pypto.set_cube_tile_shapes(m_tile, k_tile, n_tile)
            h = pypto.matmul(x, weight, pypto.DT_FP32, b_trans=True)
            pypto.set_vec_tile_shapes(m_tile[0], n_tile[0])
            out[:] = pypto.add(h, bias)

    return linear_kernel

def get_pypto_relu_kernel(features):
    tile_b, tile_f = relu_tile_shapes(features)

    @pypto.frontend.jit
    def relu_kernel(
        x: pypto.Tensor([], pypto.DT_FP32),
        out: pypto.Tensor([], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(tile_b, tile_f)
        out[:] = pypto.relu(x)
    return relu_kernel

# ========== 自动微分包装 ==========
class PyPTOLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, kernel):
        ctx.save_for_backward(X, weight)
        out = torch.zeros(X.shape[0], weight.shape[0], device=X.device, dtype=X.dtype)
        kernel(X, weight, bias, out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        X, weight = ctx.saved_tensors
        grad_bias = grad_output.sum(dim=0) if grad_output.shape[1] > 0 else None
        grad_weight = grad_output.t() @ X
        grad_X = grad_output @ weight
        return grad_X, grad_weight, grad_bias, None

class PyPTOReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, kernel):
        ctx.save_for_backward(X)
        out = torch.empty_like(X)
        kernel(X, out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (X,) = ctx.saved_tensors
        return grad_output * (X > 0).float(), None

# ========== nn.Module 包装 ==========
class PyPTOLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = "npu:0"
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32, device=self.device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32, device=self.device))
        self.reset_parameters()
        self._kernel = get_pypto_linear_kernel(in_features, out_features)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        if X.shape[-1] != self.in_features:
            raise ValueError(f"期望输入维度 {self.in_features}, 得到 {X.shape[-1]}")
        if X.dtype != self.weight.dtype:
            raise TypeError(f"输入类型 {X.dtype} 必须与权重类型 {self.weight.dtype} 一致")

        # 处理多维输入
        leading_shape = X.shape[:-1]  # X.shape[:-1]：取形状的除最后一个之外的所有维度
        X_2d = X.reshape(-1, self.in_features).contiguous()  # -1 表示"自动计算这个维度的大小"
        out_2d = PyPTOLinearFunction.apply(X_2d, self.weight, self.bias, self._kernel)
        
        return out_2d.reshape(*leading_shape, self.out_features)

class PyPTOReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernels = {} 

    def forward(self, X):
        features = X.shape[-1]
        if features not in self._kernels:
            self._kernels[features] = get_pypto_relu_kernel(features)
        leading_shape = X.shape[:-1]
        X_2d = X.reshape(-1, features).contiguous()
        out_2d = PyPTOReLUFunction.apply(X_2d, self._kernels[features])
        return out_2d.reshape(*leading_shape, features)

class PyPTOLazyLinear(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.device = "npu:0"
        self.weight = nn.UninitializedParameter()
        self.bias = nn.UninitializedParameter()
        self._kernel = None
        self._initialized = False

    def _lazy_init(self, X):
        if not self._initialized:
            self.in_features = X.shape[-1]
            self.weight.materialize((self.out_features, self.in_features), 
                                    dtype=torch.float32, device=self.device)                              
            self.bias.materialize((self.out_features,), 
                                  dtype=torch.float32, device=self.device)
            self.reset_parameters()
            self._kernel = get_pypto_linear_kernel(self.in_features, self.out_features)
            self._initialized = True

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        self._lazy_init(X)
        
        if X.shape[-1] != self.in_features:
            raise ValueError(f"期望输入维度 {self.in_features}, 得到 {X.shape[-1]}")
        if X.dtype != self.weight.dtype:
            raise TypeError(f"输入类型 {X.dtype} 必须与权重类型 {self.weight.dtype} 一致")

        leading_shape = X.shape[:-1]
        X_2d = X.reshape(-1, self.in_features).contiguous()
        out_2d = PyPTOLinearFunction.apply(X_2d, self.weight, self.bias, self._kernel)
        
        return out_2d.reshape(*leading_shape, self.out_features)
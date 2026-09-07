# ====== 工具函数：设备选择、基准测试、BN 融合（内嵌）======
"""算子优化实验共用工具函数。"""

import copy
import time

import torch
import torch.nn as nn


def get_device():
    """优先使用 Ascend NPU，其次 CUDA，最后 CPU。"""
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device("npu:0")
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def sync_device(device):
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def count_modules(model):
    """统计模型中的 Conv2d 与 BatchNorm2d 数量。"""
    conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    return {"conv2d": conv_count, "batchnorm2d": bn_count, "total": conv_count + bn_count}


def benchmark_inference(model, x, device, warmup=5, repeats=50):
    """在指定设备上运行模型推理并返回平均时延与吞吐。"""
    model = model.to(device).eval()
    x = x.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync_device(device)

        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        sync_device(device)
        elapsed = time.perf_counter() - start

    ms_per_iter = elapsed / repeats * 1000
    throughput = repeats * x.size(0) / elapsed
    return {"ms_per_iter": ms_per_iter, "throughput": throughput, "device": str(device)}


def measure_peak_memory(fn, device):
    """运行 fn 并返回设备峰值显存（MB），CPU 上返回 None。"""
    if device.type == "npu":
        torch.npu.reset_peak_memory_stats()
        fn()
        torch.npu.synchronize()
        return torch.npu.max_memory_allocated() / 1024 ** 2
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return None


def fuse_conv_bn_eval(conv, bn):
    """将推理阶段的 BatchNorm 折叠进 Conv2d。"""
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        fused.weight.data.copy_(conv.weight)
        bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.bias)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused.weight.data.mul_(scale.reshape(-1, 1, 1, 1))
        fused.bias.data.copy_((bias - bn.running_mean) * scale + bn.bias)
    return fused


def fuse_model(model):
    """递归折叠模型中所有 Conv+BN 结构（仅推理）。"""
    model = copy.deepcopy(model).eval()
    for name, child in model.named_children():
        if (
            isinstance(child, nn.Sequential)
            and len(child) >= 2
            and isinstance(child[0], nn.Conv2d)
            and isinstance(child[1], nn.BatchNorm2d)
        ):
            fused_conv = fuse_conv_bn_eval(child[0], child[1])
            remaining = [m for m in list(child)[2:]]
            setattr(model, name, nn.Sequential(fused_conv, *remaining))
        elif isinstance(child, nn.Module):
            setattr(model, name, fuse_model(child))
    return model

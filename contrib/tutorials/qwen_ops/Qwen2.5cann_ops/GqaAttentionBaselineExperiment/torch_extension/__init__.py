from pathlib import Path
import ctypes
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch

_LOADED = False


def load_torch_ops():
    global _LOADED
    if _LOADED:
        return
    lib_dir = Path(__file__).resolve().parents[1] / "out" / "lib"
    kernel = lib_dir / "libascendc_gqa_kernels.so"
    register = lib_dir / "libgqa_attention_torch_register.so"
    if not kernel.exists():
        raise FileNotFoundError(f"GQA kernel library not found: {kernel}")
    if not register.exists():
        raise FileNotFoundError(f"Torch register library not found: {register}")
    ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(kernel), mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(str(register))
    _LOADED = True

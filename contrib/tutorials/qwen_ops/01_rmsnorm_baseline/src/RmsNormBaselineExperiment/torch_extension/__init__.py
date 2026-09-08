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

    root = Path(__file__).resolve().parents[1]
    lib_dir = root / "out" / "lib"
    kernel = lib_dir / "librmsnorm_kernels_npu.so"
    register = lib_dir / "librmsnorm_torch_register.so"
    if not kernel.exists():
        raise FileNotFoundError(f"RMSNorm kernel library not found: {kernel}")
    if not register.exists():
        raise FileNotFoundError(f"Torch register library not found: {register}")
    ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(kernel), mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(str(register))
    _LOADED = True

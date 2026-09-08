from pathlib import Path
import ctypes
import torch

_LOADED = False


def load_torch_ops():
    global _LOADED
    if _LOADED:
        return

    root = Path(__file__).resolve().parents[1]
    lib_dir = root / "out" / "lib"
    kernel_lib = next((lib_dir / name for name in (
        "libswiglu_kernels_npu.so", "libascendc_kernels_npu.so"
    ) if (lib_dir / name).exists()), None)
    register_lib = lib_dir / "libswiglu_torch_register.so"

    if kernel_lib is None:
        raise FileNotFoundError(f"AscendC kernel library not found in: {lib_dir}")
    if not register_lib.exists():
        raise FileNotFoundError(f"Torch register library not found: {register_lib}")

    ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(kernel_lib), mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(str(register_lib))
    _LOADED = True

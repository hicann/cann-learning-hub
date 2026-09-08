"""QwenRoPeCustom — import 时自动 LD_LIBRARY_PATH + 禁用 torch_npu"""

import os, sys, glob as _g
from pathlib import Path

_SENTINEL = "_HERMES_LD_OK"
_LOADED = False

def _setup_env():
    if os.environ.get(_SENTINEL): return
    root = Path(__file__).resolve().parents[1]
    out_lib = str(root / "out" / "lib")

    cann = []
    for pat in [
        "/usr/local/Ascend/cann-*/*/lib64", "/usr/local/Ascend/cann-*/lib64",
        os.path.expanduser("~/Ascend/ascend-toolkit/latest/*/lib64"),
        os.path.expanduser("~/Ascend/ascend-toolkit/latest/lib64"),
        os.path.expanduser("~/Ascend/ascend-toolkit/cann-*/lib64"),
        os.path.expanduser("~/Ascend/ascend-toolkit/cann-*/*/lib64"),
    ]:
        for d in sorted(_g.glob(pat), reverse=True):
            if os.path.isdir(d) and d not in cann: cann.append(d)

    torch_lib = ""
    try:
        import torch
        torch_lib = str(Path(torch.__path__[0]) / "lib")
    except Exception: pass

    needed = cann + [torch_lib, out_lib]
    cur = set(os.environ.get("LD_LIBRARY_PATH", "").split(":"))
    missing = [d for d in needed if d and d not in cur]
    if missing:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join(missing + list(cur - {""}))
        env[_SENTINEL] = "1"
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

_setup_env()

def load_torch_ops():
    global _LOADED
    if _LOADED: return
    import torch
    root = Path(__file__).resolve().parents[1]
    so = root / "out" / "lib" / "librope_torch_register.so"
    if not so.exists():
        raise FileNotFoundError(f"{so} not found. Run: bash scripts/build.sh")
    torch.ops.load_library(str(so))
    _LOADED = True

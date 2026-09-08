"""
自动环境设置：检测 CANN + 设置 LD_LIBRARY_PATH，必要时 os.execve 重启。

所有 NPU 测试脚本只需在开头 import 即可：
    from _setup_env import auto_setup
    auto_setup()

此后 CANN 环境已就绪，torch/torch_npu 可直接 import。

原理：
  1. 检查 ASCEND_HOME_PATH 是否已设置
  2. 若未设置，搜索 CANN 安装目录，通过 subprocess 捕获 bash source 后的环境
  3. 合并 LD_LIBRARY_PATH（包含项目的 out/lib + torch lib）
  4. 若环境有变化，os.execve 重启当前进程（哨兵防循环）
"""

import os
import sys
import subprocess
import glob as _g
from pathlib import Path

_SENTINEL = "_NPU_ENV_READY"


def _find_cann_root():
    """搜索 CANN 安装根目录，返回 (cann_root, set_env_script) 或 (None, None)。"""
    candidates = [
        os.path.expanduser("~/Ascend/ascend-toolkit/latest"),
        os.path.expanduser("~/Ascend/ascend-toolkit/cann-*"),
        os.path.expanduser("~/CANN/ascend-toolkit"),
        os.path.expanduser("~/CANN/cann-*"),
        "/usr/local/Ascend/ascend-toolkit/latest",
    ]

    for pat in candidates:
        for d in sorted(_g.glob(pat)):
            if not os.path.isdir(d):
                continue

            # 支持 set_env.sh 和 bin/setenv.bash 两种命名
            for script in ["set_env.sh", "bin/setenv.bash"]:
                sp = os.path.join(d, script)
                if os.path.isfile(sp):
                    return d, sp

    return None, None


def _capture_cann_env(script_path):
    """通过 bash source 捕获 CANN 环境变量。"""
    try:
        result = subprocess.run(
            ["bash", "-c", f"source '{script_path}' >/dev/null 2>&1 && env"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}

        env = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                env[key] = val
        return env
    except Exception:
        return {}


def _find_torch_lib():
    """获取 torch 的 lib 目录。"""
    try:
        import torch
        return str(Path(torch.__path__[0]) / "lib")
    except Exception:
        return ""


def auto_setup():
    """自动设置 CANN 环境。若环境已就绪则直接返回。"""

    if os.environ.get(_SENTINEL) == "1":
        return  # 哨兵：已重启过一次，不再重复

    root = Path(__file__).resolve().parents[1]  # QwenRoPeCustomOpt/
    out_lib = str(root / "out" / "lib")
    torch_lib = _find_torch_lib()
    env_changed = False

    # ── 1. CANN 环境 ──
    asc_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not asc_home or not os.path.isdir(asc_home):
        cann_root, set_script = _find_cann_root()
        if cann_root and set_script:
            cann_env = _capture_cann_env(set_script)
            if cann_env:
                os.environ.update(cann_env)
                env_changed = True

    # ── 2. TORCH_DEVICE_BACKEND_AUTOLOAD ──
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD", "") != "0":
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        env_changed = True

    # ── 3. LD_LIBRARY_PATH ──
    cur_ld = set(os.environ.get("LD_LIBRARY_PATH", "").split(":"))
    need = [out_lib]
    if torch_lib:
        need.append(torch_lib)
    missing = [d for d in need if d and d not in cur_ld]

    if missing:
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing + list(cur_ld - {""}))
        env_changed = True

    # ── 4. 若环境有变化，os.execve 重启 ──
    if env_changed:
        os.environ[_SENTINEL] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

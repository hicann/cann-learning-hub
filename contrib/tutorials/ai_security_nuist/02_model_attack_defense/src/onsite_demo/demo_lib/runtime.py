from __future__ import annotations

from typing import Iterable


def _mode_to_context(ms_mode: str):
    import mindspore.context as context

    normalized = str(ms_mode).upper().strip()
    if normalized == "GRAPH":
        return context.GRAPH_MODE
    if normalized == "PYNATIVE":
        return context.PYNATIVE_MODE
    raise ValueError("ms_mode 只能是 GRAPH 或 PYNATIVE")


def _try_configure(target: str, ms_mode: str) -> None:
    import numpy as np
    import mindspore as ms
    import mindspore.context as context

    context.set_context(mode=_mode_to_context(ms_mode), device_target=target)
    probe = ms.Tensor(np.asarray([1.0], dtype=np.float32), ms.float32) + 1
    _ = probe.asnumpy()


def _candidate_devices(preferred: str, *, allow_cpu_fallback: bool = True) -> Iterable[str]:
    normalized = str(preferred or "auto").strip()
    if normalized.lower() == "auto":
        return ("Ascend", "GPU", "CPU")
    if normalized.upper() == "CPU":
        return ("CPU",)
    if not allow_cpu_fallback:
        return (normalized,)
    return (normalized, "CPU")


def configure_mindspore_device(
    preferred: str = "auto",
    ms_mode: str = "GRAPH",
    *,
    allow_cpu_fallback: bool = True,
) -> str:
    """Configure MindSpore with optional CPU fallback.

    Returns the actual device target. Missing Ascend/GPU never crashes the
    notebook when fallback is allowed; missing MindSpore is reported with
    SKIP_REASON for validation.
    """

    try:
        import mindspore as ms
    except ImportError as exc:
        message = f"SKIP_REASON=当前 Python 环境无法导入 MindSpore：{exc}"
        print(message)
        raise ImportError(message) from exc

    errors: list[str] = []
    candidates = tuple(_candidate_devices(preferred, allow_cpu_fallback=allow_cpu_fallback))
    for index, target in enumerate(candidates):
        try:
            _try_configure(target, ms_mode)
            print(f"MindSpore 版本：{getattr(ms, '__version__', 'unknown')}")
            print(f"MindSpore 运行模式：{str(ms_mode).upper().strip()}")
            print(f"实际 device target：{target}")
            if str(preferred).lower() == "auto" and target != "Ascend":
                print("中文说明：Ascend 当前不可用，已自动 fallback 到可用设备。")
            elif str(preferred).lower() != "auto" and target.upper() == "CPU" and str(preferred).upper() != "CPU":
                print(f"中文说明：指定设备 {preferred} 不可用，已自动 fallback 到 CPU。")
            return target
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{target}: {exc}")
            has_next_device = index < len(candidates) - 1
            if target != "CPU" and has_next_device:
                print(f"中文说明：{target} 不可用，继续尝试下一个设备。")
            elif target != "CPU":
                print(f"中文说明：{target} 不可用，且当前配置禁止 CPU fallback。")

    detail = "\n".join(errors)
    attempted = "/".join(candidates)
    raise RuntimeError(f"MindSpore 设备配置失败，已尝试 {attempted}。\n{detail}")

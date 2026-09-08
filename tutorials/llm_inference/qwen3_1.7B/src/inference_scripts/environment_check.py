"""Dependency checks shared by the Qwen3-8B notebooks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path


def read_pinned_requirements(requirements_path: Path) -> dict[str, str]:
    """Read the exact package pins from the unified requirements file."""
    pinned = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name, separator, expected = line.partition("==")
        if not separator or not name.strip() or not expected.strip():
            raise ValueError(f"统一依赖必须使用精确版本（package==version）：{raw_line}")
        pinned[name.strip()] = expected.strip()
    return pinned


def version_matches(actual: str, expected: str) -> bool:
    """Match exact pins while accepting an installed local suffix when unpinned."""
    return actual == expected if "+" in expected else actual.split("+", 1)[0] == expected


def find_unmet_requirements(pinned: dict[str, str]) -> list[tuple[str, str | None, str]]:
    """Return missing packages and packages whose versions do not match."""
    unmet = []
    for name, expected in pinned.items():
        try:
            actual = installed_version(name)
        except PackageNotFoundError:
            actual = None
        if actual is None or not version_matches(actual, expected):
            unmet.append((name, actual, expected))
    return unmet


def require_unified_environment(requirements_path: Path) -> None:
    """Fail clearly when a later chapter is run before chapter 2 prepares the environment."""
    unmet = find_unmet_requirements(read_pinned_requirements(requirements_path))
    if not unmet:
        print("[依赖检查] 统一环境已就绪，复用第 2 章环境")
        return

    details = ", ".join(
        f"{name}: 当前={actual or '未安装'}, 要求={expected}"
        for name, actual, expected in unmet
    )
    raise RuntimeError(
        "统一环境未就绪："
        f"{details}。请先运行 02_baseline_inference.ipynb 的环境准备单元，"
        "再按章节顺序执行；本章不会自动安装或替换依赖。"
    )

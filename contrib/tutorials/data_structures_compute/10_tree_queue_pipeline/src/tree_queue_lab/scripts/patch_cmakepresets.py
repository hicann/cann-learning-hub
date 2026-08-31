#!/usr/bin/env python3
"""Patch an msopgen CMake preset to the requested Ascend target."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: patch_cmakepresets.py <CMakePresets.json> [target]")
    path = Path(sys.argv[1])
    target = sys.argv[2] if len(sys.argv) > 2 else "ascend910b"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for preset in payload.get("configurePresets", []):
        variables = preset.setdefault("cacheVariables", {})
        variables["ASCEND_COMPUTE_UNIT"] = target
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched {path} with {target}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("usage: patch_cmakepresets.py <CMakePresets.json> [ascend310b|ascend910b]")

path = Path(sys.argv[1])
target = sys.argv[2] if len(sys.argv) > 2 else "ascend910b"
ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
data = json.loads(path.read_text(encoding="utf-8"))

for preset in data.get("configurePresets", []):
    variables = preset.setdefault("cacheVariables", {})
    variables["ASCEND_COMPUTE_UNIT"] = {"type": "STRING", "value": target}
    variables["ASCEND_CANN_PACKAGE_PATH"] = {"type": "PATH", "value": ascend_home}
    variables["ENABLE_CROSS_COMPILE"] = {"type": "BOOL", "value": "False"}
    variables["ASCEND_PACK_SHARED_LIBRARY"] = {"type": "BOOL", "value": "False"}

path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
print(f"Patched {path} for {target}")

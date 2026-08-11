#!/usr/bin/env python3
"""Patch CMakePresets.json for a given target platform.

Usage: patch_cmakepresets.py <CMakePresets.json> [target]
  target: ascend310b | ascend910b (default: ascend910b)
"""
import json, os, sys
from pathlib import Path

if len(sys.argv) < 2:
    print('usage: patch_cmakepresets.py <CMakePresets.json> [target]')
    sys.exit(1)

p = Path(sys.argv[1])
target = sys.argv[2] if len(sys.argv) >= 3 else 'ascend910b'
data = json.loads(p.read_text())
ascend_home = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')

for preset in data.get('configurePresets', []):
    cv = preset.setdefault('cacheVariables', {})
    if 'ASCEND_COMPUTE_UNIT' in cv:
        cv['ASCEND_COMPUTE_UNIT']['value'] = target
    else:
        cv['ASCEND_COMPUTE_UNIT'] = {'type': 'STRING', 'value': target}
    if 'ASCEND_CANN_PACKAGE_PATH' in cv:
        cv['ASCEND_CANN_PACKAGE_PATH']['value'] = ascend_home
    else:
        cv['ASCEND_CANN_PACKAGE_PATH'] = {'type': 'PATH', 'value': ascend_home}
    if 'ENABLE_CROSS_COMPILE' in cv:
        cv['ENABLE_CROSS_COMPILE']['value'] = 'False'
    else:
        cv['ENABLE_CROSS_COMPILE'] = {'type': 'BOOL', 'value': 'False'}
    if 'ASCEND_PACK_SHARED_LIBRARY' in cv:
        cv['ASCEND_PACK_SHARED_LIBRARY']['value'] = 'False'

p.write_text(json.dumps(data, indent=4) + '\n')
print(f'Patched {p} with {target}')

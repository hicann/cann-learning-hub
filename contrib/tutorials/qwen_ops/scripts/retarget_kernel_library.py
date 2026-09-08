#!/usr/bin/env python3
"""Give an AscendC kernel library a per-operator SONAME after installation.

Some CANN 8.5 builds emit every AscendC target as
``libascendc_kernels_npu.so``.  That is harmless for a single operator, but
not when RoPE, RMSNorm and SwiGLU are loaded into the same Python process.
This post-install step copies the generic library under an operator-specific
name and changes the dynamic-string entry in it and its torch register
library.  The replacement names are no longer than the generic name, so the
ELF string table layout is preserved.
"""

import argparse
import shutil
from pathlib import Path


GENERIC = b"libascendc_kernels_npu.so"


def patch_name(path: Path, replacement: bytes) -> None:
    data = path.read_bytes()
    if replacement in data and GENERIC not in data:
        return
    occurrences = data.count(GENERIC)
    if occurrences != 1:
        raise RuntimeError(f"expected one generic SONAME entry in {path}, found {occurrences}")
    padded = replacement + b"\0" * (len(GENERIC) - len(replacement))
    path.write_bytes(data.replace(GENERIC, padded))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib-dir", type=Path, required=True)
    parser.add_argument("--register", required=True)
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()

    kernel_name = args.kernel.encode()
    if len(kernel_name) > len(GENERIC):
        raise ValueError(f"kernel name is longer than {GENERIC.decode()}: {args.kernel}")

    generic = args.lib_dir / GENERIC.decode()
    target = args.lib_dir / args.kernel
    register = args.lib_dir / args.register
    if not register.is_file():
        raise FileNotFoundError(register)

    if generic.is_file():
        shutil.copy2(generic, target)
        patch_name(target, kernel_name)
    elif target.is_file():
        # Newer CANN versions may already emit the operator-specific filename.
        # Only patch it when its embedded SONAME is still the generic name.
        if GENERIC in target.read_bytes():
            patch_name(target, kernel_name)
    else:
        raise FileNotFoundError(
            f"neither generic nor operator-specific kernel exists: {generic}, {target}"
        )

    patch_name(register, kernel_name)
    print(f"kernel ready: {target.name}; register ready: {register.name}")


if __name__ == "__main__":
    main()

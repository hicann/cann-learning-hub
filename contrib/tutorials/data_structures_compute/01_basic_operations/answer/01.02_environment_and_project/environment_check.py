from pathlib import Path
import shutil

TOOLS = ["cmake", "npu-smi", "c++"]


def tool_status(name):
    path = shutil.which(name)
    if path is None:
        return "MISSING"
    return f"FOUND: {Path(path).resolve()}"


if __name__ == "__main__":
    for tool in TOOLS:
        print(f"{tool}: {tool_status(tool)}")

"""项目通用工具函数。

这里集中放一些“不依赖具体模型逻辑”的公共能力，
例如：
1. 项目路径管理；
2. 目录创建；
3. JSON 保存；
4. 随机种子设置。
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "src" / "data"
OUTPUT_ROOT = PROJECT_ROOT / "src" / "outputs"


def ensure_dir(path: Path) -> Path:
    """确保目录存在，不存在就自动创建。"""

    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """统一设置随机种子。"""

    # 这里把 MindSpore 延迟导入，避免“纯文件处理脚本”被框架依赖阻塞。
    import mindspore as ms

    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def timestamp() -> str:
    """生成便于文件命名的时间戳。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: Any, path: Path) -> Path:
    """把对象保存成 UTF-8 编码的 JSON 文件。"""

    ensure_dir(path.parent)

    if is_dataclass(data):
        data = asdict(data)

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    """读取 JSON 文件并返回 Python 对象。"""

    return json.loads(path.read_text(encoding="utf-8"))

"""專案路徑常數。

以本檔案位置往上回推專案根目錄:
``src/mia/utils/paths.py`` → parents[3] 即為 repo 根。
"""

from __future__ import annotations

from pathlib import Path

from mia.distribution.layout import LAYOUT

__all__ = [
    "ASSETS_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "LOG_DIR",
    "MODELS_DIR",
    "PROJECT_ROOT",
]

PROJECT_ROOT: Path = LAYOUT.resources

CONFIG_DIR: Path = PROJECT_ROOT / "config"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
DATA_DIR: Path = LAYOUT.data
MODELS_DIR: Path = PROJECT_ROOT / "models"
LOG_DIR: Path = LAYOUT.logs

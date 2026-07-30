"""專案路徑常數。

以本檔案位置往上回推專案根目錄:
``src/mia/utils/paths.py`` → parents[3] 即為 repo 根。
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ASSETS_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "LOG_DIR",
    "MODELS_DIR",
    "PROJECT_ROOT",
]

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

CONFIG_DIR: Path = PROJECT_ROOT / "config"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
LOG_DIR: Path = PROJECT_ROOT / "logs"

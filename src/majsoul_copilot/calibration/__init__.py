"""牌桌矩形校正 —— 建立整個視覺管線的座標基準。"""

from majsoul_copilot.calibration.letterbox import (
    estimate_background,
    find_content_rect,
    fit_aspect,
)
from majsoul_copilot.calibration.stable import StableCalibrator
from majsoul_copilot.calibration.table import Calibration, TableCalibrator

__all__ = [
    "Calibration",
    "StableCalibrator",
    "TableCalibrator",
    "estimate_background",
    "find_content_rect",
    "fit_aspect",
]

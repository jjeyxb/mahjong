"""Steam 全螢幕的版面。"""

from __future__ import annotations

import numpy as np

from mia.calibration.table import TableCalibrator
from mia.calibration.unity import unity_layout
from mia.capture.base import Frame, WindowInfo
from mia.utils.geometry import Rect, Size
from mia.vision.hud import HudDigits, read_hud


def _frame(width: int, height: int) -> Frame:
    window = WindowInfo(handle=1, title="", owner="", bounds=Rect(0, 0, width, height), pid=1)
    image = np.zeros((height, width, 3), np.uint8)
    return Frame(image=image, window=window, scale=1.0, captured_at=0.0)


def test_the_measured_window_gives_a_sixteen_by_nine_scene() -> None:
    layout = unity_layout(Size(3024, 1898))
    assert layout is not None
    table, panel = layout
    assert abs(table.width / table.height - 16 / 9) < 0.01
    assert panel == Rect(30, 65, 470, 215)


def test_the_same_ratio_scales() -> None:
    layout = unity_layout(Size(1512, 949))
    assert layout is not None
    assert layout[0] == Rect(-4, 51, 1520, 855)


def test_an_unmeasured_ratio_is_refused() -> None:
    assert unity_layout(Size(1512, 844)) is None
    assert unity_layout(Size(2560, 1440)) is None


def test_the_calibrator_uses_it_only_for_steam() -> None:
    frame = _frame(3024, 1898)
    steam = TableCalibrator(unity=True).calibrate(frame)
    assert steam.panel == Rect(30, 65, 470, 215)
    assert steam.source == "canvas"
    assert TableCalibrator().calibrate(frame).panel is None


def test_the_sticks_follow_the_panel() -> None:
    """面板移到別處,本場與供託要跟著切,否則讀到的是別的東西。"""
    digits = HudDigits.load()
    image = np.zeros((1898, 3024, 3), np.uint8)
    table = Rect(-8, 102, 3040, 1710)
    assert read_hud(image, table, digits, panel=Rect(30, 65, 470, 215)).honba is None

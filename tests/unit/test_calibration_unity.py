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
    """Mac 一般視窗含標題列,比例 1.79 只差 16:9 約 0.8%,不能當成整片場景。"""
    assert unity_layout(Size(1512, 844)) is None
    assert unity_layout(Size(1600, 1200)) is None


def test_a_sixteen_by_nine_screen_is_the_whole_scene() -> None:
    """Windows 全螢幕 2560x1440 實測:整張就是牌桌,面板在標準位置。"""
    assert unity_layout(Size(2560, 1440)) == (Rect(0, 0, 2560, 1440), None)
    assert unity_layout(Size(1920, 1080)) == (Rect(0, 0, 1920, 1080), None)


def test_a_sixteen_by_nine_steam_screen_ignores_the_browser_canvas() -> None:
    """選過網頁的畫布也不能蓋過去。

    畫布是 CSS 像素,換算要乘 DPI 縮放;Unity 畫的是實體像素。125% 螢幕上原本算成
    3200px 去對 2560px 的影像,整條畫面辨識都起不來。
    """
    from mia.calibration.canvas import Canvas, CanvasChoice

    frame = Frame(
        image=np.zeros((1440, 2560, 3), np.uint8),
        window=WindowInfo(handle=1, title="", owner="", bounds=Rect(0, 0, 2048, 1152), pid=1),
        scale=1.25,
        captured_at=0.0,
    )
    calibrator = TableCalibrator(unity=True, canvas=CanvasChoice(Canvas.parse("2560x1440")))
    calibration = calibrator.calibrate(frame)
    assert calibration.table_rect == Rect(0, 0, 2560, 1440)
    assert calibration.panel is None


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
